"""Control centroids and the CYTAP input.

Notation (per cell i of donor d and cell type k):
    z_i       cell embedding (e.g. State SE-600M, ``obsm['X_state']``)
    c_k       control centroid of cell type k, pooled over donors
    c_{d,k}   control centroid of donor d, cell type k

    CYTAP input (Comp) = [ z_i - (c_{d,k} - c_k) ; z_i - c_{d,k} ]
                         (donor-adjusted embedding ; deviation from the matched control)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_SINGLE = "__all__"


@dataclass
class Centroids:
    """Row-matched centroids for the kept cells.

    keep      bool mask over input cells (cells that receive a prediction)
    ck        (n_celltypes, d) table of c_k
    cdk       (n_groups, d) table of c_{d,k}
    ct_idx    row of ``ck`` for each kept cell
    grp_idx   row of ``cdk`` for each kept cell
    source    number of groups whose c_{d,k} came from own controls,
              counterfactual cells, or the pooled fallback
    dropped   number of input cells left out: missing donor / cell-type label,
              excluded cell type, donor x cell-type group without controls
    """

    keep: np.ndarray
    ck: np.ndarray
    cdk: np.ndarray
    ct_idx: np.ndarray
    grp_idx: np.ndarray
    source: dict
    dropped: dict = field(default_factory=dict)

    def rows(self, sl=slice(None)):
        return self.ck[self.ct_idx[sl]], self.cdk[self.grp_idx[sl]]


def _labels(values, n) -> pd.Series:
    if values is None:
        return pd.Series([_SINGLE] * n)
    return pd.Series(np.asarray(values, dtype=object))


def _is_valid(s: pd.Series) -> np.ndarray:
    u = s.astype(str).str.upper()
    return (s.notna() & ~u.isin(["NA", "NAN"])).to_numpy()


def group_means(X, codes, mask, n):
    """Mean of X over rows with ``mask`` for each code in [0, n).

    Rows of each group are averaged in their original order, so results are
    identical to ``X[np.where(...)[0]].mean(axis=0)``.
    """
    means = np.zeros((n, X.shape[1]), dtype=X.dtype)
    has = np.zeros(n, dtype=bool)
    sel = np.flatnonzero(mask & (codes >= 0))
    if sel.size == 0:
        return means, has
    order = np.argsort(codes[sel], kind="stable")
    sel = sel[order]
    for idx in np.split(sel, np.flatnonzero(np.diff(codes[sel])) + 1):
        k = codes[idx[0]]
        means[k] = X[idx].mean(axis=0)
        has[k] = True
    return means, has


def compute_centroids(
    X: np.ndarray,
    is_control: np.ndarray,
    donor=None,
    celltype=None,
    counterfactual: tuple | None = None,
    fallback: bool = False,
    exclude_celltypes=None,
) -> Centroids:
    """Match each cell to c_k and c_{d,k}.

    Parameters
    ----------
    X : (n, d) embeddings of all cells, controls included.
    is_control : (n,) bool, True for control cells.
    donor : (n,) donor labels, or None for pooled matching (c_{d,k} = c_k).
    celltype : (n,) cell-type labels, or None for a single cell type.
    counterfactual : optional ``(X_cf, donor_cf, celltype_cf)``; predicted
        control cells used for groups without real control cells.
    fallback : use c_k for groups that have neither real nor counterfactual
        controls. If False those groups are dropped.
    exclude_celltypes : cell types to drop before matching.

    Cells with missing labels ('NA', 'nan') are dropped. A remaining cell type
    without any real control cell raises ValueError; drop it with
    ``exclude_celltypes``.
    """
    X = np.asarray(X)
    n = X.shape[0]
    is_control = np.asarray(is_control, dtype=bool)
    d_s, c_s = _labels(donor, n), _labels(celltype, n)
    valid = _is_valid(d_s) & _is_valid(c_s)
    n_missing, n_excluded = int((~valid).sum()), 0
    if exclude_celltypes:
        excl = [str(c) for c in exclude_celltypes]
        unknown = sorted(set(excl) - set(c_s.astype(str)))
        if unknown:
            warnings.warn(f"exclude_celltypes not found in data: {unknown}")
        is_excl = c_s.astype(str).isin(excl).to_numpy()
        n_excluded = int((valid & is_excl).sum())
        valid &= ~is_excl

    c_codes, c_uni = pd.factorize(c_s.astype(str))
    d_codes, d_uni = pd.factorize(d_s.astype(str))
    nc, nd = len(c_uni), len(d_uni)
    g_codes = d_codes * nc + c_codes
    ctrl = valid & is_control

    ck, has_ck = group_means(X, c_codes, ctrl, nc)
    cdk, has_own = group_means(X, g_codes, ctrl, nd * nc)

    has_cf = np.zeros(nd * nc, dtype=bool)
    if counterfactual is not None:
        Xcf, dcf, ccf = counterfactual
        Xcf = np.asarray(Xcf)
        if Xcf.shape[1] != X.shape[1]:
            raise ValueError(f"counterfactual dim {Xcf.shape[1]} != embedding dim {X.shape[1]}")
        dcf_codes = pd.Index(d_uni).get_indexer(_labels(dcf, len(Xcf)).astype(str))
        ccf_codes = pd.Index(c_uni).get_indexer(_labels(ccf, len(Xcf)).astype(str))
        gcf = np.where((dcf_codes >= 0) & (ccf_codes >= 0), dcf_codes * nc + ccf_codes, -1)
        cf_means, has_cf = group_means(Xcf, gcf, np.ones(len(Xcf), bool), nd * nc)
        use_cf = has_cf & ~has_own
        cdk[use_cf] = cf_means[use_cf]  # cast to X.dtype

    grp_ct = np.arange(nd * nc) % nc
    use_fb = ~has_own & ~has_cf & has_ck[grp_ct]
    if not fallback:
        use_fb[:] = False
    cdk[use_fb] = ck[grp_ct[use_fb]]
    has_cdk = has_own | has_cf | use_fb

    no_ctrl_ct = valid & ~has_ck[c_codes]
    if no_ctrl_ct.any():
        counts = c_s[no_ctrl_ct].astype(str).value_counts()
        detail = ", ".join(f"{k} ({v} cells)" for k, v in counts.items())
        raise ValueError(
            f"No control cells for cell type(s): {detail}. "
            f"Remove them with exclude_celltypes={list(counts.index)}."
        )
    keep = valid & has_cdk[g_codes]
    n_drop_grp = int((valid & has_ck[c_codes] & ~has_cdk[g_codes]).sum())
    if n_drop_grp:
        warnings.warn(f"Dropped {n_drop_grp} cells in donor x cell-type groups without control cells")

    present = np.unique(g_codes[keep])
    source = {
        "own": int(has_own[present].sum()),
        "counterfactual": int((has_cf & ~has_own)[present].sum()),
        "pooled_fallback": int(use_fb[present].sum()),
    }
    dropped = {"missing_label": n_missing, "excluded_celltype": n_excluded, "no_control_group": n_drop_grp}
    return Centroids(keep, ck, cdk, c_codes[keep], g_codes[keep], source, dropped)


def compose(z, ck, cdk) -> np.ndarray:
    """CYTAP input [z - (c_{d,k} - c_k) ; z - c_{d,k}] from row-matched z, c_k, c_{d,k}."""
    return np.concatenate([z - (cdk - ck), z - cdk], axis=1)

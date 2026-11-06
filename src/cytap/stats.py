"""Downstream statistics on CYTAP predictions.

All functions read ``obsm[f'{key}_logits']`` (or ``obsm[f'{key}_probs']``) and
``uns[f'{key}_classes']`` written by :func:`cytap.predict`. Logits are
log(p / (1 - p)) with p clipped to [eps, 1 - eps]; when ``obsm[f'{key}_logits']``
is missing (older outputs) or another ``eps`` is requested they are computed
from the probabilities, which gives identical values.
"""
from __future__ import annotations

from typing import Sequence

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import linregress, spearmanr, ttest_ind, ttest_rel


LOGIT_EPS = 1e-6


# ---------------------------------------------------------------- helpers
def _probs(adata: AnnData, key: str):
    return np.asarray(adata.obsm[f"{key}_probs"]), [str(c) for c in adata.uns[f"{key}_classes"]]


def _logit(p, eps=LOGIT_EPS):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _logits(adata: AnnData, key: str, eps: float = LOGIT_EPS):
    """Stored logits (obsm[f'{key}_logits']) or logits computed from the probabilities."""
    classes = [str(c) for c in adata.uns[f"{key}_classes"]]
    stored = f"{key}_logits"
    if stored in adata.obsm and eps == LOGIT_EPS:
        return np.asarray(adata.obsm[stored]), classes
    return _logit(np.asarray(adata.obsm[f"{key}_probs"]), eps), classes


def _values(adata, key, value, eps):
    if value == "logit":
        return _logits(adata, key, eps)
    if value == "prob":
        return _probs(adata, key)
    raise ValueError("value must be 'logit' or 'prob'")


def _subset(adata, celltype_key, celltype):
    if celltype is None:
        return adata
    if celltype_key is None:
        raise ValueError("celltype requires celltype_key")
    cts = [celltype] if isinstance(celltype, str) else list(celltype)
    sub = adata[adata.obs[celltype_key].isin(cts)]
    if sub.n_obs == 0:
        raise ValueError(f"no cells with {celltype_key} in {cts}")
    return sub


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (same as statsmodels 'fdr_bh')."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = p[order] / (np.arange(1, len(p) + 1) / len(p))
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    adj[adj > 1] = 1
    out = np.empty_like(adj)
    out[order] = adj
    return out


# ---------------------------------------------------------------- tables
def logits(adata: AnnData, key: str = "pred", eps: float = LOGIT_EPS) -> pd.DataFrame:
    """Cells x cytokines logit table."""
    lg, classes = _logits(adata, key, eps)
    return pd.DataFrame(lg, index=adata.obs_names, columns=classes)


def pseudobulk(
    adata: AnnData,
    donor_key: str,
    condition_key: str | None = None,
    celltype_key: str | None = None,
    *,
    cytokines: Sequence[str] | None = None,
    conditions: Sequence[str] | None = None,
    require_all_conditions: bool = False,
    value: str = "logit",
    stat: str = "median",
    key: str = "pred",
    eps: float = LOGIT_EPS,
) -> pd.DataFrame:
    """Per (cell type, donor, condition) median logit (or probability).

    conditions : keep only these conditions, in this order (ordered categorical).
    require_all_conditions : keep only cell types observed in every condition.
    """
    vals, classes = _values(adata, key, value, eps)
    cols = list(cytokines) if cytokines is not None else classes
    idx = [classes.index(c) for c in cols]
    df = pd.DataFrame(vals[:, idx], columns=cols)
    keys = [k for k in (celltype_key, donor_key, condition_key) if k]
    for k in keys:
        df[k] = adata.obs[k].values
    if conditions is not None:
        if condition_key is None:
            raise ValueError("conditions requires condition_key")
        df = df[df[condition_key].isin(conditions)].copy()
        df[condition_key] = pd.Categorical(df[condition_key], categories=list(conditions), ordered=True)
    out = df.groupby(keys, observed=True)[cols].agg(stat).reset_index().dropna()
    if require_all_conditions and celltype_key and condition_key:
        n_expected = len(conditions) if conditions is not None else out[condition_key].nunique()
        n_obs = out.groupby(celltype_key, observed=True)[condition_key].nunique()
        out = out[out[celltype_key].isin(n_obs[n_obs == n_expected].index)].copy()
    for k in keys:
        if isinstance(out[k].dtype, pd.CategoricalDtype):
            out[k] = out[k].cat.remove_unused_categories()
    return out.reset_index(drop=True)


def paired_delta(
    adata: AnnData,
    cytokine: str,
    condition_key: str,
    baseline: str,
    targets: str | Sequence[str],
    donor_key: str,
    celltype_key: str | None = None,
    *,
    celltype=None,
    value: str = "logit",
    key: str = "pred",
    eps: float = LOGIT_EPS,
) -> pd.DataFrame:
    """Within-donor change of the median logit: target - baseline.

    Returns columns ``group`` (cell type or 'All Cells'), ``donor``,
    ``condition`` (target) and ``delta``; only donors with both conditions.
    """
    targets = [targets] if isinstance(targets, str) else list(targets)
    sub = _subset(adata, celltype_key, celltype)
    vals, classes = _values(sub, key, value, eps)
    df = pd.DataFrame({
        "group": sub.obs[celltype_key].values if celltype_key else ["All Cells"] * sub.n_obs,
        "donor": sub.obs[donor_key].values,
        "condition": sub.obs[condition_key].values,
        "value": vals[:, classes.index(cytokine)],
    })
    df = df[df["condition"].isin([baseline] + targets)]
    med = df.groupby(["group", "donor", "condition"], observed=True)["value"].median().reset_index().dropna()
    wide = med.pivot(index=["group", "donor"], columns="condition", values="value")
    if baseline not in wide.columns:
        raise ValueError(f"baseline '{baseline}' not found")
    wide = wide.dropna(subset=[baseline])
    parts = []
    for t in targets:
        if t not in wide.columns:
            continue
        d = (wide[t] - wide[baseline]).dropna().reset_index(name="delta")
        d["condition"] = t
        parts.append(d)
    if not parts:
        raise ValueError("no donor has both baseline and target conditions")
    out = pd.concat(parts, ignore_index=True)[["group", "donor", "condition", "delta"]]
    out["condition"] = pd.Categorical(out["condition"], categories=targets, ordered=True)
    return out


# ---------------------------------------------------------------- tests
def compare_conditions(
    adata: AnnData,
    condition_key: str,
    ctrl: str,
    stim: str,
    *,
    donor_key: str | None = None,
    celltype_key: str | None = None,
    celltype=None,
    paired: bool = True,
    alternative: str = "greater",
    key: str = "pred",
    eps: float = LOGIT_EPS,
) -> pd.DataFrame:
    """Per-cytokine comparison of ``stim`` vs ``ctrl``.

    With more than one donor, cells are summarised as the per-donor median
    logit (and probability) and donors are the replicates: paired t-test over
    donors present in both conditions (``paired=True``) or Welch's t-test.
    The default ``alternative="greater"`` is one-sided (higher logit in ``stim``),
    i.e. it tests for induced cytokine responses; use ``"two-sided"`` for both
    directions. With one donor (or no ``donor_key``) only effect sizes are
    reported (p = 1) to avoid pseudo-replication.

    Columns
    -------
    mean_logit_stim / mean_logit_ctrl : mean over donors of the median logit
    delta_logit : mean_logit_stim - mean_logit_ctrl (effect size)
    log2fc_prob : log2 ratio of the mean median probabilities (+1e-9)
    pval, fdr : t-test (``alternative``) and Benjamini-Hochberg FDR over cytokines
    n_stim, n_ctrl : number of donors (or cells) per group
    """
    sub = _subset(adata, celltype_key, celltype)
    probs, classes = _probs(sub, key)
    lg, _ = _logits(sub, key, eps)
    cond = sub.obs[condition_key].values
    n_donors = sub.obs[donor_key].nunique() if donor_key else 0

    if n_donors > 1:
        level = "donor"
        frames = []
        for mat in (lg, probs):
            df = pd.DataFrame(mat, columns=classes)
            df["_donor"], df["_cond"] = sub.obs[donor_key].values, cond
            frames.append(df.groupby(["_donor", "_cond"], observed=True).median().reset_index().dropna())
        ml, mp = frames
        if paired:
            donors = sorted(set(ml.loc[ml["_cond"] == stim, "_donor"]) & set(ml.loc[ml["_cond"] == ctrl, "_donor"]))
            if len(donors) < 2:
                raise ValueError("paired test needs at least 2 donors with both conditions")

            def grab(m, c):
                return m[m["_cond"] == c].set_index("_donor").loc[donors]
        else:
            def grab(m, c):
                return m[m["_cond"] == c]
        Ls, Lc, Ps, Pc = grab(ml, stim), grab(ml, ctrl), grab(mp, stim), grab(mp, ctrl)
    else:
        level, paired = "cell", False
        Ls, Lc = (pd.DataFrame(lg[cond == c], columns=classes) for c in (stim, ctrl))
        Ps, Pc = (pd.DataFrame(probs[cond == c], columns=classes) for c in (stim, ctrl))

    rows, pvals = [], []
    for c in classes:
        s, k = Ls[c].to_numpy(), Lc[c].to_numpy()
        ps, pc = Ps[c].to_numpy(), Pc[c].to_numpy()
        s, k, ps, pc = s[~np.isnan(s)], k[~np.isnan(k)], ps[~np.isnan(ps)], pc[~np.isnan(pc)]
        if level == "cell" or len(s) < 2 or len(k) < 2 or (np.var(s) == 0 and np.var(k) == 0):
            p = 1.0
        elif paired:
            p = ttest_rel(s, k, alternative=alternative).pvalue
        else:
            p = ttest_ind(s, k, alternative=alternative, equal_var=False).pvalue
        pvals.append(p)
        rows.append({
            "cytokine": c,
            "mean_logit_stim": np.mean(s),
            "mean_logit_ctrl": np.mean(k),
            "delta_logit": np.mean(s) - np.mean(k),
            "log2fc_prob": np.log2((np.mean(ps) + 1e-9) / (np.mean(pc) + 1e-9)),
            "n_stim": len(s),
            "n_ctrl": len(k),
        })
    out = pd.DataFrame(rows)
    out["pval"] = np.nan_to_num(np.asarray(pvals, float), nan=1.0)
    out["fdr"] = bh_fdr(out["pval"])
    out.attrs.update(level=level, paired=paired, alternative=alternative, stim=stim, ctrl=ctrl)
    return out


def severity_trend(
    adata: AnnData,
    condition_key: str,
    order: Sequence[str],
    donor_key: str,
    *,
    celltype_key: str | None = None,
    celltype=None,
    key: str = "pred",
    eps: float = LOGIT_EPS,
) -> pd.DataFrame:
    """Monotonic trend of donor-level logits across ordered conditions.

    Conditions are scored 0..len(order)-1; cells in other conditions are
    ignored. Each donor is summarised by its median logit and is one
    observation, so each donor should belong to a single condition (e.g. one
    sample per donor); a warning is given otherwise. Returns Spearman rho and
    P, linear-regression slope, BH FDR and the number of donors.
    """
    sub = _subset(adata, celltype_key, celltype)
    raw = pd.Series(sub.obs[condition_key].astype(str).to_numpy())
    cond = pd.Categorical(raw.where(raw.isin(order)), categories=list(order), ordered=True)
    n = pd.Series(sub.obs[donor_key].values).groupby(cond, observed=False).nunique()
    if (n == 0).any():
        raise ValueError(f"no donors for: {list(n.index[n == 0])}")
    score = cond.codes.astype(float)
    keep = score >= 0
    lg, classes = _logits(sub, key, eps)
    df = pd.DataFrame(lg[keep], columns=classes)
    df["_donor"], df["_score"] = sub.obs[donor_key].values[keep], score[keep]
    dm = df.groupby(["_donor", "_score"], observed=True).median().reset_index()
    multi = dm["_donor"].duplicated().sum()
    if multi:
        warnings.warn(f"{multi} donors appear in more than one condition; their donor x condition medians "
                      "are treated as independent observations")

    rows = []
    for c in classes:
        rho, pv = spearmanr(dm["_score"], dm[c])
        rows.append({"cytokine": c, "rho": rho, "slope": linregress(dm["_score"], dm[c]).slope, "pval": pv})
    out = pd.DataFrame(rows)
    ok = out["pval"].notna()
    out["fdr"] = np.nan
    if ok.any():
        out.loc[ok, "fdr"] = bh_fdr(out.loc[ok, "pval"])
    out["n_donors"] = dm["_donor"].nunique()
    return out

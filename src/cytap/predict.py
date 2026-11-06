"""Cytokine-response inference on query scRNA-seq data."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, Sequence, Union

import numpy as np
import torch
from anndata import AnnData
from scipy.sparse import issparse

from .features import compose, compute_centroids
from .model import load_model
from .stats import _logit

if TYPE_CHECKING:
    from .counterfactual import CellOT

Design = Literal["paired", "unpaired"]
Reference = Literal["pooled", "counterfactual"]


@torch.no_grad()
def predict_proba(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Softmax probabilities for a ready-made input matrix."""
    device = next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[i : i + batch_size], dtype=np.float32)).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.vstack(out)


def _dense(X):
    return X.toarray() if issparse(X) else np.asarray(X)


def predict(
    adata: AnnData,
    model,
    *,
    design: Design,
    control_key: str,
    control_label: str,
    donor_key: str | None = None,
    celltype_key: str | None = None,
    reference: Reference | None = None,
    counterfactual: Union[AnnData, Sequence[AnnData], "CellOT", str, None] = None,
    cf_embed_key: str | None = None,
    cellot_kwargs: dict | None = None,
    exclude_celltypes: Sequence[str] | None = None,
    embed_key: str | None = None,
    batch_size: int = 4096,
    key_added: str = "pred",
) -> AnnData:
    """Predict per-cell probabilities for the cytokines in ``model.classes``.

    Query designs
    -------------
    design="paired"      every donor has its own control cells: c_{d,k} from the donor's
                         control cells of the same cell type. Donor x cell-type groups
                         without controls are dropped. Needs ``donor_key``.
    design="unpaired"    control cells come from other donors (e.g. healthy volunteers vs
                         patients). The control reference is set by ``reference``:
        reference="pooled" (default)
                         c_{d,k} = c_k, the control centroid pooled over donors, so the
                         input is [z ; z - c_k].
        reference="counterfactual"
                         donor-matched counterfactual controls: donors with real controls
                         use them; other donors use counterfactual control cells of the same
                         donor x cell type (``counterfactual``, default CellOT); c_k is the
                         fallback. Needs ``donor_key``.
    c_k always comes from real control cells.

    Parameters
    ----------
    adata : query cells with the State embedding in ``obsm[embed_key]`` (default ``X_state``).
    model : path to a checkpoint or a model from :func:`cytap.load_model`.
    design : ``"paired"`` or ``"unpaired"`` (see above).
    control_key, control_label : ``obs`` column with the condition and its control value;
        control cells are scored too.
    donor_key : ``obs`` column with the donor (sample) of each cell.
    celltype_key : ``obs`` column with cell types; omit for a single cell type. Centroids are
        computed per cell type.
    reference : unpaired control reference, ``"pooled"`` (default) or ``"counterfactual"``.
    counterfactual : with ``reference="counterfactual"``, one of
        - ``"cellot"`` (default): fit CellOT maps (each non-control condition -> control) on
          ``adata`` with :meth:`cytap.counterfactual.CellOT.fit` and transport the cells that
          lack own controls. ``cellot_kwargs`` go to ``CellOT.fit``; add
          ``save="file.pt"`` to keep the maps for reuse.
        - a fitted :class:`cytap.counterfactual.CellOT` (from ``fit``, ``load`` or ``from_cellot``).
        - AnnData (or list) of predicted control cells with ``donor_key`` and
          ``celltype_key`` in ``.obs``; embedding in ``.X`` unless ``cf_embed_key``.
    exclude_celltypes : cell types (values of ``celltype_key``) to leave out.
        Every remaining cell type must contain control cells, otherwise a
        ValueError lists the cell types to exclude.

    Returns
    -------
    Copy of ``adata`` restricted to predicted cells, with
    ``obsm[f'{key_added}_probs']``, ``obsm[f'{key_added}_logits']`` (used by ``cytap.stats``),
    ``uns[f'{key_added}_classes']`` (column order) and run settings in ``uns['cytap']``.
    """
    if isinstance(model, str):
        model = load_model(model)
    cfg = model.config
    if cfg.get("representation") != "comp":
        raise ValueError(f"predict() uses the CYTAP (Comp) input; this model was trained on "
                         f"'{cfg.get('representation')}' input")
    embed_key = embed_key or cfg["embedding_key"]
    if embed_key not in adata.obsm:
        raise KeyError(f"obsm['{embed_key}'] not found")

    if design not in ("paired", "unpaired"):
        raise ValueError(f"design must be 'paired' or 'unpaired', got {design!r}")
    if design == "paired":
        if reference is not None:
            raise ValueError("reference is only used with design='unpaired' (paired data use each donor's own controls)")
        mode = "paired"
    else:
        reference = reference or "pooled"
        if reference not in ("pooled", "counterfactual"):
            raise ValueError(f"reference must be 'pooled' or 'counterfactual', got {reference!r}")
        mode = reference
    if mode != "counterfactual" and (counterfactual is not None or cellot_kwargs or cf_embed_key):
        raise ValueError("counterfactual, cellot_kwargs and cf_embed_key need design='unpaired', reference='counterfactual'")
    if mode == "counterfactual" and counterfactual is None:
        counterfactual = "cellot"
    if mode in ("paired", "counterfactual") and donor_key is None:
        raise ValueError(f"{'design=paired' if mode == 'paired' else 'reference=counterfactual'} requires donor_key")
    if exclude_celltypes and celltype_key is None:
        raise ValueError("exclude_celltypes requires celltype_key")

    X = np.asarray(adata.obsm[embed_key])
    obs = adata.obs
    is_ctrl = (obs[control_key] == control_label).to_numpy()
    if not is_ctrl.any():
        raise ValueError(f"no cells with {control_key} == '{control_label}'")

    cf, cf_info = None, None
    if mode == "counterfactual" and (isinstance(counterfactual, str) or hasattr(counterfactual, "transport_pcs")):
        counterfactual, cf_info = _cellot_counterfactual(
            adata, counterfactual, cellot_kwargs, embed_key=embed_key, control_key=control_key,
            control_label=control_label, is_ctrl=is_ctrl, donor=obs[donor_key].astype(str).to_numpy(),
            celltype=obs[celltype_key].astype(str).to_numpy() if celltype_key else None,
            exclude=exclude_celltypes)
    elif cellot_kwargs:
        raise ValueError("cellot_kwargs is only used with counterfactual='cellot'")
    if mode == "counterfactual":
        cfs = [counterfactual] if isinstance(counterfactual, AnnData) else list(counterfactual)
        get = (lambda a: _dense(a.X)) if cf_embed_key is None else (lambda a: np.asarray(a.obsm[cf_embed_key]))
        cf = (
            np.vstack([get(a) for a in cfs]),
            np.concatenate([a.obs[donor_key].astype(str).to_numpy() for a in cfs]),
            np.concatenate([a.obs[celltype_key].astype(str).to_numpy() for a in cfs])
            if celltype_key else None,
        )

    with warnings.catch_warnings():  # reported once below, with all reasons
        warnings.filterwarnings("ignore", message="Dropped .* cells in donor x cell-type groups")
        cents = compute_centroids(
            X,
            is_ctrl,
            donor=None if mode == "pooled" else obs[donor_key].to_numpy(),
            celltype=obs[celltype_key].to_numpy() if celltype_key else None,
            counterfactual=cf,
            fallback=(mode == "counterfactual"),
            exclude_celltypes=exclude_celltypes,
        )
    n_drop = adata.n_obs - int(cents.keep.sum())
    if n_drop:
        why = {"missing_label": "missing donor/cell-type label", "excluded_celltype": "excluded cell types",
               "no_control_group": "donor x cell-type groups without control cells"}
        detail = ", ".join(f"{v} {why[k]}" for k, v in cents.dropped.items() if v)
        warnings.warn(f"{n_drop} of {adata.n_obs} cells were not scored ({detail}); the returned AnnData has "
                      f"{adata.n_obs - n_drop} cells. Assign it to a new name to keep the input "
                      f"(out = cytap.predict(adata, ...)); uns['cytap']['dropped_cells'] has the counts.",
                      stacklevel=2)
    Xk = X[cents.keep]
    expected = cfg["input_dim"]
    probs = []
    for i in range(0, len(Xk), batch_size):
        sl = slice(i, i + batch_size)
        ck, cdk = cents.rows(sl)
        feats = compose(Xk[sl], ck, cdk)
        if feats.shape[1] != expected:
            raise ValueError(f"input dim {feats.shape[1]} != model input_dim {expected}")
        probs.append(predict_proba(model, feats, batch_size))
    probs = np.vstack(probs)

    out = adata[cents.keep].copy()
    classes = np.asarray(model.classes)
    out.obsm[f"{key_added}_probs"] = probs
    out.obsm[f"{key_added}_logits"] = _logit(probs)        # log(p / (1 - p)), p clipped to [1e-6, 1 - 1e-6]
    out.uns[f"{key_added}_classes"] = classes
    out.uns["cytap"] = {
        "design": design,
        "reference": "own controls" if design == "paired" else reference,
        "representation": cfg["representation"],
        "embed_key": embed_key,
        "control_key": control_key,
        "control_label": control_label,
        "donor_key": donor_key or "",
        "celltype_key": celltype_key or "",
        "exclude_celltypes": [str(c) for c in (exclude_celltypes or [])],
        "centroid_source": cents.source,
        "n_input_cells": int(adata.n_obs),
        "n_predicted_cells": int(out.n_obs),
        "dropped_cells": {k: int(v) for k, v in cents.dropped.items()},
    }
    if cf_info:
        out.uns["cytap"]["cellot"] = cf_info
    return out


def _cellot_counterfactual(adata, counterfactual, cellot_kwargs, *, embed_key, control_key, control_label,
                           is_ctrl, donor, celltype, exclude):
    """Counterfactual control cells from CellOT for donor x cell-type groups without controls."""
    from .counterfactual import CellOT

    kw = dict(cellot_kwargs or {})
    save = kw.pop("save", None)
    if isinstance(counterfactual, str):
        if counterfactual != "cellot":
            raise ValueError(f"unknown counterfactual method '{counterfactual}'")
        cot = CellOT.fit(adata, condition_key=control_key, control_label=control_label, embed_key=embed_key, **kw)
        if save:
            cot.save(save)
    else:
        if kw or save:
            raise ValueError("cellot_kwargs is only used with counterfactual='cellot'")
        cot = counterfactual
        if cot.config["control_label"] != control_label:
            raise ValueError(f"CellOT maps go to '{cot.config['control_label']}', not '{control_label}'")
    if cot.components.shape[1] != adata.obsm[embed_key].shape[1]:
        raise ValueError("CellOT PCA dimension does not match the embedding")

    grp = donor if celltype is None else np.char.add(np.char.add(donor, "\x1f"), celltype)
    need = ~is_ctrl & ~np.isin(grp, np.unique(grp[is_ctrl]))
    if celltype is not None and exclude:
        need &= ~np.isin(celltype, [str(c) for c in exclude])
    cond = adata.obs[control_key].astype(str).to_numpy()
    untrained = sorted(set(cond[need]) - set(cot.source_labels))
    if untrained:
        warnings.warn(f"no CellOT map for {untrained}; those cells use the pooled c_k fallback")
    cf = cot.transport(adata, embed_key=embed_key, condition_key=control_key, mask=need)
    info = {"source_labels": cot.source_labels, "n_transported": int(cf.n_obs),
            "n_pcs": int(cot.components.shape[0]),
            "best_step": {k: int(v.get("best_step", -1)) for k, v in cot.info.items()},
            "min_mmd": {k: float(v.get("min_mmd", np.nan)) for k, v in cot.info.items()}}
    return cf, info

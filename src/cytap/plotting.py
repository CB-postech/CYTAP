"""Quick-look plots of CYTAP results (matplotlib)."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

from . import stats

XLABELS = {"delta_logit": "Effect size (delta logit)", "rho": "Spearman rho (severity)"}


def _ax(ax, figsize):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _box_points(ax, groups, positions, color, width, seed=0):
    rng = np.random.default_rng(seed)
    bp = ax.boxplot(groups, positions=positions, widths=width, showfliers=False, patch_artist=True,
                    medianprops={"color": "black"})
    for b in bp["boxes"]:
        b.set(facecolor=color, alpha=0.6)
    for x, g in zip(positions, groups):
        ax.scatter(x + rng.uniform(-width / 4, width / 4, len(g)), g, s=10, color="black", alpha=0.7, zorder=3)


def lollipop(res: pd.DataFrame, x: str = "delta_logit", top_n: int = 10, fdr: float | None = 0.05,
             ax=None, title: str | None = None):
    """Top cytokines with positive ``x`` (e.g. ``delta_logit`` from
    :func:`cytap.stats.compare_conditions` or ``rho`` from :func:`cytap.stats.severity_trend`),
    coloured by -log10(FDR)."""
    d = res[res[x] > 0]
    if fdr is not None:
        d = d[d["fdr"] < fdr]
    d = d.sort_values(x).tail(top_n)
    if d.empty:
        raise ValueError("no cytokine passes the filter")
    ax = _ax(ax, (5, max(2.5, 0.35 * len(d))))
    ax.hlines(d["cytokine"], 0, d[x], color="grey", alpha=0.6)
    sc = ax.scatter(d[x], d["cytokine"], c=-np.log10(d["fdr"] + 1e-300), cmap="Reds",
                    s=80, edgecolor="black", linewidth=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label="-log10(FDR)")
    ax.set_xlim(0, d[x].max() * 1.05)
    ax.set_xlabel(XLABELS.get(x, x))
    ax.set_title(title or f"Top {len(d)} cytokines" + (f" (FDR < {fdr})" if fdr is not None else ""))
    return ax


def response(adata: AnnData, cytokine: str, donor_key: str, condition_key: str,
             celltype_key: str | None = None, *, conditions: Sequence[str] | None = None,
             baseline: str | None = None, value: str = "logit", colors: Sequence[str] | None = None,
             ax=None, **kw):
    """Per-donor median logit of one cytokine by condition (and cell type).

    With ``baseline`` (e.g. the control group), each value is shown relative to the mean of the
    ``baseline`` donors of the same cell type. Use this to compare cell types: the absolute logit of
    control cells is a cell-type-specific default of the classifier, not a response, so absolute levels
    are comparable only within a cell type.
    """
    df = stats.pseudobulk(adata, donor_key, condition_key, celltype_key, cytokines=[cytokine],
                       conditions=conditions, value=value, **kw)
    conds = list(conditions) if conditions is not None else sorted(df[condition_key].astype(str).unique())
    if baseline is not None:
        if baseline not in conds:
            raise ValueError(f"baseline '{baseline}' not among the conditions {conds}")
        grp = df[celltype_key].astype(str) if celltype_key else pd.Series("All Cells", index=df.index)
        ref = df[df[condition_key].astype(str) == baseline].groupby(grp, observed=True)[cytokine].mean()
        df = df.assign(**{cytokine: df[cytokine] - grp.map(ref).astype(float)}).dropna(subset=[cytokine])
    groups = sorted(df[celltype_key].astype(str).unique()) if celltype_key else ["All Cells"]
    colors = colors or plt.cm.tab10.colors
    ax = _ax(ax, (max(4, 1.2 * len(groups) * len(conds) / 2), 4))
    w = 0.8 / len(conds)
    for j, c in enumerate(conds):
        sel = df[df[condition_key].astype(str) == c]
        vals = [sel[sel[celltype_key].astype(str) == g][cytokine].to_numpy() if celltype_key else sel[cytokine].to_numpy()
                for g in groups]
        pos = np.arange(len(groups)) - 0.4 + w * (j + 0.5)
        keep = [i for i, v in enumerate(vals) if len(v)]
        _box_points(ax, [vals[i] for i in keep], pos[keep], colors[j % len(colors)], w * 0.9, seed=j)
        ax.plot([], [], "s", color=colors[j % len(colors)], label=c)
    ax.set_xticks(np.arange(len(groups)), groups, rotation=45 if len(groups) > 2 else 0, ha="right" if len(groups) > 2 else "center")
    if baseline is not None:
        ax.axhline(0, color="red", ls="--", alpha=0.5)
        ax.set_ylabel(f"Δ median {value} vs {baseline} mean ({cytokine})")
    else:
        ax.set_ylabel(f"Median {value} ({cytokine})")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    return ax


def delta(adata: AnnData, cytokine: str, condition_key: str, baseline: str, targets,
          donor_key: str, celltype_key: str | None = None, *, value: str = "logit",
          color: str = "#8ecae6", ax=None, **kw):
    """Within-donor change (target - baseline) of the median logit.
    One target: x = cell type, sorted by the median change (largest first). Several targets: x = condition."""
    df = stats.paired_delta(adata, cytokine, condition_key, baseline, targets, donor_key, celltype_key,
                         value=value, **kw)
    by = "condition" if df["condition"].nunique() > 1 else "group"
    if by == "condition":
        labels = [str(c) for c in df[by].cat.categories]
    else:
        labels = [str(c) for c in df.groupby(by, observed=True)["delta"].median().sort_values(ascending=False).index]
    vals = [df[df[by].astype(str) == lab]["delta"].to_numpy() for lab in labels]
    ax = _ax(ax, (max(4, 0.7 * len(labels)), 4))
    ax.axhline(0, color="red", ls="--", alpha=0.5)
    _box_points(ax, vals, np.arange(len(labels)), color, 0.6)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45 if len(labels) > 2 else 0, ha="right" if len(labels) > 2 else "center")
    ax.set_ylabel(f"Δ median {value} ({cytokine})")
    return ax


def embedding(adata: AnnData, cytokine: str, basis: str = "X_umap", *, value: str = "logit",
              key: str = "pred", cmap: str | None = None, s: float = 2, ax=None):
    """Cells on a 2-D embedding coloured by the cytokine logit (or probability)."""
    vals, classes = stats._values(adata, key, value, stats.LOGIT_EPS)
    v = vals[:, classes.index(cytokine)]
    xy = np.asarray(adata.obsm[basis])[:, :2]
    o = np.argsort(v)
    ax = _ax(ax, (5, 4.5))
    sc = ax.scatter(xy[o, 0], xy[o, 1], c=v[o], s=s, cmap=cmap or ("coolwarm" if value == "logit" else "Reds"),
                    vmin=None if value == "logit" else 0, vmax=None if value == "logit" else 1, rasterized=True)
    plt.colorbar(sc, ax=ax, label=f"{cytokine} {value}")
    ax.set_axis_off()
    ax.set_title(f"{cytokine} ({value})")
    return ax

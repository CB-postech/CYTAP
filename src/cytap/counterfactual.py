"""Counterfactual control cells for unpaired data with CellOT (neural optimal transport).

For each non-control condition s, a CellOT map g_s is learned in PCA space that moves cells of
condition s onto the distribution of control cells. Transported cells are decoded back to the
embedding with the inverse PCA and used as donor-matched controls in
``cytap.predict(design="unpaired", reference="counterfactual")``.

The training loop reproduces CellOT ``train_cellot`` (Bunne et al., Nat. Methods 2023) with the
settings used in the CYTAP paper (configs/*_cellot.yaml): ICNN f, g with 4 x 64 hidden units,
U(0, 0.1) initialisation, fnorm penalty 1 on g, Adam(lr 1e-4, betas (0.5, 0.9)), 10 g-steps per
f-step, 10,000 iterations, 80/20 train/test split per group (random_state 0), and the g with the
lowest test MMD (evaluated every 100 iterations) is kept.

    cf_model = cytap.counterfactual.CellOT.fit(adata, condition_key="Source", control_label="HV")
    cf_model.save("combat_cellot.pt")
    out = cytap.predict(adata, model, design="unpaired", reference="counterfactual",
                        control_key="Source", control_label="HV",
                        donor_key="donor_id", celltype_key="major_subset", counterfactual=cf_model)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split

from ._cellot import ICNN, compute_loss_f, compute_loss_g, compute_scalar_mmd, compute_w2_distance, uniform_init

FIT_DEFAULTS = dict(
    n_pcs=100,
    hidden_units=(64, 64, 64, 64),
    n_iters=10_000,
    n_inner_iters=10,
    batch_size=256,
    test_size=0.2,
    split_seed=0,
    eval_freq=100,
    lr=1e-4,
    betas=(0.5, 0.9),
    fnorm_penalty=1.0,
    init_b=0.1,
)


def _device(device):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


class _Cycle:
    """Endless shuffled mini-batches, identical to iterating a cycled
    ``DataLoader(TensorDataset(X), batch_size, shuffle=True, drop_last=True)``:
    each new pass draws the loader's base seed and the sampler seed from the global torch RNG
    and permutes with a generator seeded by the latter.
    """

    def __init__(self, X: torch.Tensor, batch_size: int):
        self.X, self.bs = X, batch_size
        self.n_batches = len(X) // batch_size
        if self.n_batches == 0:
            raise ValueError("fewer cells than batch_size")
        self._perm, self._i = None, self.n_batches

    def __next__(self):
        if self._i == self.n_batches:
            torch.empty((), dtype=torch.int64).random_()  # DataLoader iterator base seed
            seed = int(torch.empty((), dtype=torch.int64).random_().item())  # RandomSampler seed
            gen = torch.Generator()
            gen.manual_seed(seed)
            self._perm, self._i = torch.randperm(len(self.X), generator=gen), 0
        idx = self._perm[self._i * self.bs : (self._i + 1) * self.bs]
        self._i += 1
        return self.X[idx]


def split_train_test(groups, test_size=0.2, random_state=0) -> np.ndarray:
    """CellOT ``split_cell_data_train_test``: per-group train/test split. Returns 'train'/'test'
    (None where the group label is missing)."""
    groups = pd.Series(np.asarray(groups, dtype=object))
    split = np.full(len(groups), None, dtype=object)
    for _, idx in groups.groupby(groups, sort=True).indices.items():
        tr, te = train_test_split(idx, random_state=random_state, test_size=test_size)
        split[tr], split[te] = "train", "test"
    return split


def fit_map(
    Xp: np.ndarray,
    is_source: np.ndarray,
    split: np.ndarray,
    *,
    seed: int | None = 0,
    device=None,
    log_fn: Callable[[str], None] | None = print,
    log_every: int = 1000,
    history_path: str | Path | None = None,
    map_name: str = "map",
    **params,
):
    """Train one CellOT map (source -> control) on PCA coordinates.

    Xp         (n, n_pcs) cells of the source condition and control cells, in original order
    is_source  bool mask; False = control (CellOT 'target')
    split      'train' / 'test' per cell (see :func:`split_train_test`); None = unused

    history_path  CSV that gets one row per evaluation (map, step, g_loss, f_loss, w2, test_mmd),
                  appended while training, e.g. to follow the run from another notebook
    Returns (g_state_dict, info) with the lowest-test-MMD g.
    """
    unknown = set(params) - set(FIT_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown CellOT parameters: {sorted(unknown)}")
    p = {**FIT_DEFAULTS, **params}
    device = _device(device)
    if seed is not None:
        torch.manual_seed(seed)

    d = Xp.shape[1]
    init = uniform_init(b=p["init_b"])
    f = ICNN(d, list(p["hidden_units"]), kernel_init_fxn=init).to(device)
    g = ICNN(d, list(p["hidden_units"]), fnorm_penalty=p["fnorm_penalty"], kernel_init_fxn=init).to(device)
    opt_f = torch.optim.Adam(f.parameters(), lr=p["lr"], betas=tuple(p["betas"]), weight_decay=0)
    opt_g = torch.optim.Adam(g.parameters(), lr=p["lr"], betas=tuple(p["betas"]), weight_decay=0)

    X = torch.from_numpy(np.ascontiguousarray(Xp, dtype=np.float32))
    split = np.asarray(split, dtype=object)
    sub = {(s, t): np.flatnonzero((split == s) & (is_source == (t == "source")))
           for s in ("test", "train") for t in ("source", "target")}
    for k, v in sub.items():
        if len(v) == 0:
            raise ValueError(f"no {k[1]} cells in the {k[0]} split")
    bs = {s: min(len(sub[(s, "source")]), len(sub[(s, "target")]), p["batch_size"]) for s in ("test", "train")}
    it = {k: _Cycle(X[v], bs[k[0]]) for k, v in sub.items()}

    best, best_state, best_step, history = np.inf, None, -1, []
    t0 = time.time()
    for step in range(p["n_iters"]):
        target = next(it[("train", "target")]).to(device)
        for _ in range(p["n_inner_iters"]):
            source = next(it[("train", "source")]).to(device).requires_grad_(True)
            opt_g.zero_grad()
            gl = compute_loss_g(f, g, source).mean()
            if not g.softplus_W_kernels and g.fnorm_penalty > 0:
                gl = gl + g.penalize_w()
            gl.backward()
            opt_g.step()

        source = next(it[("train", "source")]).to(device).requires_grad_(True)
        opt_f.zero_grad()
        fl = compute_loss_f(f, g, source, target).mean()
        fl.backward()
        opt_f.step()
        if torch.isnan(gl) or torch.isnan(fl):
            raise ValueError(f"CellOT loss is NaN at step {step}")
        f.clamp_w()

        if step % p["eval_freq"] == 0:
            t_te = next(it[("test", "target")]).to(device)
            s_te = next(it[("test", "source")]).to(device).requires_grad_(True)
            tr = g.transport(s_te).detach()
            mmd = compute_scalar_mmd(t_te.cpu().numpy(), tr.cpu().numpy())
            row = {"map": map_name, "step": step, "g_loss": gl.item(), "f_loss": fl.item(),
                   "w2": float(compute_w2_distance(f, g, s_te, t_te, tr)), "test_mmd": float(mmd)}
            history.append(row)
            if history_path:
                pd.DataFrame([row]).to_csv(history_path, mode="a", index=False,
                                           header=not Path(history_path).exists())
            if mmd < best:
                best, best_step = mmd, step
                best_state = {k: v.detach().cpu().clone() for k, v in g.state_dict().items()}
        if log_fn and (step + 1) % log_every == 0:
            log_fn(f"  step {step + 1}/{p['n_iters']}  g {gl.item():.4f}  f {fl.item():.4f}  "
                   f"best MMD {best:.5f} (step {best_step})  {time.time() - t0:.0f}s")

    if best_state is None:
        raise ValueError("no finite test MMD during training")
    info = {"best_step": best_step, "min_mmd": float(best), "history": history,
            "n_train": {k[1]: int(len(v)) for k, v in sub.items() if k[0] == "train"},
            "batch_size": bs}
    return best_state, info


def _embedding(adata, embed_key):
    X = adata.obsm[embed_key] if embed_key else adata.X
    return X.toarray() if issparse(X) else np.asarray(X)


class CellOT:
    """CellOT maps (one per source condition) plus the PCA used to encode and decode cells."""

    def __init__(self, components, mean, maps: dict, config: dict, info: dict | None = None, device=None):
        self.components = np.ascontiguousarray(components)
        self.mean = np.ascontiguousarray(mean)
        self.config = dict(config)
        self.info = info or {}
        self.device = _device(device)
        self.maps = {}
        for label, state in maps.items():
            g = ICNN(self.components.shape[0], list(self.config["hidden_units"]),
                     fnorm_penalty=self.config.get("fnorm_penalty", 0))
            g.load_state_dict(state)
            self.maps[str(label)] = g.to(self.device).eval()

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(
        cls,
        adata: AnnData,
        *,
        condition_key: str,
        control_label: str,
        source_labels: Sequence[str] | None = None,
        embed_key: str | None = "X_state",
        split_key: str | None = None,
        seed: int | None = 0,
        device=None,
        log_fn: Callable[[str], None] | None = print,
        history_path: str | Path | None = None,
        **params,
    ) -> "CellOT":
        """Fit the PCA and one CellOT map per source condition.

        condition_key   obs column with control and other conditions
        control_label   control value of ``condition_key`` (CellOT target)
        source_labels   conditions to map to control (default: all other values)
        embed_key       obsm key of the embedding (None = ``.X``)
        split_key       obs column whose groups are split 80/20 separately
                        (default ``condition_key``; the Parse-PBMC map used 'cytokine')
        seed            torch seed set before each map (CellOT itself sets none)
        history_path    CSV with the training curves, appended at each evaluation (see :attr:`history`)
        params          overrides of :data:`FIT_DEFAULTS` (e.g. n_pcs, n_iters, batch_size)

        The PCA is fitted on control and source cells together, as in the paper.
        """
        from sklearn.decomposition import PCA

        unknown = set(params) - set(FIT_DEFAULTS)
        if unknown:
            raise TypeError(f"unknown CellOT parameters: {sorted(unknown)}")
        p = {**FIT_DEFAULTS, **params}
        cond = adata.obs[condition_key].astype(str).to_numpy()
        if not (cond == control_label).any():
            raise ValueError(f"no cells with {condition_key} == '{control_label}'")
        if source_labels is None:
            source_labels = [c for c in pd.unique(cond) if c != control_label]
        source_labels = [str(s) for s in source_labels]
        missing = [s for s in source_labels if not (cond == s).any()]
        if missing:
            raise ValueError(f"no cells for source labels {missing}")

        use = np.isin(cond, source_labels + [control_label])
        X = _embedding(adata, embed_key)[use].astype(np.float32, copy=False)
        cond = cond[use]
        groups = (adata.obs[split_key].to_numpy()[use] if split_key else cond)

        if p["n_pcs"] > min(X.shape):
            raise ValueError(f"n_pcs={p['n_pcs']} exceeds min(n_cells, n_features)={min(X.shape)}")
        if log_fn:
            log_fn(f"PCA on {X.shape[0]} cells ...")
        pca = PCA(n_components=p["n_pcs"], random_state=p["split_seed"])
        Xp = pca.fit_transform(X).astype(np.float32)
        if log_fn:
            log_fn(f"PCA {X.shape[1]} -> {p['n_pcs']}: {100 * pca.explained_variance_ratio_.sum():.1f}% variance")

        maps, info = {}, {}
        for s in source_labels:
            m = (cond == s) | (cond == control_label)
            split = split_train_test(np.asarray(groups, dtype=object)[m], p["test_size"], p["split_seed"])
            if log_fn:
                log_fn(f"CellOT {s} -> {control_label}: {int((cond == s).sum())} source, "
                       f"{int((cond == control_label).sum())} control cells")
            maps[s], info[s] = fit_map(Xp[m], cond[m] == s, split, seed=seed, device=device, log_fn=log_fn,
                                       history_path=history_path, map_name=s,
                                       **{k: v for k, v in p.items() if k not in ("n_pcs",)})
            if log_fn:
                log_fn(f"  kept step {info[s]['best_step']} (test MMD {info[s]['min_mmd']:.5f})")

        config = {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}
        config.update(condition_key=condition_key, control_label=control_label, embed_key=embed_key,
                      split_key=split_key or condition_key, seed=seed,
                      explained_variance=float(pca.explained_variance_ratio_.sum()))
        return cls(pca.components_, pca.mean_, maps, config, info, device=device)

    # ---------------------------------------------------------------- transport
    @property
    def history(self) -> pd.DataFrame:
        """Training curves of the maps fitted in this session (one row per evaluation; not saved to file)."""
        rows = [r for v in self.info.values() for r in v.get("history", [])]
        return pd.DataFrame(rows, columns=["map", "step", "g_loss", "f_loss", "w2", "test_mmd"])

    @property
    def source_labels(self):
        return list(self.maps)

    def encode(self, X):
        return ((np.asarray(X) - self.mean) @ self.components.T).astype(np.float32)

    def decode(self, Xp):
        return (np.asarray(Xp) @ self.components + self.mean).astype(np.float32)

    def transport_pcs(self, Xp, source_label: str, chunk: int = 200_000) -> np.ndarray:
        """Apply map g_{source_label} to PCA coordinates."""
        g = self.maps[str(source_label)]
        Xp = np.asarray(Xp, dtype=np.float32)
        out = np.empty_like(Xp)
        with torch.enable_grad():
            for i in range(0, len(Xp), chunk):
                x = torch.tensor(Xp[i : i + chunk], device=self.device, requires_grad=True)
                out[i : i + chunk] = g.transport(x).detach().cpu().numpy()
        return out

    def transport(self, adata: AnnData, *, embed_key: str | None = None, condition_key: str | None = None,
                  mask=None, chunk: int = 200_000) -> AnnData:
        """Counterfactual control cells for the cells of trained source conditions.

        Returns AnnData with the decoded embedding in ``.X``, PCA coordinates in
        ``obsm['X_cellot_pca']``, obs of the source cells with ``condition_key`` set to the
        control label and the original condition in ``obs['cellot_source']``.
        ``mask`` restricts the cells to transport.
        """
        cfg = self.config
        embed_key = cfg.get("embed_key") if embed_key is None else embed_key
        condition_key = condition_key or cfg["condition_key"]
        cond = adata.obs[condition_key].astype(str).to_numpy()
        sel = np.isin(cond, self.source_labels)
        if mask is not None:
            sel &= np.asarray(mask, dtype=bool)
        idx = np.flatnonzero(sel)
        X = _embedding(adata, embed_key)
        pcs = np.empty((len(idx), self.components.shape[0]), dtype=np.float32)
        for s in self.source_labels:
            m = cond[idx] == s
            if m.any():
                pcs[m] = self.transport_pcs(self.encode(X[idx[m]]), s, chunk)
        out = AnnData(X=self.decode(pcs), obs=adata.obs.iloc[idx].copy())
        out.obs["cellot_source"] = cond[idx]
        out.obs[condition_key] = cfg["control_label"]
        out.obsm["X_cellot_pca"] = pcs
        return out

    # ------------------------------------------------------------------ storage
    def save(self, path):
        torch.save({
            "format": "cytap.cellot/1",
            "components": torch.from_numpy(self.components),
            "mean": torch.from_numpy(self.mean),
            "maps": {k: g.state_dict() for k, g in self.maps.items()},
            "config": self.config,
            "info": {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in self.info.items()},
        }, path)

    @classmethod
    def load(cls, path, device=None) -> "CellOT":
        """Load a saved CellOT file, or a pretrained map by name ("combat_cellot": the COMBAT maps of the paper,
        downloaded on first use)."""
        from .model import PRETRAINED, model_path

        if not Path(path).exists() and str(path) in PRETRAINED:
            path = model_path(str(path))
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck.get("format") != "cytap.cellot/1":
            raise ValueError(f"{path} is not a cytap CellOT file")
        return cls(ck["components"].numpy(), ck["mean"].numpy(), ck["maps"], ck["config"], ck["info"], device)

    @classmethod
    def from_cellot(cls, map_dirs: dict, pca_path, *, condition_key: str, control_label: str,
                    embed_key: str | None = "X_state", device=None) -> "CellOT":
        """Load maps trained with the CellOT repository.

        map_dirs  {source label: CellOT output dir with config.yaml and cache/model.pt}
        pca_path  pickled sklearn PCA used to make the CellOT input (*_pca_model_100d.pkl)

        Both files are unpickled: only load files you created.
        """
        import pickle

        import yaml

        with open(pca_path, "rb") as fh:
            pca = pickle.load(fh)
        if getattr(pca, "whiten", False):
            raise ValueError("whitened PCA is not supported")
        maps, info, hidden, fnorm = {}, {}, None, 0
        for label, d in map_dirs.items():
            d = Path(d)
            with open(d / "config.yaml") as fh:
                mc = (yaml.safe_load(fh) or {}).get("model", {})
            h = list(mc.get("hidden_units", [64] * 4))
            if hidden is not None and h != hidden:
                raise ValueError("all maps must share hidden_units")
            hidden, fnorm = h, mc.get("g", {}).get("fnorm_penalty", mc.get("fnorm_penalty", 0))
            ck = torch.load(d / "cache" / "model.pt", map_location="cpu", weights_only=False)
            maps[str(label)] = ck["g_state"]
            info[str(label)] = {"best_step": int(ck.get("step", -1)), "min_mmd": float(ck.get("minmmd", np.nan)),
                                "source_dir": str(d)}
        config = dict(FIT_DEFAULTS, hidden_units=hidden, fnorm_penalty=fnorm, n_pcs=int(pca.n_components_),
                      condition_key=condition_key, control_label=control_label, embed_key=embed_key,
                      pca_path=str(pca_path))
        config = {k: (list(v) if isinstance(v, tuple) else v) for k, v in config.items()}
        return cls(pca.components_, pca.mean_, maps, config, info, device=device)

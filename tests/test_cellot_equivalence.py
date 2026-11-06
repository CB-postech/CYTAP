"""cytap.counterfactual.fit_map trains the same map as the original CellOT code (same seed, CPU).

Run in the CellOT environment (python 3.9, https://github.com/bunnech/cellot) with cytap installed; skipped otherwise:
    CELLOT_REPO=~/cellot pytest tests/test_cellot_equivalence.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = os.environ.get("CELLOT_REPO")
pytestmark = pytest.mark.skipif(not REPO, reason="set CELLOT_REPO to a clone of the CellOT repository")


def test_same_map_as_cellot(tmp_path, monkeypatch):
    import anndata as ad
    import pandas as pd

    sys.path.insert(0, str(Path(REPO).expanduser()))
    pytest.importorskip("ml_collections")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from ml_collections import ConfigDict

    import cellot.train.train as T
    from cytap.counterfactual import fit_map, split_train_test

    rng = np.random.default_rng(1)
    n_s, n_t, d, n_iters, bs = 900, 650, 30, 401, 256
    lab = np.array(["SRC"] * n_s + ["CTRL"] * n_t)
    X = np.vstack([rng.normal(0.8, 1.0, (n_s, d)), rng.normal(0.0, 1.2, (n_t, d))])
    perm = rng.permutation(len(lab))
    X, lab = X[perm].astype(np.float32), lab[perm]

    (tmp_path / "cache").mkdir()
    ad.AnnData(X, obs=pd.DataFrame({"cond": lab}, index=[f"c{i}" for i in range(len(lab))])).write_h5ad(tmp_path / "d.h5ad")
    config = ConfigDict({
        "data": {"type": "cell", "source": "SRC", "target": "CTRL", "condition": "cond", "path": str(tmp_path / "d.h5ad")},
        "datasplit": {"groupby": "cond", "name": "train_test", "test_size": 0.2, "random_state": 0},
        "dataloader": {"batch_size": bs, "shuffle": True},
        "model": {"name": "cellot", "hidden_units": [64] * 4, "softplus_W_kernels": False,
                  "g": {"fnorm_penalty": 1}, "kernel_init_fxn": {"b": 0.1, "name": "uniform"}},
        "optim": {"optimizer": "Adam", "lr": 1e-4, "beta1": 0.5, "beta2": 0.9, "weight_decay": 0},
        "training": {"n_iters": n_iters, "n_inner_iters": 10, "cache_freq": 200, "eval_freq": 100, "logs_freq": 20},
    })
    torch.manual_seed(0)
    T.train_cellot(tmp_path, config)
    ref = torch.load(tmp_path / "cache/model.pt", map_location="cpu", weights_only=False)

    state, info = fit_map(X, lab == "SRC", split_train_test(lab, 0.2, 0), seed=0, device="cpu", log_fn=None,
                          n_iters=n_iters, batch_size=bs)
    assert info["best_step"] == ref["step"]
    assert all(torch.equal(ref["g_state"][k].cpu(), state[k]) for k in state)

"""cytap.counterfactual: CellOT counterfactual controls."""
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from anndata import AnnData
from torch.utils.data import DataLoader, TensorDataset

import cytap
from cytap.counterfactual import CellOT, _Cycle, split_train_test


def _cycled_loader(X, bs):
    loader = DataLoader(TensorDataset(X), batch_size=bs, shuffle=True, drop_last=True)
    while True:
        for (x,) in loader:
            yield x


def test_cycle_matches_dataloader():
    X = torch.arange(53 * 3, dtype=torch.float32).reshape(53, 3)
    torch.manual_seed(7)
    ref = [next(it) for it in [_cycled_loader(X, 10)] for _ in range(23)]
    ref_after = torch.rand(1)
    torch.manual_seed(7)
    c = _Cycle(X, 10)
    ours = [next(c) for _ in range(23)]
    assert all(torch.equal(a, b) for a, b in zip(ref, ours))
    assert torch.equal(ref_after, torch.rand(1))  # same global RNG consumption


def test_split_per_group():
    g = np.array(["a"] * 50 + ["b"] * 30)
    s = split_train_test(g, 0.2, 0)
    assert (s[g == "a"] == "test").sum() == 10 and (s[g == "b"] == "test").sum() == 6
    assert np.array_equal(s, split_train_test(g, 0.2, 0))


def _toy(seed=0, n_donor=6, d=40):
    rng = np.random.default_rng(seed)
    rows, X = [], []
    for i in range(n_donor):
        cond = "HV" if i < 2 else ("MILD" if i < 4 else "SEV")
        for ct in ["T", "B"]:
            n = 60
            shift = {"HV": 0, "MILD": 0.5, "SEV": 1.0}[cond] + (0.3 if ct == "B" else 0)
            X.append(rng.normal(shift, 1, (n, d)))
            rows += [(cond, f"D{i}", ct)] * n
    obs = pd.DataFrame(rows, columns=["Source", "donor", "celltype"])
    obs.index = obs.index.astype(str)
    a = AnnData(obs=obs)
    a.obsm["X_state"] = np.vstack(X).astype(np.float32)
    return a


FAST = dict(n_pcs=8, n_iters=30, eval_freq=10, batch_size=32, hidden_units=(16, 16))


def test_fit_transport_save_load(tmp_path):
    a = _toy()
    cot = CellOT.fit(a, condition_key="Source", control_label="HV", log_fn=None, device="cpu", **FAST)
    assert sorted(cot.source_labels) == ["MILD", "SEV"]
    cf = cot.transport(a)
    assert cf.n_obs == (a.obs["Source"] != "HV").sum() and cf.X.shape[1] == 40
    assert (cf.obs["Source"] == "HV").all() and set(cf.obs["cellot_source"]) == {"MILD", "SEV"}
    cot.save(tmp_path / "m.pt")
    cf2 = CellOT.load(tmp_path / "m.pt", device="cpu").transport(a)
    np.testing.assert_array_equal(cf.X, cf2.X)
    # same seed -> same maps
    cf3 = CellOT.fit(a, condition_key="Source", control_label="HV", log_fn=None, device="cpu", **FAST).transport(a)
    np.testing.assert_array_equal(cf.X, cf3.X)


def test_load_pretrained_name(tmp_path, monkeypatch):
    """CellOT.load("combat_cellot") reads the cached release file (download skipped when cached)."""
    a = _toy()
    cot = CellOT.fit(a, condition_key="Source", control_label="HV", log_fn=None, device="cpu", **FAST)
    monkeypatch.setenv("CYTAP_CACHE", str(tmp_path))
    cot.save(tmp_path / "combat_cellot.pt")
    np.testing.assert_array_equal(cot.transport(a).X, CellOT.load("combat_cellot", device="cpu").transport(a).X)


def test_from_cellot_layout(tmp_path):
    """Maps saved in the CellOT repository layout load and transport identically."""
    from sklearn.decomposition import PCA

    a = _toy()
    cot = CellOT.fit(a, condition_key="Source", control_label="HV", log_fn=None, device="cpu", **FAST)
    pca = PCA(n_components=8)
    pca.fit(a.obsm["X_state"][:20])
    pca.components_, pca.mean_ = cot.components, cot.mean
    with open(tmp_path / "pca.pkl", "wb") as fh:
        pickle.dump(pca, fh)
    dirs = {}
    for s, g in cot.maps.items():
        d = tmp_path / s
        (d / "cache").mkdir(parents=True)
        yaml.dump({"model": {"name": "cellot", "hidden_units": [16, 16], "g": {"fnorm_penalty": 1}}},
                  open(d / "config.yaml", "w"))
        torch.save({"g_state": g.state_dict(), "f_state": {}, "step": 10, "minmmd": np.float64(0.1)},
                   d / "cache" / "model.pt")
        dirs[s] = d
    legacy = CellOT.from_cellot(dirs, tmp_path / "pca.pkl", condition_key="Source", control_label="HV", device="cpu")
    np.testing.assert_array_equal(cot.transport(a).X, legacy.transport(a).X)
    pcs = cot.encode(a.obsm["X_state"][a.obs["Source"].to_numpy() == "SEV"])
    np.testing.assert_allclose(legacy.decode(pcs), pca.inverse_transform(pcs), rtol=0, atol=1e-5)


def _model(d, classes=("IL2", "IFNb", "TNF")):
    torch.manual_seed(0)
    m = cytap.CytokineClassifier(2 * d, len(classes), hidden_dims=(8,))
    m.classes = list(classes)
    m.config = {"representation": "comp", "embedding_key": "X_state", "input_dim": 2 * d,
                "hidden_dims": [8], "dropout": 0.3}
    return m.eval()


def test_predict_cellot():
    a = _toy()
    m = _model(40)
    out = cytap.predict(a, m, design="unpaired", reference="counterfactual", control_key="Source",
                        control_label="HV", donor_key="donor", celltype_key="celltype",
                        cellot_kwargs=dict(log_fn=None, device="cpu", **FAST))      # counterfactual="cellot" by default
    info = out.uns["cytap"]["cellot"]
    assert out.n_obs == a.n_obs and info["n_transported"] == (a.obs["Source"] != "HV").sum()
    assert out.uns["cytap"]["centroid_source"]["counterfactual"] == 8  # 4 donors x 2 cell types
    # a fitted object gives the same result
    cot = CellOT.fit(a, condition_key="Source", control_label="HV", log_fn=None, device="cpu", **FAST)
    out2 = cytap.predict(a, m, design="unpaired", reference="counterfactual", control_key="Source",
                         control_label="HV", donor_key="donor", celltype_key="celltype", counterfactual=cot)
    np.testing.assert_allclose(out.obsm["pred_probs"], out2.obsm["pred_probs"], rtol=0, atol=1e-6)
    with pytest.raises(TypeError):
        CellOT.fit(a, condition_key="Source", control_label="HV", n_iter=5)


def test_predict_stores_logits():
    a = _toy()
    out = cytap.predict(a, _model(40), design="unpaired", control_key="Source", control_label="HV",
                        celltype_key="celltype")
    p = out.obsm["pred_probs"]
    np.testing.assert_array_equal(out.obsm["pred_logits"], cytap.stats._logit(p))
    stored = cytap.stats.logits(out)
    del out.obsm["pred_logits"]
    np.testing.assert_array_equal(stored.to_numpy(), cytap.stats.logits(out).to_numpy())  # same values without stored

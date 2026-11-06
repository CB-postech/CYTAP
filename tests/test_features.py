"""Centroids, inputs and predict(): equality with the original per-dataset inference code used for the
paper (reimplemented below as legacy_*), and input checks."""
import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

import cytap
from cytap.features import compose, compute_centroids
from cytap.train import build_reference_inputs


# ---------------- original inference logic (paper analyses) ----------------
def legacy_paired(X, obs, donor, group, ctrl_col, ctrl_label):
    """Paired design (G-CSF, macrophage, IL-7 analyses)."""
    obs_df = obs.copy()
    ctrl_mask = obs_df[ctrl_col] == ctrl_label
    if not group:
        obs_df["group_key"] = obs_df[donor].astype(str)
    else:
        obs_df["group_key"] = obs_df[donor].astype(str) + "_" + obs_df[group].astype(str)
        s = obs_df[group].astype(str).str.upper()
        obs_df = obs_df[obs_df[group].notna() & (s != "NA") & (s != "NAN")]
        ctrl_mask = obs_df[ctrl_col] == ctrl_label
    valid = [g for g in obs_df["group_key"].unique() if ((obs_df["group_key"] == g) & ctrl_mask).sum() > 0]
    keep = obs.index.isin(obs_df.index[obs_df["group_key"].isin(valid)])
    Xr, o = X[keep], obs[keep].copy()
    o["group_key"] = o[donor].astype(str) if not group else o[donor].astype(str) + "_" + o[group].astype(str)
    ctrl = o[ctrl_col] == ctrl_label
    o["global_key"] = "Global" if not group else o[group].astype(str)
    gm = {k: Xr[np.where((o["global_key"] == k) & ctrl)[0]].mean(axis=0) for k in o["global_key"].unique()}
    dm = {k: Xr[np.where((o["group_key"] == k) & ctrl)[0]].mean(axis=0) for k in o["group_key"].unique()}
    Xd, Xg = np.empty_like(Xr), np.empty_like(Xr)
    for k in o["group_key"].unique():
        idx = np.where(o["group_key"] == k)[0]
        Xd[idx] = dm[k]
        Xg[idx] = gm["Global" if not group else k.split("_", 1)[1]]
    return keep, np.concatenate([Xr - (Xd - Xg), Xr - Xd], axis=1)


def legacy_pooled(X, obs, group, ctrl_col, ctrl_label):
    """Unpaired design, pooled control centroid (COMBAT)."""
    o = obs.copy()
    s = o[group].astype(str).str.upper()
    o = o[o[group].notna() & (s != "NA") & (s != "NAN")]
    ctrl = o[ctrl_col] == ctrl_label
    valid = [g for g in o[group].astype(str).unique() if ((o[group].astype(str) == g) & ctrl).sum() > 0]
    keep = obs.index.isin(o.index[o[group].astype(str).isin(valid)])
    Xr, o = X[keep], obs[keep].copy()
    gk = o[group].astype(str)
    ctrl = o[ctrl_col] == ctrl_label
    Xg = np.empty_like(Xr)
    for g in gk.unique():
        Xg[np.where(gk == g)[0]] = Xr[np.where((gk == g) & ctrl)[0]].mean(axis=0)
    return keep, np.concatenate([Xr, Xr - Xg], axis=1)


def legacy_cellot(X, obs, cf_X, cf_obs, donor, group, ctrl_col, ctrl_label):
    """Unpaired design, CellOT counterfactual controls (COMBAT)."""
    gk = (obs[donor].astype(str) + "_" + obs[group].astype(str)).to_numpy()
    ct = obs[group].astype(str).to_numpy()
    hv = (obs[ctrl_col] == ctrl_label).to_numpy()
    g2c = dict(zip(gk, ct))
    hv_groups = set(gk[hv])
    cfk = (cf_obs[donor].astype(str) + "_" + cf_obs[group].astype(str)).to_numpy()
    gm = {c: X[np.where((ct == c) & hv)[0]].mean(axis=0) for c in np.unique(ct)}
    dm = {}
    for g in np.unique(gk):
        if g in hv_groups:
            dm[g] = X[np.where((gk == g) & hv)[0]].mean(axis=0)
            continue
        idx = np.where(cfk == g)[0]
        dm[g] = cf_X[idx].mean(axis=0) if len(idx) else gm[g2c[g]]
    Xd, Xg = np.empty_like(X), np.empty_like(X)
    for g in np.unique(gk):
        idx = np.where(gk == g)[0]
        Xd[idx], Xg[idx] = dm[g], gm[g2c[g]]
    return np.concatenate([X - (Xd - Xg), X - Xd], axis=1)


# ---------------- synthetic data ----------------
def make_data(n=3000, d=16, seed=0, donors=("D1", "D2", "D3", "D4"), na=True):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({
        "donor": rng.choice(donors, n),
        "celltype": rng.choice(["B", "CD4 T", "NK", "cMono"], n),
        "cond": rng.choice(["ctrl", "stim"], n),
    }, index=[f"c{i}" for i in range(n)])
    obs.loc[(obs.donor == donors[0]) & (obs.celltype == "NK"), "cond"] = "stim"  # group without control
    if na:
        obs.loc[obs.index[:20], "celltype"] = np.nan
        obs.loc[obs.index[20:40], "celltype"] = "NA"
    X = rng.normal(size=(n, d)).astype(np.float32)
    return X, obs


def ours(X, obs, **kw):
    c = compute_centroids(X, (obs["cond"] == "ctrl").to_numpy(), **kw)
    ck, cdk = c.rows()
    return c.keep, compose(X[c.keep], ck, cdk)


def test_paired_matches_legacy():
    X, obs = make_data()
    k0, f0 = legacy_paired(X, obs, "donor", "celltype", "cond", "ctrl")
    k1, f1 = ours(X, obs, donor=obs.donor.values, celltype=obs.celltype.values)
    assert (k0 == k1).all() and np.array_equal(f0, f1)


def test_paired_without_celltype_matches_legacy():
    X, obs = make_data(na=False)
    k0, f0 = legacy_paired(X, obs, "donor", "", "cond", "ctrl")
    k1, f1 = ours(X, obs, donor=obs.donor.values)
    assert (k0 == k1).all() and np.array_equal(f0, f1)


def test_pooled_matches_legacy():
    X, obs = make_data()
    k0, f0 = legacy_pooled(X, obs, "celltype", "cond", "ctrl")
    k1, f1 = ours(X, obs, celltype=obs.celltype.values)
    assert (k0 == k1).all() and np.array_equal(f0, f1)
    d = X.shape[1]
    assert np.array_equal(f1[:, :d], X[k1])  # Adj == Raw when pooled


def test_counterfactual_matches_legacy():
    X, obs = make_data(na=False)
    obs.loc[obs.donor.isin(["D1", "D2"]), "cond"] = "stim"  # D1, D2 unpaired patients
    obs.loc[obs.donor.isin(["D3", "D4"]), "cond"] = "ctrl"  # D3, D4 healthy
    rng = np.random.default_rng(1)
    cf_obs = obs[obs.donor == "D1"].copy()  # counterfactual only for D1 -> D2 uses pooled fallback
    cf_X = rng.normal(size=(len(cf_obs), X.shape[1])).astype(np.float32)
    f0 = legacy_cellot(X, obs, cf_X, cf_obs, "donor", "celltype", "cond", "ctrl")
    c = compute_centroids(X, (obs.cond == "ctrl").to_numpy(), obs.donor.values, obs.celltype.values,
                          counterfactual=(cf_X, cf_obs.donor.values, cf_obs.celltype.values), fallback=True)
    ck, cdk = c.rows()
    assert c.keep.all()
    assert np.array_equal(f0, compose(X, ck, cdk))
    assert c.source == {"own": 8, "counterfactual": 4, "pooled_fallback": 4}


def test_celltype_without_control_raises():
    X, obs = make_data()
    obs.loc[obs.celltype == "NK", "cond"] = "stim"
    with pytest.raises(ValueError, match="exclude_celltypes=\\['NK'\\]"):
        ours(X, obs, celltype=obs.celltype.values)
    k0, f0 = legacy_pooled(X, obs, "celltype", "cond", "ctrl")  # legacy silently dropped NK
    k1, f1 = ours(X, obs, celltype=obs.celltype.values, exclude_celltypes=["NK"])
    assert (k0 == k1).all() and np.array_equal(f0, f1)


def test_exclude_celltypes():
    X, obs = make_data()
    k, _ = ours(X, obs, donor=obs.donor.values, celltype=obs.celltype.values, exclude_celltypes=["B", "NK"])
    assert not obs.celltype[k].isin(["B", "NK"]).any()
    with pytest.warns(UserWarning, match="not found"):
        ours(X, obs, celltype=obs.celltype.values, exclude_celltypes=["Bcell"])


def test_donor_ids_with_underscore():
    X, obs = make_data(donors=("pt_1", "pt_2", "pt_3", "pt_4"), na=False)
    obs2 = obs.assign(donor=obs.donor.str.replace("_", ""))
    k1, f1 = ours(X, obs, donor=obs.donor.values, celltype=obs.celltype.values)
    k2, f2 = ours(X, obs2, donor=obs2.donor.values, celltype=obs2.celltype.values)
    assert (k1 == k2).all() and np.array_equal(f1, f2)


def test_reference_inputs_formula():
    X, obs = make_data(na=False)
    X_in, y, classes, donor, ct = build_reference_inputs(
        X, obs.cond.values, obs.donor.values, obs.celltype.values, control_label="ctrl")
    treated = (obs.cond != "ctrl").to_numpy()
    assert X_in.shape == (treated.sum(), 2 * X.shape[1]) and list(classes) == ["stim"]
    i = np.flatnonzero(treated)[0]
    d, k = obs.donor.iloc[i], obs.celltype.iloc[i]
    ctrl = obs.cond == "ctrl"
    ck = X[(ctrl & (obs.celltype == k)).to_numpy()].mean(0)
    m = (ctrl & (obs.celltype == k) & (obs.donor == d)).to_numpy()
    cdk = X[m].mean(0) if m.any() else ck
    np.testing.assert_allclose(X_in[0], np.concatenate([X[i] - (cdk - ck), X[i] - cdk]), rtol=1e-6)


def test_predict_end_to_end(tmp_path):
    X, obs = make_data()
    classes = ["IL6", "IL7", "IFNB1"]
    torch.manual_seed(0)
    net = cytap.CytokineClassifier(2 * X.shape[1], len(classes), (32, 16), 0.3)
    path = tmp_path / "m.pt"
    cytap.save_model(path, net, classes, {"hidden_dims": [32, 16], "embedding_key": "X_emb"})
    model = cytap.load_model(path, device="cpu")
    adata = AnnData(X=np.zeros((len(obs), 1), np.float32), obs=obs, obsm={"X_emb": X})

    out = cytap.predict(adata, model, design="paired", control_key="cond", control_label="ctrl",
                        donor_key="donor", celltype_key="celltype", batch_size=100)
    keep, feats = legacy_paired(X, obs, "donor", "celltype", "cond", "ctrl")
    np.testing.assert_allclose(out.obsm["pred_probs"], cytap.predict_proba(model, feats), atol=1e-6)
    assert list(out.obs_names) == list(obs.index[keep])
    assert out.uns["cytap"]["design"] == "paired" and out.uns["cytap"]["reference"] == "own controls"

    pooled = cytap.predict(adata, model, design="unpaired", control_key="cond", control_label="ctrl",
                           celltype_key="celltype")
    assert pooled.uns["cytap"]["reference"] == "pooled"
    kw = dict(control_key="cond", control_label="ctrl", celltype_key="celltype")
    with pytest.raises(ValueError, match="donor_key"):           # paired without donors
        cytap.predict(adata, model, design="paired", **kw)
    with pytest.raises(ValueError, match="donor_key"):           # counterfactual without donors
        cytap.predict(adata, model, design="unpaired", reference="counterfactual", **kw)
    with pytest.raises(ValueError, match="unpaired"):            # reference with paired data
        cytap.predict(adata, model, design="paired", reference="pooled", donor_key="donor", **kw)
    with pytest.raises(ValueError, match="counterfactual"):      # counterfactual source without reference
        cytap.predict(adata, model, design="unpaired", counterfactual="cellot", **kw)
    with pytest.raises(ValueError, match="design"):
        cytap.predict(adata, model, design="pooled", **kw)


def test_predict_requires_comp_model(tmp_path):
    X, obs = make_data()
    adata = AnnData(X=np.zeros((len(obs), 1), np.float32), obs=obs, obsm={"X_emb": X})
    net = cytap.CytokineClassifier(X.shape[1], 3, (8,), 0.1)
    cytap.save_model(tmp_path / "raw.pt", net, ["a", "b", "c"], {"hidden_dims": [8], "representation": "raw",
                                                                  "embedding_key": "X_emb"})
    with pytest.raises(ValueError, match="Comp"):
        cytap.predict(adata, cytap.load_model(tmp_path / "raw.pt", device="cpu"), design="paired",
                      control_key="cond", control_label="ctrl", donor_key="donor")


def test_predict_warns_about_dropped_cells(tmp_path):
    X, obs = make_data(na=False)
    obs = obs.copy()
    obs.iloc[:5, obs.columns.get_loc("celltype")] = "NA"          # missing labels
    adata = AnnData(X=np.zeros((len(obs), 1), np.float32), obs=obs, obsm={"X_emb": X})
    net = cytap.CytokineClassifier(2 * X.shape[1], 3, (8,), 0.1)
    cytap.save_model(tmp_path / "m.pt", net, ["a", "b", "c"], {"hidden_dims": [8], "embedding_key": "X_emb"})
    with pytest.warns(UserWarning, match="cells were not scored"):
        out = cytap.predict(adata, cytap.load_model(tmp_path / "m.pt", device="cpu"), design="unpaired",
                            control_key="cond", control_label="ctrl", celltype_key="celltype")
    d = out.uns["cytap"]["dropped_cells"]
    assert d["missing_label"] == 5 and out.n_obs == adata.n_obs - sum(d.values())

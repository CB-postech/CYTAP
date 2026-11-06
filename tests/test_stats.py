"""cytap.stats: equality with the statistics code of the paper analyses (condition comparison, severity
trend, within-donor change, pseudobulk; reimplemented below as legacy_*), BH-FDR and plots."""
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.stats import linregress, spearmanr, ttest_ind, ttest_rel

from cytap import stats

multipletests = pytest.importorskip("statsmodels.stats.multitest").multipletests
CLASSES = ["IL-6", "IL-7", "IFN-beta", "G-CSF", "TNF-alpha"]


def make_adata(n=4000, donors=6, seed=0, conds=("ctrl", "stim")):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({
        "donor": pd.Categorical(rng.choice([f"D{i}" for i in range(donors)], n)),
        "cond": pd.Categorical(rng.choice(list(conds), n)),
        "ct": pd.Categorical(rng.choice(["B", "CD4", "NK"], n)),
    }, index=[f"c{i}" for i in range(n)])
    logit = rng.normal(size=(n, len(CLASSES))) + (obs.cond == conds[-1]).to_numpy()[:, None] * [1, 0, 0.3, 0, -0.5]
    p = np.exp(logit) / np.exp(logit).sum(1, keepdims=True)
    a = AnnData(np.zeros((n, 1), np.float32), obs=obs)
    a.obsm["pred_probs"] = p.astype(np.float32)
    a.uns["pred_classes"] = np.array(CLASSES)
    return a


# ---------------------------------------------------------------- legacy copies
def legacy_stats(adata, ctrl_label, stim_label, condition_col, donor_col=None, cell_col=None, target_cell=None, paired=True):
    adata_sub = adata[adata.obs[cell_col] == target_cell].copy() if (cell_col and target_cell) else adata.copy()
    probs = adata_sub.obsm["pred_probs"]
    class_names = adata_sub.uns["pred_classes"]
    pc = np.clip(probs, 1e-6, 1 - 1e-6)
    logits = np.log(pc / (1 - pc))
    df_logits = pd.DataFrame(logits, columns=class_names)
    df_logits["Condition"] = adata_sub.obs[condition_col].values
    df_probs = pd.DataFrame(probs, columns=class_names)
    df_probs["Condition"] = adata_sub.obs[condition_col].values
    has_donors = donor_col and (donor_col in adata_sub.obs.columns)
    unique_donors = adata_sub.obs[donor_col].nunique() if has_donors else 0
    if has_donors:
        df_logits["Donor"] = adata_sub.obs[donor_col].values
        df_probs["Donor"] = adata_sub.obs[donor_col].values
    results, pv = [], []
    if has_donors and unique_donors > 1:
        level = "Donor-level"
        dl = df_logits.groupby(["Donor", "Condition"], observed=True).median().reset_index().dropna()
        dp = df_probs.groupby(["Donor", "Condition"], observed=True).median().reset_index().dropna()
        if paired:
            valid = list(set(dl[dl.Condition == stim_label].Donor) & set(dl[dl.Condition == ctrl_label].Donor))
    else:
        level, paired = "Single-donor", False
    for cyto in class_names:
        if level == "Donor-level":
            if paired:
                g = lambda m, c: np.array([m[(m.Donor == d) & (m.Condition == c)][cyto].values[0] for d in valid])
                sl, cl, sp, cp = g(dl, stim_label), g(dl, ctrl_label), g(dp, stim_label), g(dp, ctrl_label)
            else:
                sl, cl = dl[dl.Condition == stim_label][cyto].values, dl[dl.Condition == ctrl_label][cyto].values
                sp, cp = dp[dp.Condition == stim_label][cyto].values, dp[dp.Condition == ctrl_label][cyto].values
        else:
            sl, cl = df_logits[df_logits.Condition == stim_label][cyto].values, df_logits[df_logits.Condition == ctrl_label][cyto].values
            sp, cp = df_probs[df_probs.Condition == stim_label][cyto].values, df_probs[df_probs.Condition == ctrl_label][cyto].values
        eff = np.mean(sl) - np.mean(cl)
        l2 = np.log2((np.mean(sp) + 1e-9) / (np.mean(cp) + 1e-9))
        if level == "Single-donor" or len(sl) < 2 or len(cl) < 2:
            p = 1.0
        elif paired:
            p = ttest_rel(sl, cl, alternative="greater")[1]
        else:
            p = ttest_ind(sl, cl, alternative="greater", equal_var=False)[1]
        pv.append(p)
        results.append({"Cytokine": cyto, "Effect_Size (Delta Logit)": eff, "Log2FC (Probability)": l2})
    _, fdr, _, _ = multipletests(np.nan_to_num(pv, nan=1.0), method="fdr_bh")
    out = pd.DataFrame(results)
    out["T-test_FDR"] = fdr
    return out


def legacy_trend(adata, condition_col, donor_col, severity_order, cell_col=None, target_cell=None):
    adata_sub = adata[adata.obs[cell_col] == target_cell].copy() if target_cell else adata.copy()
    adata_sub.obs[condition_col] = pd.Categorical(adata_sub.obs[condition_col], categories=severity_order, ordered=True)
    smap = {l: i for i, l in enumerate(severity_order)}
    adata_sub.obs["severity_score"] = adata_sub.obs[condition_col].map(smap)
    adata_sub = adata_sub[~adata_sub.obs["severity_score"].isna()].copy()
    probs = adata_sub.obsm["pred_probs"]
    logits = np.log(np.clip(probs, 1e-6, 1 - 1e-6) / (1 - np.clip(probs, 1e-6, 1 - 1e-6)))
    df = pd.DataFrame(logits, columns=adata_sub.uns["pred_classes"])
    df["severity_score"] = adata_sub.obs["severity_score"].values
    df["donor_id"] = adata_sub.obs[donor_col].values
    dm = df.groupby(["donor_id", "severity_score"], observed=True).median().reset_index()
    rows = []
    for c in adata_sub.uns["pred_classes"]:
        rho, p = spearmanr(dm["severity_score"], dm[c])
        rows.append({"Cytokine": c, "Correlation_Rho": rho, "Regression_Slope": linregress(dm["severity_score"], dm[c])[0], "P_value": p})
    r = pd.DataFrame(rows)
    _, r["FDR"], _, _ = multipletests(r["P_value"], method="fdr_bh")
    return r


def legacy_delta(adata, target_cytokine, baseline_cond, target_conds, cell_col, donor_col, condition_col):
    raw = adata.obsm["pred_probs"][:, list(adata.uns["pred_classes"]).index(target_cytokine)]
    pc = np.clip(raw, 1e-6, 1 - 1e-6)
    df = pd.DataFrame({"Group": adata.obs[cell_col].values, "Donor": adata.obs[donor_col].values,
                       "Condition": adata.obs[condition_col].values, "Value": np.log(pc / (1 - pc))})
    df = df[df.Condition.isin([baseline_cond] + target_conds)]
    m = df.groupby(["Group", "Donor", "Condition"], observed=True)["Value"].median().reset_index().dropna()
    w = m.pivot(index=["Group", "Donor"], columns="Condition", values="Value").dropna(subset=[baseline_cond])
    parts = []
    for t in target_conds:
        d = (w[t] - w[baseline_cond]).dropna().reset_index(name="Delta")
        d["Target_Condition"] = t
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def legacy_response(adata, cyto, cell_col, donor_col, condition_col, conditions_to_plot):
    raw = adata.obsm["pred_probs"][:, list(adata.uns["pred_classes"]).index(cyto)]
    pc = np.clip(raw, 1e-6, 1 - 1e-6)
    df = pd.DataFrame({"Group": adata.obs[cell_col].values, "Donor": adata.obs[donor_col].values,
                       "Condition": adata.obs[condition_col].values, "Value": np.log(pc / (1 - pc))})
    df = df[df.Condition.isin(conditions_to_plot)].copy()
    df["Condition"] = pd.Categorical(df.Condition, categories=conditions_to_plot, ordered=True)
    m = df.groupby(["Group", "Donor", "Condition"], observed=True)["Value"].median().reset_index().dropna()
    cc = m.groupby("Group", observed=True)["Condition"].nunique()
    return m[m.Group.isin(cc[cc == len(conditions_to_plot)].index)]


# ---------------------------------------------------------------- tests
@pytest.mark.parametrize("paired", [True, False])
@pytest.mark.parametrize("celltype", [None, "CD4"])
def test_compare_matches_legacy(paired, celltype):
    a = make_adata()
    old = legacy_stats(a, "ctrl", "stim", "cond", "donor", "ct" if celltype else None, celltype, paired)
    new = stats.compare_conditions(a, "cond", "ctrl", "stim", donor_key="donor",
                                celltype_key="ct", celltype=celltype, paired=paired)
    assert list(new.cytokine) == list(old.Cytokine)
    # legacy pairs donors in list(set(...)) order, which varies between runs -> float32 rounding
    np.testing.assert_allclose(new.delta_logit, old["Effect_Size (Delta Logit)"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(new.log2fc_prob, old["Log2FC (Probability)"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(new.fdr, old["T-test_FDR"], rtol=1e-5)
    assert new.loc[0, "fdr"] < 0.05 and new.loc[0, "delta_logit"] > 0


def test_compare_single_donor_has_no_pvalue():
    a = make_adata(donors=1)
    new = stats.compare_conditions(a, "cond", "ctrl", "stim", donor_key="donor")
    old = legacy_stats(a, "ctrl", "stim", "cond", "donor")
    assert (new.pval == 1).all() and new.attrs["level"] == "cell"
    np.testing.assert_allclose(new.delta_logit, old["Effect_Size (Delta Logit)"], rtol=1e-6)


def test_trend_matches_legacy():
    order = ["HV", "MILD", "SEV", "CRIT"]
    a = make_adata(n=6000, donors=24, conds=order)
    donor_cond = {d: order[i % 4] for i, d in enumerate(a.obs.donor.cat.categories)}  # one condition per donor
    cond = a.obs.donor.map(donor_cond).astype(str).to_numpy()
    cond[:300] = "Sepsis"  # condition outside the order is ignored
    a.obs["cond"] = pd.Categorical(cond)
    for ct in [None, "NK"]:
        old = legacy_trend(a, "cond", "donor", order, "ct", ct)
        new = stats.severity_trend(a, "cond", order, "donor", celltype_key="ct", celltype=ct)
        np.testing.assert_allclose(new.rho, old.Correlation_Rho, rtol=1e-6)
        np.testing.assert_allclose(new.slope, old.Regression_Slope, rtol=1e-5)
        np.testing.assert_allclose(new.fdr, old.FDR, rtol=1e-6)
    with pytest.raises(ValueError, match="no donors"):
        stats.severity_trend(a, "cond", order + ["X"], "donor")


def test_paired_delta_and_pseudobulk_match_legacy():
    a = make_adata(conds=("ctrl", "s1", "s2"))
    old = legacy_delta(a, "IL-6", "ctrl", ["s1", "s2"], "ct", "donor", "cond")
    new = stats.paired_delta(a, "IL-6", "cond", "ctrl", ["s1", "s2"], "donor", "ct")
    np.testing.assert_allclose(new.delta.to_numpy(), old.Delta.to_numpy(), rtol=1e-6)
    assert list(new.condition.astype(str)) == list(old.Target_Condition)

    a.obs.loc[(a.obs.ct == "NK") & (a.obs.cond == "s2"), "cond"] = "s1"
    old = legacy_response(a, "IL-6", "ct", "donor", "cond", ["ctrl", "s1", "s2"])
    new = stats.pseudobulk(a, "donor", "cond", "ct", cytokines=["IL-6"], conditions=["ctrl", "s1", "s2"],
                        require_all_conditions=True)
    np.testing.assert_allclose(new["IL-6"].to_numpy(), old.Value.to_numpy(), rtol=1e-6)
    assert "NK" not in set(new.ct)


def test_bh_matches_statsmodels():
    p = np.random.default_rng(1).uniform(size=200) ** 3
    p[:10] = p[10]  # ties
    np.testing.assert_allclose(stats.bh_fdr(p), multipletests(p, method="fdr_bh")[1])


def test_plots_run(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from cytap import plotting
    a = make_adata(conds=("ctrl", "s1", "s2"))
    a.obsm["X_umap"] = np.random.default_rng(0).normal(size=(a.n_obs, 2))
    res = stats.compare_conditions(a, "cond", "ctrl", "s1", donor_key="donor")
    plotting.lollipop(res, fdr=None)
    plotting.response(a, "IL-6", "donor", "cond", "ct", conditions=["ctrl", "s1", "s2"])
    plotting.delta(a, "IL-6", "cond", "ctrl", ["s1", "s2"], "donor")
    plotting.delta(a, "IL-6", "cond", "ctrl", "s1", "donor", "ct")
    ax = plotting.embedding(a, "IL-6")
    ax.figure.savefig(tmp_path / "e.png")

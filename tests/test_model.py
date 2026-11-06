"""cytap.model: pretrained model download and cache."""
import hashlib

import pytest

import cytap
from cytap import model as M


def _ckpt(tmp_path):
    net = M.CytokineClassifier(8, 3, hidden_dims=(4,))
    f = tmp_path / "src.pt"
    M.save_model(f, net, ["a", "b", "c"], {"hidden_dims": [4]})
    return f


def test_pretrained_download_and_cache(tmp_path, monkeypatch):
    src = _ckpt(tmp_path)
    monkeypatch.setenv("CYTAP_CACHE", str(tmp_path / "cache"))
    monkeypatch.setitem(M.PRETRAINED, "toy", {"url": src.as_uri(), "sha256": hashlib.sha256(src.read_bytes()).hexdigest()})
    m = cytap.load_model("toy", device="cpu")
    assert m.classes == ["a", "b", "c"] and (tmp_path / "cache" / "toy.pt").exists()
    src.unlink()                                   # second call uses the cache
    assert cytap.load_model("toy", device="cpu").classes == ["a", "b", "c"]


def test_pretrained_checksum_mismatch(tmp_path, monkeypatch):
    src = _ckpt(tmp_path)
    monkeypatch.setenv("CYTAP_CACHE", str(tmp_path / "cache"))
    monkeypatch.setitem(M.PRETRAINED, "toy", {"url": src.as_uri(), "sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="checksum"):
        cytap.load_model("toy", device="cpu")
    assert not (tmp_path / "cache" / "toy.pt").exists()


def test_local_path_still_works(tmp_path):
    assert cytap.load_model(str(_ckpt(tmp_path)), device="cpu").classes == ["a", "b", "c"]


def test_load_example_cached(tmp_path, monkeypatch):
    """load_example("ifnb") reads the cached release file; release files that are not models are refused."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    monkeypatch.setenv("CYTAP_CACHE", str(tmp_path))
    a = ad.AnnData(obs=pd.DataFrame({"stim": ["ctrl", "stim"]}, index=["c1", "c2"]))
    a.obsm["X_state"] = np.zeros((2, 4), np.float32)
    a.write_h5ad(tmp_path / "ifnb_example.h5ad")
    b = cytap.load_example("ifnb")
    assert list(b.obs["stim"]) == ["ctrl", "stim"] and b.obsm["X_state"].shape == (2, 4)
    with pytest.raises(ValueError, match="not a CYTAP model"):
        cytap.load_model("ifnb_example")
    with pytest.raises(ValueError, match="unknown example"):
        cytap.load_example("combat")

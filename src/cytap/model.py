"""CYTAP classifier and checkpoint I/O."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CONFIG = {
    "hidden_dims": [2048, 1024],
    "dropout": 0.3,
    "representation": "comp",
    "embedding_key": "X_state",
    "embedding_model": "State SE-600M",
}


class CytokineClassifier(nn.Module):
    """MLP: [Linear -> BatchNorm -> ReLU -> Dropout] x len(hidden_dims) -> Linear."""

    def __init__(self, input_dim, num_classes, hidden_dims=(2048, 1024), dropout_rate=0.3):
        super().__init__()
        layers, in_dim = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout_rate)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        loss = self.alpha * (1 - torch.exp(-ce)) ** self.gamma * ce
        return loss.mean() if self.reduction == "mean" else loss.sum()


def _strip_compile_prefix(state_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def save_model(path, model: nn.Module, classes, config: dict | None = None):
    """Save weights, class names and architecture in one file."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    lin = [m for m in model.modules() if isinstance(m, nn.Linear)]
    cfg["input_dim"] = lin[0].in_features
    torch.save(
        {"state_dict": _strip_compile_prefix(model.state_dict()), "classes": [str(c) for c in classes], "config": cfg},
        path,
    )


# Files of the GitHub release, downloaded on first use to $CYTAP_CACHE (default ~/.cache/cytap): the CYTAP
# model (load_model), the COMBAT CellOT maps of the paper (counterfactual.CellOT.load), example data (load_example).
RELEASE = "https://github.com/CB-postech/CYTAP/releases/download/v0.1.0"
PRETRAINED = {
    "cytap_state_comp_v1": {
        "url": f"{RELEASE}/cytap_state_comp_v1.pt",
        "sha256": "512bd260eed8d210b7447385ef91577ce533ee4814e6a8c6d641c0e01e2c9658",
    },
    "combat_cellot": {
        "url": f"{RELEASE}/combat_cellot.pt",
        "kind": "cellot",
        "sha256": "91bbed50f0b242ad1fb6019385f0cf75aaa54f64a54740741de94099516694c4",
    },
    "ifnb_example": {
        "url": f"{RELEASE}/ifnb_example.h5ad",
        "kind": "data",
        "file": "ifnb_example.h5ad",
        "sha256": "8796d1c513185650e6801bb1a8de739b279601c76c7fc666e0e671dcb1b5f9ef",
    },
}
DEFAULT_MODEL = "cytap_state_comp_v1"
LICENSE_NOTICE = ("CYTAP model weights: CC BY-NC 4.0 and the Arc Institute State Model Non-Commercial License "
                  "(non-commercial use only; see MODEL_LICENSE.md in https://github.com/CB-postech/CYTAP).")


def cache_dir() -> Path:
    return Path(os.environ.get("CYTAP_CACHE", Path.home() / ".cache" / "cytap"))


def model_path(name: str = DEFAULT_MODEL) -> Path:
    """Local file of a release file (PRETRAINED); downloaded (and checksum-verified) if not cached yet."""
    if name not in PRETRAINED:
        raise ValueError(f"unknown pretrained file '{name}'; available: {', '.join(PRETRAINED)}")
    info = PRETRAINED[name]
    dst = cache_dir() / info.get("file", f"{name}.pt")
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} -> {dst}" + ("" if "kind" in info else f"\n{LICENSE_NOTICE}"), file=sys.stderr)
        tmp = dst.with_name(dst.name + ".part")
        torch.hub.download_url_to_file(info["url"], str(tmp), progress=True)
        if info["sha256"]:
            import hashlib
            h = hashlib.sha256(tmp.read_bytes()).hexdigest()
            if h != info["sha256"]:
                tmp.unlink()
                raise RuntimeError(f"checksum mismatch for {name}: {h}")
        os.replace(tmp, dst)
    return dst


def load_model(path=DEFAULT_MODEL, device: str | None = None) -> CytokineClassifier:
    """Load a CYTAP checkpoint: a file path, or the name of a pretrained model (default
    ``"cytap_state_comp_v1"``, downloaded on first use). Class names are in ``model.classes``,
    settings in ``model.config``."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not Path(path).exists() and str(path) in PRETRAINED:
        if "kind" in PRETRAINED[str(path)]:
            raise ValueError(f"'{path}' is not a CYTAP model")
        path = model_path(str(path))
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = CytokineClassifier(cfg["input_dim"], len(ckpt["classes"]), cfg["hidden_dims"], cfg["dropout"])
    model.load_state_dict(_strip_compile_prefix(ckpt["state_dict"]))
    model.classes = list(ckpt["classes"])
    model.config = cfg
    return model.to(device).eval()


def load_example(name: str = "ifnb"):
    """Example query data (obs + ``obsm['X_state']``), downloaded on first use.
    ``"ifnb"``: IFN-β-stimulated PBMCs (Kang et al. 2018; 8 donors, columns ``ind``, ``stim``, ``cell``)."""
    import anndata as ad
    key = f"{name}_example"
    if PRETRAINED.get(key, {}).get("kind") != "data":
        raise ValueError(f"unknown example '{name}'; available: ifnb")
    return ad.read_h5ad(model_path(key))

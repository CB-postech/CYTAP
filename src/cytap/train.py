"""Reference input construction and classifier training (Parse-PBMC)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .features import compose, compute_centroids
from .model import CytokineClassifier, FocalLoss, save_model

TRAIN_DEFAULTS = {
    "hidden_dims": (2048, 1024),
    "dropout": 0.3,
    "learning_rate": 1.4351e-4,
    "focal_gamma": 2.0,
    "batch_size": 4096,
    "epochs": 10,
    "pct_start": 0.3,
    "seed": 42,
}


def build_reference_inputs(
    X: np.ndarray,
    condition,
    donor,
    celltype,
    control_label: str = "PBS",
):
    """Inputs for cytokine-treated reference cells (controls excluded).

    c_k: control cells of cell type k pooled over donors.
    c_{d,k}: control cells of donor d, cell type k (c_k if none).

    Returns X_in, y, classes (sorted), donor, celltype for the treated cells.
    """
    condition = np.asarray(condition).astype(str)
    is_ctrl = condition == control_label
    cents = compute_centroids(X, is_ctrl, donor=donor, celltype=celltype, fallback=True)
    treated = ~is_ctrl[cents.keep]
    rows = np.flatnonzero(cents.keep)[treated]
    ck, cdk = cents.rows(treated)
    X_in = compose(np.asarray(X)[rows], ck, cdk)
    classes, y = np.unique(condition[rows], return_inverse=True)
    return X_in, y.astype(np.int64), classes, np.asarray(donor)[rows], np.asarray(celltype)[rows]


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    classes,
    *,
    save_path: str | None = None,
    save_epochs: tuple = (),
    config: dict | None = None,
    device: str | None = None,
    compile: bool = True,
    num_workers: int = 4,
    log_fn: Callable[[dict], None] | None = None,
    epoch_callback: Callable[[CytokineClassifier, int], dict] | None = None,
    preshuffle: bool = True,
    drop_last: bool = False,
    **params,
) -> CytokineClassifier:
    """Train the classifier.

    Focal loss (alpha=1, gamma), AdamW, one-cycle LR schedule. With
    ``preshuffle`` the cells are shuffled once with ``seed`` before batching.
    ``epoch_callback(model, epoch)`` runs after each epoch (model in eval mode)
    and its returned metrics are added to the epoch log.
    ``save_epochs`` writes ``{save_path stem}_epoch{e}.pt`` at those epochs.
    """
    p = {**TRAIN_DEFAULTS, **params}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.int64))
    if preshuffle:
        perm = np.arange(len(y))
        np.random.RandomState(p["seed"]).shuffle(perm)
        X_t, y_t = X_t[perm], y_t[perm]
    loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=p["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=drop_last,
    )

    torch.manual_seed(p["seed"])
    model = CytokineClassifier(X_t.shape[1], len(classes), p["hidden_dims"], p["dropout"]).to(device)
    net = torch.compile(model, mode="reduce-overhead") if compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=p["learning_rate"], total_steps=len(loader) * p["epochs"], pct_start=p["pct_start"]
    )
    criterion = FocalLoss(gamma=p["focal_gamma"]).to(device)
    cfg = {**(config or {}), "hidden_dims": list(p["hidden_dims"]), "dropout": p["dropout"], "train_params": p}

    for epoch in range(1, p["epochs"] + 1):
        net.train()
        tot_loss = n = top1 = top5 = 0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad()
            logits = net(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item()
            t5 = logits.topk(5, dim=1).indices
            top1 += (t5[:, 0] == yb).sum().item()
            top5 += (t5 == yb[:, None]).any(1).sum().item()
            n += len(yb)
        stats = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": tot_loss / len(loader),
                 "train_accuracy": 100 * top1 / n, "train_accuracy_top5": 100 * top5 / n}
        if epoch_callback:
            model.eval()
            stats.update(epoch_callback(model, epoch))
        print(f"Epoch {epoch:02d}/{p['epochs']} | loss {stats['train_loss']:.4f} | acc {stats['train_accuracy']:.2f}%")
        if log_fn:
            log_fn(stats)
        if save_path and epoch in save_epochs:
            stem = save_path[:-3] if save_path.endswith(".pt") else save_path
            save_model(f"{stem}_epoch{epoch}.pt", model, classes, cfg)

    if save_path:
        save_model(save_path, model, classes, cfg)
    return model.eval()

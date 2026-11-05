# train_cnn.py
"""Compact 1D CNN for PROTECT-90 fault classification.

Data is already normalized by preprocess.py (per-channel z-score fit on train
windows only). This script applies that normalization exactly once by reading
the already-normalized binary splits and never re-fits statistics.

Architecture: Conv1d(48->32->64->96) + BatchNorm + ReLU + MaxPool +
AdaptiveAvgPool1d(1) + Linear(96, 4).

Training runs on CPU (no CUDA assumed). A reproducible, stratified,
class-balanced training subset is selected before any training or probe. Valid
and test splits are evaluated batch-wise from memory-mapped files.
"""

from pathlib import Path
import struct
import sys
import json
import traceback
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
)

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))
import preprocess as P

PREPROC = Path("preprocessed")
OUT_DIR = Path("cnn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["LLL", "LL", "SLG", "LLG"]
N_CLASSES = 4
WINDOW_SIZE = 256
N_CHANNELS = 48

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CPU thread discipline: avoid oversubscribing on this host.
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

SUBSET_TARGET = 16384
SUBSET_SEED = 42
BATCH = 128
MAX_EPOCHS = 20
PATIENCE = 5


class MmDataset(Dataset):
    """Genuine memory-mapped window dataset backed by the raw binary splits.

    It does not materialize the full ``(N, 256, 48)`` arrays. Only individual
    windows requested by the loader are copied.
    """

    def __init__(self, x_path, y_path):
        sx = Path(str(x_path) + ".x.bin")
        sy = Path(str(y_path) + ".y.bin")
        with open(sx, "rb") as f:
            n, ws, nc = struct.unpack("<III", f.read(12))
            if ws != WINDOW_SIZE or nc != N_CHANNELS:
                raise ValueError(f"header mismatch: ws={ws} nc={nc}")
            body_x = f.read()
        with open(sy, "rb") as f:
            ny = struct.unpack("<I", f.read(4))[0]
            body_y = f.read()
        if n != ny:
            raise ValueError(f"header mismatch: n_x={n} n_y={ny}")
        self.n = n
        self._x_base = np.memmap(sx, dtype=np.float32, mode="r", offset=12)
        self.x_mm = self._x_base.reshape(n, WINDOW_SIZE, N_CHANNELS)
        self._y_base = np.memmap(sy, dtype=np.int64, mode="r", offset=4)
        assert self._y_base.shape[0] == n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = np.asarray(self.x_mm[idx], dtype=np.float32, order="C")
        y = int(self._y_base[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def select_subset_indices(n, y, target=SUBSET_TARGET, seed=SUBSET_SEED):
    """Reproducible stratified, class-balanced subset indices.

    Returns selected indices into the full train split.
    """
    rng = np.random.default_rng(seed)
    per_class = []
    for c in range(N_CLASSES):
        idx_c = np.where(y == c)[0]
        k = max(1, int(round(target * (idx_c.size / n))))
        k = min(k, int(idx_c.size))
        per_class.append(rng.choice(idx_c, size=k, replace=False))
    sel = np.concatenate(per_class)
    rng.shuffle(sel)
    return sel


def evaluate_from_loader(model, loader, device):
    """Evaluate from a batch-wise loader. Returns full metrics incl. confusion."""
    model.eval()
    preds, probs, trues = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(yb.numpy())
    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    acc = float(accuracy_score(trues, preds))
    macro_f1 = float(f1_score(trues, preds, average="macro"))
    pr, rc, fs, _ = precision_recall_fscore_support(
        trues, preds, labels=np.arange(N_CLASSES), zero_division=0
    )
    cm = confusion_matrix(trues, preds, labels=np.arange(N_CLASSES))
    per_class = {
        CLASS_NAMES[i]: {
            "precision": float(pr[i]),
            "recall": float(rc[i]),
            "f1": float(fs[i]),
            "support": int((trues == i).sum()),
        }
        for i in range(N_CLASSES)
    }
    conf = float(probs.max(axis=1).mean())
    cm_img = np.array(cm, dtype=np.int64)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "mean_confidence": conf,
    }, cm_img


class Small1DCNN(nn.Module):
    def __init__(self, n_channels=N_CHANNELS, n_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(96, n_classes),
        )

    def forward(self, x):
        return self.net(x.permute(0, 2, 1).contiguous())


def count_params(model):
    return sum(int(p.numel()) for p in model.parameters())


def _plot_confusion(ax, cm, title):
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(N_CLASSES),
        yticks=range(N_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True",
        xlabel="Predicted",
        title=title,
    )
    thresh = cm.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )


def main():
    try:
        print("INFO: data is already normalized by preprocess.py; no re-normalization here.", flush=True)

        print("Opening memory-mapped splits...", flush=True)
        train_full = MmDataset(PREPROC / "X_train", PREPROC / "X_train")
        val_ds = MmDataset(PREPROC / "X_val", PREPROC / "X_val")
        test_ds = MmDataset(PREPROC / "X_test", PREPROC / "X_test")

        n_train = train_full.n
        ytr = np.asarray(train_full._y_base, dtype=np.int64).copy()
        use_subset = n_train > SUBSET_TARGET
        if use_subset:
            print("Full train split has %d windows; selecting stratified subset of %d." %
                  (n_train, SUBSET_TARGET), flush=True)
            sel = select_subset_indices(n_train, ytr, target=SUBSET_TARGET, seed=SUBSET_SEED)
            subset_n = int(sel.size)
            ytr_sub = ytr[sel].copy()
            # Build a dataset over the subset indices; it still reads from mmaps.
            class SubsetMmDataset(Dataset):
                def __init__(self, base, idx):
                    self.base = base
                    self.idx = np.asarray(idx, dtype=np.int64)
                def __len__(self):
                    return len(self.idx)
                def __getitem__(self, i):
                    return self.base[self.idx[i]]
            train_ds = SubsetMmDataset(train_full, sel)
            del sel
        else:
            print("Using full train split: %d windows." % n_train, flush=True)
            subset_n = n_train
            train_ds = train_full

        print("Subset/train windows used: %d" % subset_n, flush=True)
        del ytr
        if not use_subset:
            del ytr_sub

        train_loader = DataLoader(
            train_ds, batch_size=BATCH, shuffle=True, drop_last=False, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH, shuffle=False, num_workers=0
        )

        model = Small1DCNN().to(DEVICE)
        print("Parameters:", count_params(model), flush=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        best_val_loss = float("inf")
        best_state = None
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": [],
        }
        wait = 0
        epochs_run = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            t0 = time.time()
            model.train()
            train_loss = 0.0
            n = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item()) * xb.size(0)
                n += xb.size(0)
            train_loss /= max(n, 1)

            model.eval()
            val_loss = 0.0
            correct = 0
            vn = 0
            val_preds, val_probs, val_trues = [], [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_loss += float(loss.item()) * xb.size(0)
                    pred = logits.argmax(dim=1)
                    correct += int((pred == yb).sum().item())
                    vn += xb.size(0)
                    val_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                    val_preds.append(pred.cpu().numpy())
                    val_trues.append(yb.cpu().numpy())
            val_loss /= max(vn, 1)
            val_acc = correct / max(vn, 1)
            val_preds = np.concatenate(val_preds)
            val_trues = np.concatenate(val_trues)
            val_macro_f1 = float(f1_score(val_trues, val_preds, average="macro"))

            scheduler.step(val_loss)
            elapsed = time.time() - t0
            epochs_run = epoch
            line = ("Epoch %02d train_loss=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f %.1fs"
                    % (epoch, train_loss, val_loss, val_acc, val_macro_f1, elapsed))
            print(line, flush=True)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_macro_f1"].append(val_macro_f1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    print("Early stopping triggered at epoch %d." % epoch, flush=True)
                    break

        # Load best checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), OUT_DIR / "best_model.pt")

        # Evaluate val and test once, batch-wise
        (val_metrics, val_cm) = evaluate_from_loader(model, val_loader, DEVICE)
        (test_metrics, test_cm) = evaluate_from_loader(model, test_loader, DEVICE)

        (OUT_DIR / "cnn_test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2), encoding="utf-8"
        )
        (OUT_DIR / "cnn_val_metrics.json").write_text(
            json.dumps(val_metrics, indent=2), encoding="utf-8"
        )
        (OUT_DIR / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        # Confusion matrices
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        _plot_confusion(ax, test_cm, "1D CNN - Test")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "cnn_confusion_matrix.png", dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        _plot_confusion(ax, val_cm, "1D CNN - Validation")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "cnn_val_confusion_matrix.png", dpi=130)
        plt.close(fig)

        config = {
            "device": str(DEVICE),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "batch_size": BATCH,
            "optimizer": "AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "loss": "CrossEntropyLoss",
            "max_epochs": MAX_EPOCHS,
            "epochs_run": epochs_run,
            "early_stopped": wait >= PATIENCE,
            "use_class_balanced_subset": use_subset,
            "subset_windows": int(subset_n),
            "normalization": "already normalized by preprocess.py; no re-fitting here",
            "input_shape": [None, WINDOW_SIZE, N_CHANNELS],
            "n_parameters": count_params(model),
            "classes": CLASS_NAMES,
        }
        (OUT_DIR / "cnn_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

        print("Saved:", OUT_DIR / "best_model.pt")
        print("Saved:", OUT_DIR / "cnn_test_metrics.json")
        print("Saved:", OUT_DIR / "training_history.json")
        print("Saved:", OUT_DIR / "cnn_val_metrics.json")
        print("Saved:", OUT_DIR / "cnn_config.json")
        print("Saved:", OUT_DIR / "cnn_confusion_matrix.png")
        print("Saved:", OUT_DIR / "cnn_val_confusion_matrix.png")
        print("Validation metrics:", json.dumps(val_metrics, indent=2))
        print("Test metrics:", json.dumps(test_metrics, indent=2))

    except Exception:
        print("ERROR in train_cnn.py:")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()

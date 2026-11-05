# train_hybrid.py
"""VIDYUT-KAVACHA: practical CNN-Random Forest hybrid pipeline (PROTECT-90).

Three experiments on the EXISTING episode-level train/validation/test splits:

  A. Random Forest on engineered electrical features (192 = 48 ch x 4 stats).
  B. Compact 1D CNN classifier over 48-channel waveform windows.
  C. Hybrid: 96-D CNN embedding + 192 engineered features -> Random Forest.

Protocol
--------
* Splits are the stored episode-level splits (preprocessed/*_episodes.csv);
  windows from one episode never cross splits (enforced in preprocess.py).
* Windows are already per-channel z-scored (fit on train only).  Nothing in
  this script re-applies normalisation.
* Data are read through numpy memmap, batch-wise.  No full-dataset copies.
* Training uses a fixed-seed, class-balanced stratified subset; its size is
  recorded in every metrics file.
* The CNN trains with early stopping on validation loss; the best validation
  checkpoint is used for embeddings.
* All model selection uses the validation split.  By default the test split is
  touched exactly once, for the single model selected on validation.
  Pass --test-all to also score A and B on test (documented deviation).

Run:
    python train_hybrid.py            # strict protocol (default)
    python train_hybrid.py --test-all # also evaluate A and B on test
"""

from pathlib import Path
import argparse
import json
import struct
import sys
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (  # noqa: E402
    CNN_EMBED_DIM,
    N_CHANNELS,
    N_ENG_FEATURES,
    embeddings_batched,
    features_batched,
    load_norm,
)

warnings.filterwarnings("ignore")

PREPROC = Path("preprocessed")
OUT_DIR = Path("hybrid_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["LLL", "LL", "SLG", "LLG"]
N_CLASSES = 4
WINDOW_SIZE = 256

# ---- reproducibility / budget ------------------------------------------------
SEED = 42
SUBSET_TARGET = 16384          # class-balanced, stratified, fixed seed
BATCH = 128
MAX_EPOCHS = 20
PATIENCE = 5
RF_N_ESTIMATORS = 400
RF_MIN_SAMPLES_LEAF = 2
RF_CLASS_WEIGHT = "balanced"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8)


# ----------------------------------------------------------------------------
# Memory-mapped split access
# ----------------------------------------------------------------------------
class Split:
    """Read-only memmap view of one preprocessed split."""

    def __init__(self, stem):
        x_path = Path(str(stem) + ".x.bin")
        y_path = Path(str(stem) + ".y.bin")
        with open(x_path, "rb") as f:
            n, ws, nc = struct.unpack("<III", f.read(12))
        with open(y_path, "rb") as f:
            ny = struct.unpack("<I", f.read(4))[0]
        if ws != WINDOW_SIZE or nc != N_CHANNELS:
            raise ValueError(f"{stem}: header mismatch ws={ws} nc={nc}")
        if n != ny:
            raise ValueError(f"{stem}: n_x={n} != n_y={ny}")
        self.n = n
        self.x = np.memmap(x_path, dtype=np.float32, mode="r", offset=12).reshape(n, ws, nc)
        self.y = np.memmap(y_path, dtype=np.int64, mode="r", offset=4)

    def labels(self, idx=None):
        return np.asarray(self.y if idx is None else self.y[idx], dtype=np.int64)


def stratified_subset(y, target, seed):
    """Proportional stratified sample of window indices (class-balanced)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    parts = []
    for c in range(N_CLASSES):
        idx_c = np.flatnonzero(y == c)
        k = min(int(round(target * idx_c.size / n)), idx_c.size)
        parts.append(rng.choice(idx_c, size=max(k, 1), replace=False))
    sel = np.sort(np.concatenate(parts))
    rng.shuffle(sel)
    return sel


# ----------------------------------------------------------------------------
# Compact 1D CNN
# ----------------------------------------------------------------------------
class Small1DCNN(nn.Module):
    def __init__(self, n_channels=N_CHANNELS, n_classes=N_CLASSES, embed_dim=CNN_EMBED_DIM):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.classifier(self.feature_net(x.permute(0, 2, 1).contiguous()))

    @torch.no_grad()
    def embed(self, x):
        return self.feature_net(x.permute(0, 2, 1).contiguous())


def count_params(model):
    return sum(int(p.numel()) for p in model.parameters())


def iter_batches(indices, batch_size, shuffle, rng):
    if shuffle:
        indices = indices[rng.permutation(len(indices))]
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


# ----------------------------------------------------------------------------
# Metrics / plots
# ----------------------------------------------------------------------------
def evaluate(y_true, y_pred, y_prob=None):
    pr, rc, fs, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(N_CLASSES), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(N_CLASSES))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(pr.mean()),
        "macro_recall": float(rc.mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(pr[i]),
                "recall": float(rc[i]),
                "f1": float(fs[i]),
                "support": int((y_true == i).sum()),
            }
            for i in range(N_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=np.arange(N_CLASSES),
            target_names=CLASS_NAMES, zero_division=0,
        ),
        "mean_confidence": float(np.max(y_prob, axis=1).mean()) if y_prob is not None else None,
    }


def save_confusion_matrix(path, cm, title):
    cm = np.asarray(cm, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)
    thresh = cm.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def plot_history(path, hist):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ep, hist["train_loss"], "o-", ms=3, label="Train loss")
    ax1.plot(ep, hist["val_loss"], "s-", ms=3, label="Validation loss")
    best = int(np.argmin(hist["val_loss"])) + 1
    ax1.axvline(best, ls="--", color="grey", lw=1, label=f"Best val loss (epoch {best})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.legend(loc="upper right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(ep, hist["val_acc"], "d-", ms=3, color="tab:green", label="Val accuracy")
    ax2.plot(ep, hist["val_macro_f1"], "v-", ms=3, color="tab:purple", label="Val macro F1")
    ax2.set_ylabel("Accuracy / macro F1")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="lower right", fontsize=8)
    plt.title("Compact 1D CNN: training history")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# CNN training
# ----------------------------------------------------------------------------
def train_cnn(train_split, train_idx, val_split, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Small1DCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    rng = np.random.default_rng(seed)
    hist = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "lr": []}
    best_loss, best_state, wait, epochs_run = float("inf"), None, 0, 0
    val_idx = np.arange(val_split.n)
    val_y = val_split.labels()

    print(f"CNN parameters: {count_params(model)}", flush=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        tot, nb = 0.0, 0
        for bi in iter_batches(train_idx, BATCH, True, rng):
            xb = torch.from_numpy(np.asarray(train_split.x[bi], dtype=np.float32)).to(DEVICE)
            yb = torch.from_numpy(train_split.labels(bi)).to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(bi)
            nb += len(bi)
        train_loss = tot / max(nb, 1)

        model.eval()
        vloss, correct = 0.0, 0
        preds = np.empty(len(val_idx), dtype=np.int64)
        ptr = 0
        with torch.no_grad():
            for bi in iter_batches(val_idx, 512, False, rng):
                xb = torch.from_numpy(np.asarray(val_split.x[bi], dtype=np.float32)).to(DEVICE)
                logits = model(xb)
                vloss += float(criterion(logits, torch.from_numpy(val_split.labels(bi)).to(DEVICE)).item()) * len(bi)
                p = logits.argmax(1).cpu().numpy()
                preds[ptr:ptr + len(bi)] = p
                ptr += len(bi)
                correct += int((p == val_split.labels(bi)).sum())
        vloss /= len(val_idx)
        vacc = correct / len(val_idx)
        vf1 = float(f1_score(val_y, preds, average="macro"))

        sched.step(vloss)
        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(vloss)
        hist["val_acc"].append(vacc)
        hist["val_macro_f1"].append(vf1)
        hist["lr"].append(float(opt.param_groups[0]["lr"]))
        epochs_run = epoch
        print(f"CNN epoch {epoch:02d} train_loss={train_loss:.4f} val_loss={vloss:.4f} "
              f"val_acc={vacc:.4f} val_macro_f1={vf1:.4f} {time.time() - t0:.1f}s", flush=True)

        if vloss < best_loss - 1e-5:
            best_loss, wait = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no val-loss improvement for {PATIENCE} epochs).", flush=True)
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, hist, best_loss, epochs_run, wait >= PATIENCE


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-all", action="store_true",
                    help="also evaluate experiments A and B on the test split")
    args = ap.parse_args()

    print("Windows are already z-scored by preprocess.py; no re-normalisation applied.", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    NORM_MEAN, NORM_STD = load_norm(str(PREPROC))
    print("Engineered features are computed in physical units (z-score inverted).", flush=True)

    train_s = Split(PREPROC / "X_train")
    val_s = Split(PREPROC / "X_val")
    test_s = Split(PREPROC / "X_test")
    print(f"Split windows -> train={train_s.n} val={val_s.n} test={test_s.n}", flush=True)

    y_train_all = train_s.labels()
    if train_s.n > SUBSET_TARGET:
        sub_idx = stratified_subset(y_train_all, SUBSET_TARGET, SEED)
    else:
        sub_idx = np.arange(train_s.n)
    y_sub = train_s.labels(sub_idx)
    print(f"Stratified subset: {len(sub_idx)} of {train_s.n} train windows (seed={SEED}); "
          f"per-class={np.bincount(y_sub, minlength=N_CLASSES).tolist()}", flush=True)

    val_idx = np.arange(val_s.n)
    y_val = val_s.labels()

    summary = {"protocol": {
        "subset_windows": int(len(sub_idx)),
        "train_windows_available": int(train_s.n),
        "subset_seed": SEED,
        "validation_windows": int(val_s.n),
        "test_windows": int(test_s.n),
        "test_evaluations": "all experiments (--test-all)" if args.test_all else "selected model only",
        "classes": CLASS_NAMES,
    }}

    # ---------------- Experiment A: RF on engineered features ----------------
    print("\n=== Experiment A: Random Forest on engineered features ===", flush=True)
    t0 = time.time()
    Xa_tr = features_batched(train_s.x, sub_idx, mean=NORM_MEAN, std=NORM_STD)
    Xa_val = features_batched(val_s.x, mean=NORM_MEAN, std=NORM_STD)
    print(f"Engineered features: {Xa_tr.shape}", flush=True)
    rf_a = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight=RF_CLASS_WEIGHT, random_state=SEED, n_jobs=-1,
    )
    rf_a.fit(Xa_tr, y_sub)
    resA_val = evaluate(y_val, rf_a.predict(Xa_val))
    print(f"A val macro_f1={resA_val['macro_f1']:.4f} acc={resA_val['accuracy']:.4f} "
          f"({time.time() - t0:.1f}s)", flush=True)
    save_confusion_matrix(OUT_DIR / "expA_val_confusion_matrix.png",
                          resA_val["confusion_matrix"], "A: RF engineered features - Validation")
    (OUT_DIR / "expA_val_classification_report.txt").write_text(resA_val["classification_report"], encoding="utf-8")
    joblib.dump(rf_a, OUT_DIR / "expA_rf_features_model.pkl")

    # ---------------- Experiment B: compact 1D CNN ---------------------------
    print("\n=== Experiment B: compact 1D CNN ===", flush=True)
    t0 = time.time()
    cnn, hist, best_val_loss, epochs_run, stopped = train_cnn(train_s, sub_idx, val_s, SEED)
    torch.save(cnn.state_dict(), OUT_DIR / "expB_cnn_best.pt")
    (OUT_DIR / "cnn_training_history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    plot_history(OUT_DIR / "cnn_training_history.png", hist)
    (OUT_DIR / "expB_cnn_config.json").write_text(json.dumps({
        "device": str(DEVICE), "n_parameters": count_params(cnn), "batch_size": BATCH,
        "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4,
        "max_epochs": MAX_EPOCHS, "epochs_run": epochs_run, "early_stopped": bool(stopped),
        "best_val_loss": best_val_loss, "seed": SEED,
        "training_subset_windows": int(len(sub_idx)), "classes": CLASS_NAMES,
    }, indent=2), encoding="utf-8")

    prob_b_val = np.empty((val_s.n, N_CLASSES), dtype=np.float32)
    pred_b_val = np.empty(val_s.n, dtype=np.int64)
    with torch.no_grad():
        for bi in iter_batches(val_idx, 512, False, np.random.default_rng(0)):
            logits = cnn(torch.from_numpy(np.asarray(val_s.x[bi], dtype=np.float32)).to(DEVICE))
            prob_b_val[bi] = torch.softmax(logits, 1).cpu().numpy()
            pred_b_val[bi] = logits.argmax(1).cpu().numpy()
    resB_val = evaluate(y_val, pred_b_val, prob_b_val)
    print(f"B val macro_f1={resB_val['macro_f1']:.4f} acc={resB_val['accuracy']:.4f} "
          f"({time.time() - t0:.1f}s)", flush=True)
    save_confusion_matrix(OUT_DIR / "expB_val_confusion_matrix.png",
                          resB_val["confusion_matrix"], "B: compact 1D CNN - Validation")
    (OUT_DIR / "expB_val_classification_report.txt").write_text(resB_val["classification_report"], encoding="utf-8")

    # ---------------- Experiment C: hybrid -----------------------------------
    print("\n=== Experiment C: CNN embedding + engineered features -> RF ===", flush=True)
    t0 = time.time()
    Za_tr = embeddings_batched(cnn, train_s.x, sub_idx, device=DEVICE)
    Za_val = embeddings_batched(cnn, val_s.x, device=DEVICE)
    Fc_tr = np.concatenate([Za_tr, Xa_tr], axis=1)
    Fc_val = np.concatenate([Za_val, Xa_val], axis=1)
    print(f"Fused features: train {Fc_tr.shape}, val {Fc_val.shape}", flush=True)
    rf_c = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight=RF_CLASS_WEIGHT, random_state=SEED, n_jobs=-1,
    )
    rf_c.fit(Fc_tr, y_sub)
    resC_val = evaluate(y_val, rf_c.predict(Fc_val))
    print(f"C val macro_f1={resC_val['macro_f1']:.4f} acc={resC_val['accuracy']:.4f} "
          f"({time.time() - t0:.1f}s)", flush=True)
    save_confusion_matrix(OUT_DIR / "expC_val_confusion_matrix.png",
                          resC_val["confusion_matrix"], "C: Hybrid CNN+features RF - Validation")
    (OUT_DIR / "expC_val_classification_report.txt").write_text(resC_val["classification_report"], encoding="utf-8")
    joblib.dump(rf_c, OUT_DIR / "expC_hybrid_rf_model.pkl")

    # ---------------- Validation-based model selection ------------------------
    candidates = {
        "A: RF on engineered features": {"model": "rf_a", "val": resA_val},
        "B: compact 1D CNN": {"model": "cnn", "val": resB_val},
        "C: CNN embedding + engineered features -> RF": {"model": "rf_c", "val": resC_val},
    }
    best_name = max(candidates, key=lambda k: candidates[k]["val"]["macro_f1"])
    print(f"\nSelected on validation macro F1: {best_name}", flush=True)

    for name, info in candidates.items():
        (OUT_DIR / f"exp{'ABC'[list(candidates).index(name)]}_val_metrics.json").write_text(
            json.dumps({**{"experiment": name}, **info["val"],
                        "training_subset_windows": int(len(sub_idx)), "seed": SEED}, indent=2),
            encoding="utf-8")

    # ---------------- Test evaluation ---------------------------------------
    def test_predict(name):
        if name.startswith("A"):
            X = features_batched(test_s.x, mean=NORM_MEAN, std=NORM_STD)
            return rf_a.predict(X), rf_a.predict_proba(X)
        if name.startswith("B"):
            yt = test_s.labels()
            prob = np.empty((test_s.n, N_CLASSES), dtype=np.float32)
            pred = np.empty(test_s.n, dtype=np.int64)
            with torch.no_grad():
                for bi in iter_batches(np.arange(test_s.n), 512, False, np.random.default_rng(0)):
                    logits = cnn(torch.from_numpy(np.asarray(test_s.x[bi], dtype=np.float32)).to(DEVICE))
                    prob[bi] = torch.softmax(logits, 1).cpu().numpy()
                    pred[bi] = logits.argmax(1).cpu().numpy()
            return pred, prob
        Xe = features_batched(test_s.x, mean=NORM_MEAN, std=NORM_STD)
        Ze = embeddings_batched(cnn, test_s.x, device=DEVICE)
        F = np.concatenate([Ze, Xe], axis=1)
        return rf_c.predict(F), rf_c.predict_proba(F)

    y_test = test_s.labels()
    pred, prob = test_predict(best_name)
    resTest = evaluate(y_test, pred, prob)
    resTest["experiment"] = best_name
    resTest["selection"] = "selected on validation macro F1"
    resTest["training_subset_windows"] = int(len(sub_idx))
    resTest["seed"] = SEED
    (OUT_DIR / "final_test_metrics.json").write_text(json.dumps(resTest, indent=2), encoding="utf-8")
    (OUT_DIR / "final_test_classification_report.txt").write_text(resTest["classification_report"], encoding="utf-8")
    save_confusion_matrix(OUT_DIR / "final_test_confusion_matrix.png",
                          resTest["confusion_matrix"], f"Selected model ({best_name}) - Test")
    print(f"TEST (once) {best_name}: macro_f1={resTest['macro_f1']:.4f} acc={resTest['accuracy']:.4f}", flush=True)

    if args.test_all:
        for name in candidates:
            if name == best_name:
                continue
            p, pr = test_predict(name)
            r = evaluate(y_test, p, pr)
            tag = f"exp{'ABC'[list(candidates).index(name)]}"
            (OUT_DIR / f"{tag}_test_metrics.json").write_text(
                json.dumps({**{"experiment": name}, **r}, indent=2), encoding="utf-8")
            (OUT_DIR / f"{tag}_test_classification_report.txt").write_text(r["classification_report"], encoding="utf-8")
            save_confusion_matrix(OUT_DIR / f"{tag}_test_confusion_matrix.png",
                                  r["confusion_matrix"], f"{name} - Test (deviation)")
            print(f"TEST {name}: macro_f1={r['macro_f1']:.4f} acc={r['accuracy']:.4f}", flush=True)

    # ---------------- Comparison table --------------------------------------
    rows = []
    for i, (name, info) in enumerate(candidates.items()):
        rows.append({
            "experiment": name,
            "val_accuracy": info["val"]["accuracy"],
            "val_macro_precision": info["val"]["macro_precision"],
            "val_macro_recall": info["val"]["macro_recall"],
            "val_macro_f1": info["val"]["macro_f1"],
            "selected": name == best_name,
        })
    comparison = {
        "protocol": summary["protocol"],
        "validation_comparison": rows,
        "selected_model": best_name,
        "test": {k: resTest[k] for k in
                 ("experiment", "accuracy", "macro_precision", "macro_recall", "macro_f1")},
    }
    (OUT_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    lines = [
        "| Experiment | Val accuracy | Val macro precision | Val macro recall | Val macro F1 | Selected |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['experiment']} | {r['val_accuracy']:.4f} | {r['val_macro_precision']:.4f} | "
            f"{r['val_macro_recall']:.4f} | {r['val_macro_f1']:.4f} | {'yes' if r['selected'] else 'no'} |"
        )
    lines += [
        "",
        f"Selected on validation: **{best_name}**",
        "",
        "Held-out test (single evaluation of the selected model):",
        "",
        "| Model | Accuracy | Macro precision | Macro recall | Macro F1 |",
        "|---|---|---|---|---|",
        f"| {resTest['experiment']} | {resTest['accuracy']:.4f} | {resTest['macro_precision']:.4f} | "
        f"{resTest['macro_recall']:.4f} | {resTest['macro_f1']:.4f} |",
        "",
        f"Training subset: {len(sub_idx)} windows (seed {SEED}) of {train_s.n} available.",
    ]
    (OUT_DIR / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    print(f"\nDone. Outputs in {OUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()

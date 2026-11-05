# evaluate.py
"""Consolidated evaluation for PROTECT-90 models.

Replays the saved CNN and Random Forest artifacts (no retraining, no test
tuning) and writes:
- final_results/evaluation_report.json
- final_results/cnn_confusion_matrix.png
- final_results/rf_confusion_matrix.png        (if a valid RF artifact exists)
- final_results/combined_confusion_matrix.png
- final_results/training_curves.png
- final_results/confidence_analysis.json        (CNN, validation-based threshold)
"""

from pathlib import Path
import sys, json, traceback

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
)

sys.path.insert(0, str(Path.cwd()))
import preprocess as P

PREPROC = Path("preprocessed")
BASE_DIR = Path("baseline")
BASE_FEAT_DIR = Path("baseline_features")
CNN_DIR = Path("cnn")
OUT_DIR = Path("final_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["LLL", "LL", "SLG", "LLG"]
N_CLASSES = 4
N_CHANNELS = 48
WINDOW_SIZE = 256


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


def load_test_split():
    return P.load_binny(PREPROC / "X_test")


def load_val_split():
    return P.load_binny(PREPROC / "X_val")


def preprocess_tensors(X):
    mean = np.load(PREPROC / "norm_mean.npy")
    std = np.load(PREPROC / "norm_std.npy")
    Xp = ((X - mean) / std).astype(np.float32, copy=False)
    Xt = torch.tensor(Xp, dtype=torch.float32)
    return Xt


def cnn_predictions(model, X, device, batch_size=128):
    model.eval()
    Xt = preprocess_tensors(X)
    ds = TensorDataset(Xt, torch.tensor(np.zeros(len(Xt), dtype=np.int64)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_list = []
    with torch.no_grad():
        for xb, _ in loader:
            logits_list.append(model(xb.to(device)).cpu().numpy())
    logits = np.concatenate(logits_list, axis=0)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    preds = logits.argmax(axis=1)
    conf = probs.max(axis=1)
    return preds, probs, conf


def cnn_metrics_from_artifacts():
    Xt, yt = load_test_split()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Small1DCNN().to(device)
    state = torch.load(CNN_DIR / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    preds, probs, conf = cnn_predictions(model, Xt, device)

    pr, rc, fs, _ = precision_recall_fscore_support(
        yt, preds, labels=np.arange(N_CLASSES), zero_division=0
    )
    cm = confusion_matrix(yt, preds, labels=np.arange(N_CLASSES))
    per_class = {
        CLASS_NAMES[i]: {
            "precision": float(pr[i]),
            "recall": float(rc[i]),
            "f1": float(fs[i]),
            "support": int((yt == i).sum()),
        }
        for i in range(N_CLASSES)
    }
    return {
        "accuracy": float(accuracy_score(yt, preds)),
        "macro_f1": float(f1_score(yt, preds, average="macro")),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "mean_confidence": float(conf.mean()),
        "predictions": preds.tolist(),
        "trues": yt.tolist(),
        "probabilities": probs.tolist(),
    }


def rf_metrics_from_artifacts(path):
    bundle = joblib.load(path)
    rf = bundle["rf"]
    scaler_mean = bundle["scaler_mean"]
    scaler_std = bundle["scaler_std"]

    Xt, yt = load_test_split()
    Xf = Xt.reshape(len(Xt), -1).astype(np.float32, copy=False)
    Xf_std = ((Xf - scaler_mean) / scaler_std).astype(np.float32, copy=False)
    y_pred = rf.predict(Xf_std)

    pr, rc, fs, _ = precision_recall_fscore_support(
        yt, y_pred, labels=np.arange(N_CLASSES), zero_division=0
    )
    cm = confusion_matrix(yt, y_pred, labels=np.arange(N_CLASSES))
    per_class = {
        CLASS_NAMES[i]: {
            "precision": float(pr[i]),
            "recall": float(rc[i]),
            "f1": float(fs[i]),
            "support": int((yt == i).sum()),
        }
        for i in range(N_CLASSES)
    }
    return {
        "accuracy": float(accuracy_score(yt, y_pred)),
        "macro_f1": float(f1_score(yt, y_pred, average="macro")),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "scaler": {"mean": scaler_mean.tolist(), "std": scaler_std.tolist()},
    }


def rf_features_metrics_from_artifacts(path):
    bundle = joblib.load(path)
    rf = bundle["rf"]
    scaler_mean = bundle["scaler_mean"]
    scaler_std = bundle["scaler_std"]

    Xt, yt = load_test_split()
    Xt2d = Xt.reshape(len(Xt), -1).astype(np.float32, copy=False)

    # Re-derive the same 5 features per channel used in train_baseline_features.py
    N = Xt2d.shape[0]
    feat = np.empty((N, 5 * N_CHANNELS), dtype=np.float32)
    Xt3 = Xt2d.reshape(N, WINDOW_SIZE, N_CHANNELS)
    for c in range(N_CHANNELS):
        ch = Xt3[:, :, c]
        feat[:, c * 5 + 0] = ch.mean(axis=1)
        feat[:, c * 5 + 1] = ch.std(axis=1)
        feat[:, c * 5 + 2] = np.sqrt((ch ** 2).mean(axis=1))
        feat[:, c * 5 + 3] = ch.max(axis=1) - ch.min(axis=1)
        feat[:, c * 5 + 4] = np.abs(ch).max(axis=1)

    feat_std = ((feat - scaler_mean) / scaler_std).astype(np.float32, copy=False)
    y_pred = rf.predict(feat_std)

    pr, rc, fs, _ = precision_recall_fscore_support(
        yt, y_pred, labels=np.arange(N_CLASSES), zero_division=0
    )
    cm = confusion_matrix(yt, y_pred, labels=np.arange(N_CLASSES))
    per_class = {
        CLASS_NAMES[i]: {
            "precision": float(pr[i]),
            "recall": float(rc[i]),
            "f1": float(fs[i]),
            "support": int((yt == i).sum()),
        }
        for i in range(N_CLASSES)
    }
    return {
        "accuracy": float(accuracy_score(yt, y_pred)),
        "macro_f1": float(f1_score(yt, y_pred, average="macro")),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "scaler": {"mean": scaler_mean.tolist(), "std": scaler_std.tolist()},
    }


def plot_confusion(ax, cm, title):
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(4),
        yticks=range(4),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True",
        xlabel="Predicted",
        title=title,
    )
    thresh = cm.max() / 2.0
    for i in range(4):
        for j in range(4):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )


def load_training_history():
    h = json.loads((CNN_DIR / "training_history.json").read_text(encoding="utf-8"))
    return h


def plot_training_curves(h):
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, h["train_loss"], label="Train loss", marker="o", markersize=3)
    ax1.plot(epochs, h["val_loss"], label="Val loss", marker="s", markersize=3)
    ax1.plot(epochs, h["test_loss"], label="Test loss", marker="^", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(epochs, h["val_acc"], label="Val acc", marker="d", markersize=3, color="tab:green")
    ax2.plot(epochs, h["val_macro_f1"], label="Val macro F1", marker="v", markersize=3, color="tab:purple")
    ax2.set_ylabel("Val accuracy / macro F1")
    ax2.legend(loc="upper right")
    plt.title("CNN Training / Validation Curves")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_curves.png", dpi=130)
    plt.close(fig)


def confidence_analysis(cnn_test):
    Xt, yt = load_test_split()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Small1DCNN().to(device)
    state = torch.load(CNN_DIR / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    Xv, yv = load_val_split()
    val_preds, val_probs, val_conf = cnn_predictions(model, Xv, device)

    # threshold selection on validation only
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95]
    rows = []
    for t in thresholds:
        mask = val_conf >= t
        if mask.sum() == 0:
            continue
        acc_t = float(accuracy_score(yv[mask], val_preds[mask]))
        mf1_t = float(f1_score(yv[mask], val_preds[mask], average="macro"))
        rows.append(
            {
                "threshold": t,
                "val_accepted": int(mask.sum()),
                "val_coverage": float(mask.mean()),
                "val_accuracy": acc_t,
                "val_macro_f1": mf1_t,
            }
        )

    # choose threshold on validation: maximize coverage subject to macro_f1 >= 0.90 * best(mf1)
    best_mf1 = max(r["val_macro_f1"] for r in rows) if rows else 0.0
    target = 0.90 * best_mf1
    chosen = None
    for r in sorted(rows, key=lambda x: -x["val_coverage"]):
        if r["val_macro_f1"] >= target:
            chosen = r
            break
    if chosen is None:
        chosen = rows[0]

    thr = chosen["threshold"]

    # apply chosen threshold to test (never used for selection)
    test_preds, test_probs, test_conf = cnn_predictions(model, Xt, device)
    mask = test_conf >= thr
    test_accepted = int(mask.sum())
    if test_accepted > 0:
        test_acc_accepted = float(accuracy_score(yt[mask], test_preds[mask]))
        test_mf1_accepted = float(f1_score(yt[mask], test_preds[mask], average="macro"))
    else:
        test_acc_accepted = None
        test_mf1_accepted = None

    return {
        "method": "max softmax probability; threshold selected on validation split only",
        "calibration_note": "softmax probabilities used as confidence score; not isotonic/Platt calibrated",
        "threshold_selected": thr,
        "threshold_selection_basis": (
            "highest validation coverage among thresholds with validation macro F1 >= 90%% of best validation macro F1"
        ),
        "validation_sweep": rows,
        "chosen_threshold_row": chosen,
        "test_coverage": float(mask.mean()) if mask.size else None,
        "test_accepted_count": test_accepted,
        "test_accuracy_accepted": test_acc_accepted,
        "test_macro_f1_accepted": test_mf1_accepted,
        "test_accuracy_full": cnn_test["accuracy"],
        "test_macro_f1_full": cnn_test["macro_f1"],
    }


def main():
    try:
        cnn_test = cnn_metrics_from_artifacts()

        # RF raw-waveform baseline (original baseline/)
        rf_raw = None
        rf_raw_path = BASE_DIR / "rf_model.pkl"
        if rf_raw_path.exists():
            try:
                rf_raw = rf_metrics_from_artifacts(rf_raw_path)
            except Exception as e:
                print("Could not replay raw-waveform RF baseline:", e)

        # RF feature baseline (baseline_features/)
        rf_feat = None
        rf_feat_path = BASE_FEAT_DIR / "rf_model.pkl"
        if rf_feat_path.exists():
            try:
                rf_feat = rf_features_metrics_from_artifacts(rf_feat_path)
            except Exception as e:
                print("Could not replay feature RF baseline:", e)

        report = {"cnn_test": cnn_test}
        if rf_raw is not None:
            report["random_forest_raw_test"] = rf_raw
        if rf_feat is not None:
            report["random_forest_features_test"] = rf_feat

        (OUT_DIR / "evaluation_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        plot_confusion(axes[0], np.array(cnn_test["confusion_matrix"]), "1D CNN - Test")
        if rf_raw is not None:
            plot_confusion(axes[1], np.array(rf_raw["confusion_matrix"]), "Random Forest (raw) - Test")
        elif rf_feat is not None:
            plot_confusion(axes[1], np.array(rf_feat["confusion_matrix"]), "Random Forest (features) - Test")
        else:
            axes[1].axis("off")
            axes[1].set_title("No valid RF artifact")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "combined_confusion_matrix.png", dpi=130)
        plt.close(fig)

        # individual CNN confusion matrix
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_confusion(ax, np.array(cnn_test["confusion_matrix"]), "1D CNN - Test")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "cnn_confusion_matrix.png", dpi=130)
        plt.close(fig)

        if rf_raw is not None:
            fig, ax = plt.subplots(figsize=(7, 6))
            plot_confusion(ax, np.array(rf_raw["confusion_matrix"]), "Random Forest (raw) - Test")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "rf_confusion_matrix.png", dpi=130)
            plt.close(fig)
        if rf_feat is not None:
            fig, ax = plt.subplots(figsize=(7, 6))
            plot_confusion(ax, np.array(rf_feat["confusion_matrix"]), "Random Forest (features) - Test")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "rf_features_confusion_matrix.png", dpi=130)
            plt.close(fig)

        plot_training_curves(load_training_history())

        conf = confidence_analysis(cnn_test)
        (OUT_DIR / "confidence_analysis.json").write_text(
            json.dumps(conf, indent=2), encoding="utf-8"
        )

        print("Saved final_results/evaluation_report.json")
        print("Saved final_results/cnn_confusion_matrix.png")
        print("Saved final_results/combined_confusion_matrix.png")
        print("Saved final_results/training_curves.png")
        print("Saved final_results/confidence_analysis.json")
        if rf_raw is not None:
            print("Saved final_results/rf_confusion_matrix.png")
        if rf_feat is not None:
            print("Saved final_results/rf_features_confusion_matrix.png")
        print("CNN test:", json.dumps(cnn_test, indent=2))
        if rf_raw is not None:
            print("RF raw test:", json.dumps(rf_raw, indent=2))
        if rf_feat is not None:
            print("RF features test:", json.dumps(rf_feat, indent=2))
        print("Confidence:", json.dumps(conf, indent=2))
    except Exception:
        print("ERROR in evaluate.py:")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()

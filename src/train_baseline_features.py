# train_baseline_features.py
"""Feature-based Random Forest baseline for PROTECT-90.

Reads the already-normalized binary splits from preprocess.py and derives a
small set of physically meaningful per-channel features (mean, std, RMS,
peak-to-peak, max abs). Normalization of features is fit on train only.

This is a separate, corrected baseline in `baseline_features/` and does not
overwrite the original `baseline/` artifacts.
"""

from pathlib import Path
import sys, json, traceback, warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))
import preprocess as P

PREPROC = Path("preprocessed")
OUT_DIR = Path("baseline_features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["LLL", "LL", "SLG", "LLG"]
B = 4096


def chunked_standardize_fit(X, chunk_size=B):
    n, d = X.shape
    mean = np.zeros(d, dtype=np.float64)
    m2 = np.zeros(d, dtype=np.float64)
    seen = 0
    for start in range(0, n, chunk_size):
        x = X[start:start + chunk_size].astype(np.float64, copy=False)
        c = x.shape[0]
        xmu = x.mean(axis=0)
        delta = xmu - mean
        mean += delta * c / (seen + c)
        m2 += ((x - mean) ** 2).sum(axis=0)
        seen += c
    var = m2 / max(seen - 1, 1)
    std = np.sqrt(var)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def chunked_standardize(X, mean, std):
    out = np.empty_like(X, dtype=np.float32)
    for start in range(0, len(X), B):
        x = X[start:start + B]
        out[start:start + x.shape[0]] = ((x - mean) / std)
    return out


def features_from_windows(X):
    """X: (N, 256, 48). Returns (N, 5 * 48) feature matrix."""
    N = X.shape[0]
    out = np.empty((N, 5 * X.shape[2]), dtype=np.float32)
    for c in range(X.shape[2]):
        ch = X[:, :, c]
        out[:, c * 5 + 0] = ch.mean(axis=1)
        out[:, c * 5 + 1] = ch.std(axis=1)
        out[:, c * 5 + 2] = np.sqrt((ch ** 2).mean(axis=1))
        out[:, c * 5 + 3] = ch.max(axis=1) - ch.min(axis=1)
        out[:, c * 5 + 4] = np.abs(ch).max(axis=1)
    return out


def eval_split(y, y_pred):
    from sklearn.metrics import (
        f1_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )

    acc = float((y_pred == y).mean())
    pr, rc, fs, _ = precision_recall_fscore_support(
        y, y_pred, labels=np.arange(4), zero_division=0
    )
    cm = confusion_matrix(y, y_pred, labels=np.arange(4))
    per_class = {
        CLASS_NAMES[i]: {
            "precision": float(pr[i]),
            "recall": float(rc[i]),
            "f1": float(fs[i]),
            "support": int((y == i).sum()),
        }
        for i in range(4)
    }
    return {
        "accuracy": acc,
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def run():
    try:
        print("Loading splits...")
        Xtr, ytr = P.load_binny(PREPROC / "X_train")
        Xv, yv = P.load_binny(PREPROC / "X_val")
        Xt, yt = P.load_binny(PREPROC / "X_test")

        print("Deriving features...")
        Ftr = features_from_windows(Xtr)
        Fv = features_from_windows(Xv)
        Ft = features_from_windows(Xt)
        del Xtr, Xv, Xt

        print("Chunked standardize fit on train only...")
        mean, std = chunked_standardize_fit(Ftr)
        Ftr_std = chunked_standardize(Ftr, mean, std)
        Fv_std = chunked_standardize(Fv, mean, std)
        Ft_std = chunked_standardize(Ft, mean, std)
        del Ftr, Fv, Ft

        print("Training Random Forest...")
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=42,
            class_weight="balanced",
        )
        rf.fit(Ftr_std, ytr)

        import joblib

        joblib.dump(
            {"rf": rf, "scaler_mean": mean, "scaler_std": std},
            OUT_DIR / "rf_model.pkl",
        )

        metrics_test = eval_split(yt, rf.predict(Ft_std))
        metrics_val = eval_split(yv, rf.predict(Fv_std))
        metrics_train = eval_split(ytr, rf.predict(Ftr_std))

        metrics = {"train": metrics_train, "validation": metrics_val, "test": metrics_test}
        (OUT_DIR / "baseline_features_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        cm_test = confusion_matrix(yt, rf.predict(Ft_std), labels=np.arange(4))
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_test, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=range(4),
            yticks=range(4),
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ylabel="True",
            xlabel="Predicted",
            title="Random Forest (features) - Test Confusion Matrix",
        )
        thresh = cm_test.max() / 2.0
        for i in range(4):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    format(cm_test[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm_test[i, j] > thresh else "black",
                )
        plt.tight_layout()
        fig.savefig(OUT_DIR / "baseline_features_confusion_matrix.png", dpi=130)
        plt.close(fig)

        # configuration file for reproducibility
        cfg = {
            "feature_set": ["mean", "std", "rms", "peak_to_peak", "max_abs"],
            "n_features_per_channel": 5,
            "n_channels": 48,
            "n_features": 5 * 48,
            "normalization_fit_on_train_only": True,
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": 1,
            "input_shape": [None, 256, 48],
            "classes": CLASS_NAMES,
        }
        (OUT_DIR / "baseline_features_config.json").write_text(
            json.dumps(cfg, indent=2), encoding="utf-8"
        )

        print("Saved baseline_features_metrics.json")
        print("Saved baseline_features_confusion_matrix.png")
        print("Saved rf_model.pkl, config")
        print("Test metrics:", json.dumps(metrics_test, indent=2))
    except Exception:
        print("ERROR in train_baseline_features.py:")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    run()

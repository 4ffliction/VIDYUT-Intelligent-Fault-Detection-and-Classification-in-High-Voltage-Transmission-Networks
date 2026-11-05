# evaluate_episode_level.py
"""Episode-level evaluation for the selected VIDYUT-KAVACHA model (experiment B).

The headline metrics in hybrid_results/ are per-window. Windows from one
simulation episode are strongly correlated, so per-window scores overstate
performance on a *new, unseen episode*. This script adds an episode-level view.

IMPORTANT — provenance of the window -> episode mapping
------------------------------------------------------
preprocessed/*.x.bin store only the label per window; no episode ID was
persisted. The mapping is therefore RECONSTRUCTED, not read:

preprocess.build_split() iterates the split episode list in CSV order and appends
each episode's windows contiguously, so window blocks are ordered and
contiguous. Window counts follow deterministically from
episode_to_windows(t_evnt_start, t_evnt_end, dt).

The reconstruction is not trusted: every reconstructed episode is re-windowed
from its raw PKL, z-scored with the saved train statistics, and compared
byte-for-byte against the stored windows at the hypothesised offset. If any
window disagrees the script aborts instead of reporting numbers.

Nothing is retrained. The checkpoint and all existing results are read-only.

Run: python evaluate_episode_level.py
"""

from pathlib import Path
import json
import struct
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import DATA_DIR, episode_to_windows  # noqa: E402
from train_hybrid import Small1DCNN  # noqa: E402

PREPROC = Path("preprocessed")
RESULTS = Path("hybrid_results")
OUT = Path("episode_evaluation")
OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["LLL", "LL", "SLG", "LLG"]
N_CLASSES = 4
BATCH = 512
ATOL = 1e-5          # tolerance for the reconstruction check (observed diff: 0.0)


# ----------------------------------------------------------------------------
def load_split(stem):
    with open(str(stem) + ".x.bin", "rb") as f:
        n, ws, nc = struct.unpack("<III", f.read(12))
    x = np.memmap(str(stem) + ".x.bin", dtype=np.float32, mode="r", offset=12).reshape(n, ws, nc)
    with open(str(stem) + ".y.bin", "rb") as f:
        ny = struct.unpack("<I", f.read(4))[0]
    y = np.memmap(str(stem) + ".y.bin", dtype=np.int64, mode="r", offset=4)
    if n != ny:
        raise ValueError(f"{stem}: n={n} != n_y={ny}")
    return x, y, n


def reconstruct_offsets(ids, labels, x, mean, std, split_name):
    """Re-window each episode and verify it matches the stored block byte-for-byte."""
    rows = []
    offset = 0
    worst = 0.0
    t0 = time.time()
    cols_cache = {}

    for i, sid in enumerate(ids):
        meta = labels.loc[int(sid)]
        df = pd.read_pickle(DATA_DIR / f"{int(sid)}_sample_hv_double_line_90kv.pkl")
        if not cols_cache:
            cols_cache["cols"] = sorted(c for c in df.columns if c != "time_s")
        cols = cols_cache["cols"]
        dt = float(df["time_s"].iloc[1] - df["time_s"].iloc[0])
        raw, cov = episode_to_windows(
            df[cols],
            float(meta["t_evnt_start"]),
            float(meta["t_evnt_end"]),
            dt=dt,
        )
        k = int(raw.shape[0])
        if k == 0:
            rows.append({"sample_id": int(sid), "sc_type": int(meta["sc_type"]),
                         "n_windows": 0, "offset": offset, "mean_coverage": 0.0})
            continue

        z = ((raw.reshape(-1, raw.shape[2]) - mean) / std).reshape(raw.shape).astype(np.float32)
        stored = np.asarray(x[offset:offset + k], dtype=np.float32)
        if stored.shape != z.shape:
            raise SystemExit(
                f"ABORT: {split_name} episode {sid} expected {k} windows at offset {offset} "
                f"but only {stored.shape[0]} remain. Stored artifacts do not match the sampler."
            )
        diff = float(np.abs(stored - z).max())
        worst = max(worst, diff)
        if diff > ATOL:
            raise SystemExit(
                f"ABORT: {split_name} episode {sid} at offset {offset} does not match "
                f"stored windows (max|diff|={diff:.6g} > {ATOL}).\n"
                "The window->episode mapping cannot be reconstructed reliably; "
                "refusing to report episode-level metrics."
            )
        rows.append({"sample_id": int(sid), "sc_type": int(meta["sc_type"]),
                     "n_windows": k, "offset": offset,
                     "mean_coverage": float(cov.mean())})
        offset += k

        if (i + 1) % 250 == 0:
            print(f"  verified {i + 1}/{len(ids)} episodes "
                  f"({offset} windows, {time.time() - t0:.0f}s)", flush=True)

    return pd.DataFrame(rows), offset, worst


def metrics(y_true, y_pred):
    pr, rc, fs, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(N_CLASSES), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(N_CLASSES))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(pr.mean()),
        "macro_recall": float(rc.mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "n": int(len(y_true)),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(pr[i]), "recall": float(rc[i]),
                "f1": float(fs[i]), "support": int(sup[i]),
            } for i in range(N_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=np.arange(N_CLASSES),
            target_names=CLASS_NAMES, zero_division=0,
        ),
    }


def main():
    print("Loading test split (read-only) ...", flush=True)
    x, y, n = load_split(PREPROC / "X_test")
    mean = np.load(PREPROC / "norm_mean.npy").astype(np.float32)
    std = np.load(PREPROC / "norm_std.npy").astype(np.float32)
    ids = pd.read_csv(PREPROC / "test_episodes.csv")["sample_id"].to_numpy()
    labels = pd.read_csv("hv_double_line_90kv_labels.csv").set_index("sample_id")
    print(f"test windows={n}, test episodes in manifest={len(ids)}", flush=True)

    print("\nReconstructing and verifying window -> episode mapping ...", flush=True)
    ep, offset, worst = reconstruct_offsets(ids, labels, x, mean, std, "test")
    if offset != n:
        raise SystemExit(f"ABORT: episode windows sum to {offset} but split holds {n}.")
    zero_win = int((ep["n_windows"] == 0).sum())
    print(f"verified {len(ep)} episodes, {offset} windows, max|diff|={worst:.3g}, "
          f"episodes with 0 windows={zero_win}", flush=True)

    # ---- predictions from the saved best checkpoint (no training) -----------
    ckpt_path = RESULTS / "expB_cnn_best.pt"
    model = Small1DCNN()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    print(f"\nPredicting {n} test windows with {ckpt_path.name} ...", flush=True)
    prob = np.empty((n, N_CLASSES), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, BATCH):
            stop = min(start + BATCH, n)
            xb = torch.from_numpy(np.asarray(x[start:stop], dtype=np.float32))
            prob[start:stop] = torch.softmax(model(xb), dim=1).numpy()
    win_pred = prob.argmax(axis=1)
    y_true_win = np.asarray(y)

    window_metrics = metrics(y_true_win, win_pred)

    # integrity: recomputed window-level metrics must match the stored results
    stored = json.loads((RESULTS / "final_test_metrics.json").read_text(encoding="utf-8"))
    checks = {
        "accuracy": abs(window_metrics["accuracy"] - stored["accuracy"]) < 1e-9,
        "macro_f1": abs(window_metrics["macro_f1"] - stored["macro_f1"]) < 1e-9,
        "confusion_matrix": window_metrics["confusion_matrix"] == stored["confusion_matrix"],
    }
    if not all(checks.values()):
        raise SystemExit(f"ABORT: recomputed window metrics disagree with stored results: {checks}")
    print(f"window-level metrics reproduce hybrid_results/final_test_metrics.json exactly", flush=True)

    # ---- aggregate per episode ---------------------------------------------
    records = []
    for r in ep.itertuples():
        if r.n_windows == 0:
            records.append({"sample_id": r.sample_id, "sc_type": r.sc_type,
                            "n_windows": 0, "majority": -1, "meanprob": -1,
                            "votes": "", "max_vote_frac": 0.0})
            continue
        sl = slice(r.offset, r.offset + r.n_windows)
        votes = np.bincount(win_pred[sl], minlength=N_CLASSES)
        mean_p = prob[sl].mean(axis=0)

        # majority vote; ties broken by summed probability, then lowest index
        top = votes.max()
        tied = np.flatnonzero(votes == top)
        maj = int(tied[np.argmax(mean_p[tied])]) if len(tied) > 1 else int(tied[0])

        records.append({
            "sample_id": int(r.sample_id), "sc_type": int(r.sc_type),
            "n_windows": int(r.n_windows), "majority": maj,
            "meanprob": int(mean_p.argmax()),
            "votes": "|".join(str(int(v)) for v in votes),
            "max_vote_frac": float(top / r.n_windows),
            "n_tied": int(len(tied)),
        })
    ep_df = pd.DataFrame(records)
    evaluated = ep_df[ep_df["n_windows"] > 0].copy()

    maj_metrics = metrics(evaluated["sc_type"].to_numpy(), evaluated["majority"].to_numpy())
    mp_metrics = metrics(evaluated["sc_type"].to_numpy(), evaluated["meanprob"].to_numpy())

    # ---- save ---------------------------------------------------------------
    summary = {
        "model": "B: compact 1D CNN (saved best validation-loss checkpoint)",
        "checkpoint": str(ckpt_path),
        "protocol": {
            "window_to_episode_mapping": "reconstructed deterministically from the "
                                         "episode split order and episode_to_windows(); "
                                         "verified window-by-window against stored arrays",
            "reconstruction_max_abs_diff": worst,
            "reconstruction_atol": ATOL,
            "aggregation_primary": "majority vote over window predictions",
            "aggregation_secondary": "argmax of mean class probability per episode",
            "tie_break": "majority ties broken by higher mean probability, then lowest class index",
            "ties_encountered": int((evaluated["n_tied"] > 1).sum()) if "n_tied" in evaluated else 0,
            "retrained": False,
            "test_labels_used_for": "scoring only",
        },
        "counts": {
            "test_episodes_in_manifest": int(len(ep_df)),
            "test_episodes_evaluated": int(len(evaluated)),
            "test_episodes_with_zero_windows": zero_win,
            "test_windows": int(n),
            "windows_per_episode_min": int(evaluated["n_windows"].min()),
            "windows_per_episode_median": float(evaluated["n_windows"].median()),
            "windows_per_episode_max": int(evaluated["n_windows"].max()),
            "episode_class_distribution": {
                CLASS_NAMES[c]: int((evaluated["sc_type"] == c).sum()) for c in range(N_CLASSES)
            },
        },
        "window_level": window_metrics,
        "episode_level_majority_vote": maj_metrics,
        "episode_level_mean_probability": mp_metrics,
    }
    (OUT / "episode_level_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for key in ("episode_level_majority_vote", "episode_level_mean_probability", "window_level"):
        m = summary[key]
        for cname in CLASS_NAMES:
            pc = m["per_class"][cname]
            rows.append({
                "level": key, "class": cname,
                "precision": round(pc["precision"], 6),
                "recall": round(pc["recall"], 6),
                "f1": round(pc["f1"], 6),
                "support": pc["support"],
                "accuracy": round(m["accuracy"], 6),
                "macro_precision": round(m["macro_precision"], 6),
                "macro_recall": round(m["macro_recall"], 6),
                "macro_f1": round(m["macro_f1"], 6),
            })
    pd.DataFrame(rows).to_csv(OUT / "episode_level_classification_report.csv", index=False)
    ep_df.to_csv(OUT / "episode_predictions.csv", index=False)

    # confusion matrix: episode-level (majority) with window-level alongside
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title in (
        (axes[0], maj_metrics, "Episode-level (majority vote) - Test"),
        (axes[1], window_metrics, "Window-level - Test"),
    ):
        cm = np.asarray(key["confusion_matrix"], dtype=np.int64)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
        thresh = cm.max() / 2.0
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(OUT / "episode_level_confusion_matrix.png", dpi=140)
    plt.close(fig)

    print("\n=== RESULTS ===", flush=True)
    for key in ("window_level", "episode_level_majority_vote", "episode_level_mean_probability"):
        m = summary[key]
        print(f"{key:34s} n={m['n']:6d} acc={m['accuracy']:.4f} "
              f"macroF1={m['macro_f1']:.4f}", flush=True)
    print(f"\nepisodes evaluated={len(evaluated)}, median windows/episode="
          f"{evaluated['n_windows'].median():.0f}, ties={summary['protocol']['ties_encountered']}",
          flush=True)
    print(f"outputs -> {OUT.resolve()}", flush=True)


if __name__ == "__main__":
    main()

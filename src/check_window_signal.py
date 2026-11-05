# check_window_signal.py
"""Diagnostic: do the preprocessed windows actually contain the fault?

The stored preprocessed bins were built with a window sampler that emits two
fault-aligned windows and then fills the remaining 22 slots from sample 0
onwards with stride 64.  Because the fault starts near t=0.43 s of a 1.0 s
record, almost every window is pre-fault steady state.

This script quantifies the defect and tests, on a small episode sample, whether
windows drawn from inside the fault interval are classifiable at all.

Run: python check_window_signal.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

DATA_DIR = Path("hv_double_line_90kv_preprocessed_data_extracted") / "hv_double_line_90kv_preprocessed_data"
LABEL_CSV = "hv_double_line_90kv_labels.csv"
PREPROC = Path("preprocessed")

WINDOW_SIZE = 256
STRIDE = 64
MAX_WINDOWS_PER_EPISODE = 24
N_FEAT = 4


def load_episode(sample_id):
    return pd.read_pickle(DATA_DIR / f"{sample_id}_sample_hv_double_line_90kv.pkl")


def sample_starts(n, f0, f1):
    """Candidate window starts: onset-centred, then fully inside the fault span."""
    starts = []
    for s in (max(0, f0 - WINDOW_SIZE // 2), max(0, f0), max(0, f1 - WINDOW_SIZE)):
        s = min(s, max(0, n - WINDOW_SIZE))
        if s not in starts:
            starts.append(s)
    last = max(0, min(n, f1) - WINDOW_SIZE)
    for s in range(max(0, f0), last + 1, STRIDE):
        if len(starts) >= MAX_WINDOWS_PER_EPISODE:
            break
        if s not in starts:
            starts.append(s)
    return starts[:MAX_WINDOWS_PER_EPISODE]


def features(X):
    N = X.shape[0]
    out = np.empty((N, X.shape[2] * N_FEAT), dtype=np.float32)
    for c in range(X.shape[2]):
        ch = X[:, :, c]
        out[:, c * N_FEAT + 0] = np.sqrt((ch ** 2).mean(axis=1))
        out[:, c * N_FEAT + 1] = ch.std(axis=1)
        out[:, c * N_FEAT + 2] = ch.max(axis=1) - ch.min(axis=1)
        out[:, c * N_FEAT + 3] = np.abs(ch).max(axis=1)
    return out


def build_episode_windows(sid, meta, fault_only):
    df = load_episode(sid)
    dt = float(df["time_s"].iloc[1] - df["time_s"].iloc[0])
    cols = sorted([c for c in df.columns if c != "time_s"])
    X = df[cols].to_numpy(dtype=np.float32)
    n = len(df)
    f0 = int(round(float(meta["t_evnt_start"]) / dt))
    f1 = int(round(float(meta["t_evnt_end"]) / dt))
    if fault_only:
        starts = sample_starts(n, f0, f1)
    else:
        starts = [max(0, f0 - WINDOW_SIZE // 2)]
        for s in range(0, n - WINDOW_SIZE + 1, STRIDE):
            if len(starts) >= MAX_WINDOWS_PER_EPISODE:
                break
            if s not in starts:
                starts.append(s)
    w = np.stack([X[s:s + WINDOW_SIZE] for s in starts], axis=0)
    return w, int(meta["sc_type"]), starts, f0, f1


def run(mode, n_per_class=50, seed=42, standardize=False):
    labels = pd.read_csv(LABEL_CSV).set_index("sample_id")
    split = pd.read_csv(PREPROC / "train_episodes.csv")
    rng = np.random.default_rng(seed)
    ids = []
    for c in range(4):
        pool = split.loc[split["sc_type"] == c, "sample_id"].to_numpy()
        ids.append(rng.choice(pool, size=min(n_per_class, pool.size), replace=False))
    ids = np.concatenate(ids)

    raw, ys, groups = [], [], []
    cover = []
    for sid in ids:
        meta = labels.loc[int(sid)]
        w, y, starts, f0, f1 = build_episode_windows(int(sid), meta, fault_only=(mode == "fault"))
        raw.append(w)
        ys.append(np.full(w.shape[0], y, dtype=np.int64))
        groups.append(np.full(w.shape[0], int(sid), dtype=np.int64))
        if mode == "fault":
            cover.extend([max(0, min(s + WINDOW_SIZE, f1) - max(s, f0)) / WINDOW_SIZE for s in starts])
        else:
            cover.extend([max(0, min(s + WINDOW_SIZE, f1) - max(s, f0)) / WINDOW_SIZE for s in starts])

    Xw = np.concatenate(raw, axis=0)
    if standardize:
        mu = Xw.reshape(-1, Xw.shape[2]).mean(axis=0)
        sd = Xw.reshape(-1, Xw.shape[2]).std(axis=0)
        sd[sd == 0] = 1.0
        Xw = ((Xw.reshape(-1, Xw.shape[2]) - mu) / sd).reshape(Xw.shape).astype(np.float32)
    X = features(Xw)
    y = np.concatenate(ys)
    g = np.concatenate(groups)
    cover = np.array(cover)

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros_like(y)
    for tr, te in gkf.split(X, y, groups=g):
        rf = RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=8,
        )
        rf.fit(X[tr], y[tr])
        oof[te] = rf.predict(X[te])

    summary = {
        "mode": mode,
        "features_on": "z-scored" if standardize else "physical units",
        "episodes": int(len(ids)),
        "windows": int(X.shape[0]),
        "windows_per_episode": round(float(X.shape[0] / len(ids)), 2),
        "mean_fault_coverage_per_window": round(float(cover.mean()), 4),
        "frac_windows_overlapping_fault": round(float((cover > 0).mean()), 4),
        "cv_macro_f1": round(float(f1_score(y, oof, average="macro")), 4),
        "cv_accuracy": round(float((oof == y).mean()), 4),
    }
    return summary


if __name__ == "__main__":
    out = [run("legacy"), run("fault"), run("fault", standardize=True)]
    for s in out:
        print(json.dumps(s))
    Path("window_signal_diagnostic.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

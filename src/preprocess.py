# preprocess.py
# PROTECT-90 (VIDYUT-KAVACHA) preprocessing pipeline.

import json
import struct
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EXTRACT_DIR = "hv_double_line_90kv_preprocessed_data_extracted"
LABEL_CSV = "hv_double_line_90kv_labels.csv"
DATA_DIR = Path(EXTRACT_DIR) / "hv_double_line_90kv_preprocessed_data"

WINDOW_SIZE = 256
STRIDE = 64
MAX_WINDOWS_PER_EPISODE = 24
WINDOW_BUDGET = {"train": 500000, "validation": 120000, "test": 120000}
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42

OUT_DIR = Path("preprocessed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = Path("preprocessing_config.json")


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def load_labels():
    df = pd.read_csv(LABEL_CSV)
    for col in ["sample_id", "sc_type", "fault_target", "phase_select", "t_evnt_start", "t_evnt_end"]:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")
    df["sample_id"] = pd.to_numeric(df["sample_id"], errors="coerce").astype("Int64")
    df["sc_type"] = pd.to_numeric(df["sc_type"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["sample_id", "sc_type"])
    return df


def available_sample_ids():
    s = set()
    for p in DATA_DIR.glob("*_sample_hv_double_line_90kv.pkl"):
        try:
            s.add(int(p.stem.split("_")[0]))
        except Exception:
            continue
    return s


def load_waveform(sample_id):
    pkl_path = DATA_DIR / f"{sample_id}_sample_hv_double_line_90kv.pkl"
    if not pkl_path.is_file():
        raise FileNotFoundError(str(pkl_path))
    df = pd.read_pickle(pkl_path)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(type(df))
    return df


def episode_to_windows(df_ep, t_start, t_end, dt=None):
    """Window an episode inside its fault interval.

    CORRECTED WINDOW PLACEMENT (see README_LIMITATIONS / window_signal_diagnostic.json).
    The original sampler emitted two fault-aligned windows and then filled the
    remaining slots with a stride sweep starting at sample 0.  Because the fault
    starts near t=0.43 s of a 1.0 s record, only ~2 of 24 windows overlapped the
    fault (2.1% fault coverage), so ~98% of training windows were pre-fault
    steady state and every downstream model scored at chance level.

    Windows are now drawn from [t_evnt_start, t_evnt_end]: onset-centred windows
    first (so the fault transient is always captured), then a stride sweep fully
    inside the fault span, capped at MAX_WINDOWS_PER_EPISODE.

    Window size, stride, cap and per-episode label inheritance are unchanged.
    Returns (windows, coverage) where coverage[i] is the fraction of window i
    that lies inside the fault interval.
    """
    n = len(df_ep)
    if n < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE, df_ep.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # NOTE: callers normally pass dt explicitly because the feature frame has no
    # time column.  Without dt the fault alignment silently degrades to a
    # whole-record stride sweep, which was the original defect.
    if dt is None:
        time_col = next((c for c in ("time_s", "time", "t") if c in df_ep.columns), None)
        if time_col is not None and df_ep[time_col].size > 1:
            tv = pd.to_numeric(df_ep[time_col], errors="coerce").to_numpy(dtype=np.float64)
            if tv.size > 1 and np.isfinite(tv[1] - tv[0]):
                dt = float(tv[1] - tv[0])

    if t_start is not None and not (dt and dt > 0):
        raise ValueError("fault alignment requested (t_evnt_start present) but no valid dt; refusing silent fallback")

    if dt and dt > 0 and t_start is not None:
        f0 = max(0, min(int(round(float(t_start) / dt)), n - 1))
        f1 = int(round(float(t_end) / dt)) if (t_end is not None and pd.notna(t_end)) else n
        f1 = max(f0 + 1, min(int(f1), n))
        starts = []
        for s in (f0 - WINDOW_SIZE // 2, f0, f1 - WINDOW_SIZE):
            s = max(0, min(s, n - WINDOW_SIZE))
            if s not in starts:
                starts.append(s)
        last_inside = max(f0, min(n, f1) - WINDOW_SIZE)
        for s in range(f0, last_inside + 1, STRIDE):
            if len(starts) >= MAX_WINDOWS_PER_EPISODE:
                break
            if s not in starts:
                starts.append(s)
        starts = starts[:MAX_WINDOWS_PER_EPISODE]
    else:
        # No event timestamp: fall back to a plain stride sweep over the record.
        f0, f1 = 0, n
        starts = list(range(0, n - WINDOW_SIZE + 1, STRIDE))[:MAX_WINDOWS_PER_EPISODE]

    if not starts:
        starts = [0]
    arr = np.stack([df_ep.values[s:s + WINDOW_SIZE].astype(np.float32) for s in starts], axis=0)
    coverage = np.array(
        [max(0, min(s + WINDOW_SIZE, f1) - max(s, f0)) / float(WINDOW_SIZE) for s in starts],
        dtype=np.float32,
    )
    return arr, coverage


def build_windows_for_episode(sample_id, meta):
    df = load_waveform(sample_id)
    feat_cols = sorted([c for c in df.columns if c != "time_s"])
    df_feat = df[feat_cols]
    dt = None
    if "time_s" in df.columns and len(df) > 1:
        tv = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=np.float64)
        if tv.size > 1 and np.isfinite(tv[1] - tv[0]) and tv[1] > tv[0]:
            dt = float(tv[1] - tv[0])
    t_start = float(meta["t_evnt_start"]) if pd.notna(meta.get("t_evnt_start")) else None
    t_end = float(meta["t_evnt_end"]) if pd.notna(meta.get("t_evnt_end")) else None
    w, coverage = episode_to_windows(df_feat, t_start, t_end, dt=dt)
    return w.astype(np.float32), int(meta["sc_type"]), coverage


def save_binny(path, X, y):
    path_x = Path(str(path) + ".x.bin")
    path_y = Path(str(path) + ".y.bin")
    mode_x = "ab" if path_x.exists() else "wb"
    mode_y = "ab" if path_y.exists() else "wb"
    with open(path_x, mode_x) as fx, open(path_y, mode_y) as fy:
        if mode_x == "wb":
            fx.write(struct.pack("<III", 0, WINDOW_SIZE, int(X.shape[2])))
        fx.write(X.astype(np.float32).tobytes())
        if mode_y == "wb":
            fy.write(struct.pack("<I", 0))
        fy.write(y.astype(np.int64).tobytes())


def finalize_binny(path):
    path_x = Path(str(path) + ".x.bin")
    path_y = Path(str(path) + ".y.bin")
    if path_x.exists():
        with open(path_x, "rb") as f:
            f.read(12)
            body = f.read()
        n = len(body) // (WINDOW_SIZE * 48 * 4)
        tmp = path_x.with_suffix(".tmp")
        tmp.write_bytes(struct.pack("<III", n, WINDOW_SIZE, 48) + body)
        tmp.replace(path_x)
    if path_y.exists():
        with open(path_y, "rb") as f:
            f.read(4)
            body = f.read()
        n = len(body) // 8
        tmp = path_y.with_suffix(".tmp")
        tmp.write_bytes(struct.pack("<I", n) + body)
        tmp.replace(path_y)


def load_binny(path):
    path_x = Path(str(path) + ".x.bin")
    path_y = Path(str(path) + ".y.bin")
    with open(path_x, "rb") as f:
        n, ws, nc = struct.unpack("<III", f.read(12))
        X = np.frombuffer(f.read(), dtype=np.float32).reshape(n, ws, nc).copy()
    with open(path_y, "rb") as f:
        ny = struct.unpack("<I", f.read(4))[0]
        y = np.frombuffer(f.read(), dtype=np.int64).copy()
    if n != ny or nc != 48 or ws != WINDOW_SIZE:
        raise ValueError("header mismatch")
    return X, y


def build_split(ids, split_name, budget, out_x, norm_mean=None, norm_std=None, fit_norm=False):
    Xs, ys = [], []
    counts = {}
    total = 0
    failed = []
    coverage_parts = []
    for sid in ids:
        try:
            meta = labels_df.loc[labels_df["sample_id"] == sid].iloc[0]
            w, lbl, cov = build_windows_for_episode(int(sid), meta)
            if w.size == 0:
                continue
            if budget is not None and total + w.shape[0] > budget:
                break
            Xs.append(w)
            ys.append(np.full(w.shape[0], lbl, dtype=np.int64))
            coverage_parts.append(cov)
            counts[lbl] = counts.get(lbl, 0) + w.shape[0]
            total += w.shape[0]
        except Exception as e:
            failed.append((sid, type(e).__name__, str(e)[:200]))
    if not Xs:
        X = np.empty((0, WINDOW_SIZE, 48), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
    else:
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
    coverage = np.concatenate(coverage_parts) if coverage_parts else np.empty((0,), dtype=np.float32)

    if X.size:
        nc = int(X.shape[2])
        if fit_norm:
            mean = X.reshape(-1, nc).mean(axis=0).astype(np.float64)
            std = X.reshape(-1, nc).std(axis=0).astype(np.float64)
            std[std == 0] = 1.0
            norm_mean, norm_std = mean, std
        X = ((X.reshape(-1, nc) - norm_mean) / norm_std).reshape(X.shape).astype(np.float32)
    save_binny(out_x, X, y)
    return {
        "split": split_name,
        "windows": int(X.shape[0]),
        "channels": int(X.shape[2]) if X.size else 48,
        "per_class_counts": {int(k): int(v) for k, v in counts.items()},
        "budget_used": bool(budget is not None and X.shape[0] >= budget) if budget else None,
        "failed_samples": failed,
        "mean_fault_coverage": float(coverage.mean()) if coverage.size else 0.0,
        "frac_windows_with_fault": float((coverage > 0).mean()) if coverage.size else 0.0,
    }, norm_mean, norm_std


def main():
    global labels_df
    try:
        print("=== Loading config ===")
        cfg = load_config()
        cfg.setdefault("finalized_pipeline", {})
        cfg["finalized_pipeline"] = {
            "label_source": "sc_type from hv_double_line_90kv_labels.csv",
            "label_mapping": {"0": "LLL", "1": "LL", "2": "SLG", "3": "LLG"},
            "classes": [0, 1, 2, 3],
            "split_unit": "episode_id",
            "split_strategy": "stratified on sc_type, two-stage 70/15/15, seed 42",
            "split_ratios": {"train": TRAIN_FRAC, "validation": VAL_FRAC, "test": TEST_FRAC},
            "random_seed": RANDOM_SEED,
            "window_size_samples": WINDOW_SIZE,
            "window_size_ms": round(WINDOW_SIZE / 6400.0 * 1000, 4),
            "stride_samples": STRIDE,
            "max_windows_per_episode": MAX_WINDOWS_PER_EPISODE,
            "window_budget": WINDOW_BUDGET,
            "fault_alignment": "windows drawn from [t_evnt_start, t_evnt_end]: onset-centred windows first, then stride sweep fully inside the fault span; clamped to episode bounds; episode windows inherit episode sc_type label; per-episode cap enforced",
            "window_placement_correction": "Original sampler placed ~98% of windows in pre-fault steady state (mean fault coverage 0.021). Corrected to fault-interval placement; window size/stride/cap and episode splits unchanged. Evidence: window_signal_diagnostic.json",
            "channels": "all 48 waveform channels across 8 measurement locations; channel order = sorted column names; no silent discards",
            "normalization": "per-channel z-score, fit on train windows only, applied to val/test",
            "storage": "raw binary .x.bin/.y.bin with headers to avoid full in-memory split arrays",
            "no_silent_zero_fill": True,
            "leakage_prevention": "split on episode IDs before windowing; all windows of an episode stay in one split",
            "intent": "small supervised window classifier over fault episodes; windows are episode instances, not independent events",
        }

        # Remove previous binary outputs: save_binny appends when a file exists.
        for stem in ("X_train", "X_val", "X_test"):
            for ext in (".x.bin", ".y.bin"):
                p = OUT_DIR / f"{stem}{ext}"
                if p.exists():
                    p.unlink()

        print("=== Loading labels ===")
        labels_df = load_labels()
        available = available_sample_ids()
        labels_df = labels_df[labels_df["sample_id"].isin(available)].sort_values("sample_id").reset_index(drop=True)
        print(f"Labels rows after matching available PKLs: {len(labels_df)}")

        y_all = labels_df["sc_type"].to_numpy(dtype=np.int64)
        ids_all = labels_df["sample_id"].to_numpy(dtype=np.int64)

        # Reuse the existing episode-level splits verbatim when they are present so
        # that the corrected window placement cannot change split membership.
        split_files = {
            "train": OUT_DIR / "train_episodes.csv",
            "validation": OUT_DIR / "validation_episodes.csv",
            "test": OUT_DIR / "test_episodes.csv",
        }
        if all(p.is_file() for p in split_files.values()):
            ids_train = pd.read_csv(split_files["train"])["sample_id"].to_numpy(dtype=np.int64)
            ids_val = pd.read_csv(split_files["validation"])["sample_id"].to_numpy(dtype=np.int64)
            ids_test = pd.read_csv(split_files["test"])["sample_id"].to_numpy(dtype=np.int64)
            print("Reusing existing episode splits (unchanged).")
            known = set(ids_all.tolist())
            for name, arr in (("train", ids_train), ("validation", ids_val), ("test", ids_test)):
                missing = [i for i in arr.tolist() if i not in known]
                if missing:
                    raise ValueError(f"split {name} references unknown sample ids: {missing[:5]}")
            overlap_tv = set(ids_train) & set(ids_val)
            overlap_tt = set(ids_train) & set(ids_test)
            overlap_vt = set(ids_val) & set(ids_test)
            if overlap_tv or overlap_tt or overlap_vt:
                raise ValueError("episode leakage between splits detected")
        else:
            ids_train, ids_temp, y_train_ep, y_temp_ep = train_test_split(
                ids_all, y_all, train_size=TRAIN_FRAC, random_state=RANDOM_SEED, stratify=y_all
            )
            val_size = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
            ids_val, ids_test, y_val_ep, y_test_ep = train_test_split(
                ids_temp, y_temp_ep, train_size=val_size, random_state=RANDOM_SEED, stratify=y_temp_ep
            )
        print(f"Episode split counts -> train={len(ids_train)} val={len(ids_val)} test={len(ids_test)}")

        def export_episodes(sub_df, fname):
            sub_df[["sample_id", "sc_type", "fault_target", "phase_select"]].to_csv(OUT_DIR / fname, index=False)

        export_episodes(labels_df[labels_df["sample_id"].isin(ids_train)], "train_episodes.csv")
        export_episodes(labels_df[labels_df["sample_id"].isin(ids_val)], "validation_episodes.csv")
        export_episodes(labels_df[labels_df["sample_id"].isin(ids_test)], "test_episodes.csv")
        print("Saved episode split files.")

        print("=== Generating train windows ===")
        st, mean, std = build_split(ids_train, "train", WINDOW_BUDGET["train"], OUT_DIR / "X_train", fit_norm=True)
        finalize_binny(OUT_DIR / "X_train")
        print(f"Train windows: {st['windows']}, per-class: {st['per_class_counts']}")

        print("=== Generating validation windows ===")
        sv, _, _ = build_split(ids_val, "validation", WINDOW_BUDGET["validation"], OUT_DIR / "X_val", norm_mean=mean, norm_std=std)
        finalize_binny(OUT_DIR / "X_val")
        print(f"Validation windows: {sv['windows']}, per-class: {sv['per_class_counts']}")

        print("=== Generating test windows ===")
        sx, _, _ = build_split(ids_test, "test", WINDOW_BUDGET["test"], OUT_DIR / "X_test", norm_mean=mean, norm_std=std)
        finalize_binny(OUT_DIR / "X_test")
        print(f"Test windows: {sx['windows']}, per-class: {sx['per_class_counts']}")

        np.save(OUT_DIR / "norm_mean.npy", mean.astype(np.float32))
        np.save(OUT_DIR / "norm_std.npy", std.astype(np.float32))
        print("Saved normalization stats.")

        lines = [
            "Class distribution report (windows)",
            f"Train: {st['windows']} windows, per-class: {st['per_class_counts']}",
            f"Validation: {sv['windows']} windows, per-class: {sv['per_class_counts']}",
            f"Test: {sx['windows']} windows, per-class: {sx['per_class_counts']}",
            "Normalized label names: 0=LLL, 1=LL, 2=SLG, 3=LLG",
            f"Train budget used? {st['budget_used']}",
            f"Validation budget used? {sv['budget_used']}",
            f"Test budget used? {sx['budget_used']}",
        ]
        (OUT_DIR / "class_distribution_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for l in lines:
            print(l)

        Xtr, ytr = load_binny(OUT_DIR / "X_train")
        Xv, yv = load_binny(OUT_DIR / "X_val")
        Xt, yt = load_binny(OUT_DIR / "X_test")
        integ = [
            "Window-shape and data-integrity report",
            "window_placement = fault_interval (corrected; see README_LIMITATIONS.md)",
            f"train mean fault coverage per window = {st['mean_fault_coverage']:.4f}",
            f"train frac windows overlapping fault = {st['frac_windows_with_fault']:.4f}",
            f"val mean fault coverage per window = {sv['mean_fault_coverage']:.4f}",
            f"test mean fault coverage per window = {sx['mean_fault_coverage']:.4f}",
            f"X_train.shape = {list(Xtr.shape)}",
            f"y_train.shape = {list(ytr.shape)}",
            f"X_val.shape = {list(Xv.shape)}",
            f"y_val.shape = {list(yv.shape)}",
            f"X_test.shape = {list(Xt.shape)}",
            f"y_test.shape = {list(yt.shape)}",
            "n_channels = 48",
            "window_size = 256",
            f"dt = {1/6400.0} s, window_ms = {round(WINDOW_SIZE/6400.0*1000,4)}",
            f"train dtype = {Xtr.dtype}",
            f"contains_nan_train = {bool(np.isnan(Xtr).any())}",
            f"contains_inf_train = {bool(np.isinf(Xtr).any())}",
            f"norm_mean range = [{float(mean.min()):.4f}, {float(mean.max()):.4f}]",
            f"norm_std range = [{float(std.min()):.4f}, {float(std.max()):.4f}]",
            f"y_train min/max = {int(ytr.min())}/{int(ytr.max())}",
            f"any label outside 0..3? train={bool(np.any((ytr<0)|(ytr>3)))} val={bool(np.any((yv<0)|(yv>3)))} test={bool(np.any((yt<0)|(yt>3)))}",
            "episode overlap across splits = False (split files saved)",
            "",
            "Excluded samples (failures during window generation):",
            f"total excluded: {len(st['failed_samples']) + len(sv['failed_samples']) + len(sx['failed_samples'])}",
        ]
        for tag, stats in [("train", st), ("validation", sv), ("test", sx)]:
            integ.append(f"{tag} excluded:")
            for sid, kind, msg in stats["failed_samples"]:
                integ.append(f"  {sid}: {kind}: {msg}")
        (OUT_DIR / "window_shape_integrity_report.txt").write_text("\n".join(integ) + "\n", encoding="utf-8")
        for l in integ:
            print(l)

        cfg["finalized_pipeline"]["realization"] = {
            "train_windows": st["windows"],
            "validation_windows": sv["windows"],
            "test_windows": sx["windows"],
            "n_channels": 48,
            "window_size": WINDOW_SIZE,
            "train_dtype": str(Xtr.dtype),
            "train_per_class_counts": st["per_class_counts"],
            "validation_per_class_counts": sv["per_class_counts"],
            "test_per_class_counts": sx["per_class_counts"],
            "train_budget_limit": WINDOW_BUDGET["train"],
            "validation_budget_limit": WINDOW_BUDGET["validation"],
            "test_budget_limit": WINDOW_BUDGET["test"],
            "window_placement": "fault_interval",
            "mean_fault_coverage": {
                "train": round(st["mean_fault_coverage"], 4),
                "validation": round(sv["mean_fault_coverage"], 4),
                "test": round(sx["mean_fault_coverage"], 4),
            },
            "excluded_train_samples": [int(s[0]) for s in st["failed_samples"]],
            "excluded_validation_samples": [int(s[0]) for s in sv["failed_samples"]],
            "excluded_test_samples": [int(s[0]) for s in sx["failed_samples"]],
            "split_files": {
                "train_episodes": "preprocessed/train_episodes.csv",
                "validation_episodes": "preprocessed/validation_episodes.csv",
                "test_episodes": "preprocessed/test_episodes.csv",
            },
            "array_files": {
                "X_train": "preprocessed/X_train.x.bin",
                "y_train": "preprocessed/X_train.y.bin",
                "X_val": "preprocessed/X_val.x.bin",
                "y_val": "preprocessed/X_val.y.bin",
                "X_test": "preprocessed/X_test.x.bin",
                "y_test": "preprocessed/X_test.y.bin",
                "norm_mean": "preprocessed/norm_mean.npy",
                "norm_std": "preprocessed/norm_std.npy",
            },
            "load_function": "preprocess.load_binny(preprocessed/X_train)",
        }
        cfg["finalized_pipeline"]["availability_note"] = (
            "Legacy INCOMPLETE fields retained verbatim in dataset.* block; "
            "verified realizable state is in finalized_pipeline.*."
        )
        save_config(cfg)
        print("=== DONE ===")
    except Exception:
        print("ERROR in preprocess.py:")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()

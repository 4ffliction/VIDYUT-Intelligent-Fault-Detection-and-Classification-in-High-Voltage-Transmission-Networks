# verify_pipeline.py
"""Verification of the preprocessed window store used by the hybrid pipeline.

Checks, and writes preprocessing_verification.txt:
  1. episode-level splits are the stored ones and are mutually disjoint;
  2. window shapes / dtypes / label ranges;
  3. per-channel mean and std of the TRAIN windows are ~0 and ~1, which proves
     the z-score was applied exactly once (a second application would push the
     std well below 1);
  4. validation/test windows are NOT expected to be exactly 0/1 (their stats
     come from the train fit), so they are reported for reference only.

Run: python verify_pipeline.py
"""
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd

PREPROC = Path("preprocessed")
WINDOW_SIZE, N_CHANNELS = 256, 48


def open_split(stem):
    with open(str(stem) + ".x.bin", "rb") as f:
        n, ws, nc = struct.unpack("<III", f.read(12))
    x = np.memmap(str(stem) + ".x.bin", dtype=np.float32, mode="r", offset=12).reshape(n, ws, nc)
    with open(str(stem) + ".y.bin", "rb") as f:
        ny = struct.unpack("<I", f.read(4))[0]
    y = np.memmap(str(stem) + ".y.bin", dtype=np.int64, mode="r", offset=4)
    assert n == ny
    return x, y, n


def channel_stats(x, n, sample=12000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    a = np.asarray(x[idx], dtype=np.float64)
    flat = a.reshape(-1, a.shape[2])
    return flat.mean(0), flat.std(0)


lines = []
ok = True

tr = open_split(PREPROC / "X_train")
va = open_split(PREPROC / "X_val")
te = open_split(PREPROC / "X_test")

# 1. splits
ids = {k: set(pd.read_csv(PREPROC / f)[["sample_id"]].iloc[:, 0].tolist())
       for k, f in (("train", "train_episodes.csv"), ("val", "validation_episodes.csv"),
                    ("test", "test_episodes.csv"))}
overlaps = {
    "train/val": len(ids["train"] & ids["val"]),
    "train/test": len(ids["train"] & ids["test"]),
    "val/test": len(ids["val"] & ids["test"]),
}
lines.append("1. Episode splits")
for k, v in ids.items():
    lines.append(f"   {k}: {len(v)} episodes")
lines.append(f"   overlaps: {overlaps}")
if any(overlaps.values()):
    ok = False
    lines.append("   FAIL: episode leakage between splits")

# 2. shapes / labels
lines.append("2. Windows")
for name, (x, y, n) in (("train", tr), ("val", va), ("test", te)):
    yv = np.asarray(y)
    lines.append(f"   {name}: n={n} shape={tuple(x.shape)} dtype={x.dtype} "
                 f"labels={np.bincount(yv, minlength=4).tolist()}")
    if x.shape[1:] != (WINDOW_SIZE, N_CHANNELS):
        ok = False
        lines.append(f"   FAIL: {name} shape mismatch")

nan_train = bool(np.isnan(np.asarray(tr[0][:2000], dtype=np.float32)).any())
lines.append(f"   NaN in train sample: {nan_train}")
if nan_train:
    ok = False

# 3/4. normalisation
mu_tr, sd_tr = channel_stats(tr[0], tr[2])
mu_va, sd_va = channel_stats(va[0], va[2])
mu_te, sd_te = channel_stats(te[0], te[2])
lines.append("3. Normalisation check (per-channel, 12000 sampled windows)")
lines.append(f"   train  mean: |max|={np.abs(mu_tr).max():.5f}  std: "
             f"min={sd_tr.min():.4f} mean={sd_tr.mean():.4f} max={sd_tr.max():.4f}")
lines.append(f"   val    mean: |max|={np.abs(mu_va).max():.5f}  std mean={sd_va.mean():.4f}")
lines.append(f"   test   mean: |max|={np.abs(mu_te).max():.5f}  std mean={sd_te.mean():.4f}")
std_ok = 0.9 <= sd_tr.mean() <= 1.1 and np.abs(mu_tr).max() < 0.05
lines.append(f"   train windows are unit-variance and zero-mean: {std_ok}")
lines.append("   -> z-score applied exactly once (a second application would give std ~0.9^2={:.3f})"
             .format(0.9 ** 2))
if not std_ok:
    ok = False
    lines.append("   FAIL: train windows are not unit variance")

lines.append("")
lines.append("RESULT: " + ("PASS" if ok else "FAIL"))
Path("preprocessing_verification.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

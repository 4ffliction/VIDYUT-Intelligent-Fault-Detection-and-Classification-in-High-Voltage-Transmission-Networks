# verify_storage.py
"""Does preprocessed/X_train.x.bin actually contain the fault-interval windows?

Reconstructs the first training episode from its raw PKL with the corrected
sampler, applies the saved train-only z-score, and compares against the first
block of stored windows.
"""
import struct
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import episode_to_windows, DATA_DIR

with open("preprocessed/X_train.x.bin", "rb") as f:
    n, ws, nc = struct.unpack("<III", f.read(12))
X = np.memmap("preprocessed/X_train.x.bin", dtype=np.float32, mode="r", offset=12).reshape(n, ws, nc)
with open("preprocessed/X_train.y.bin", "rb") as f:
    ny = struct.unpack("<I", f.read(4))[0]
y = np.memmap("preprocessed/X_train.y.bin", dtype=np.int64, mode="r", offset=4)
mean = np.load("preprocessed/norm_mean.npy")
std = np.load("preprocessed/norm_std.npy")

ids = pd.read_csv("preprocessed/train_episodes.csv")["sample_id"].to_numpy()
labels = pd.read_csv("hv_double_line_90kv_labels.csv").set_index("sample_id")
sid = int(ids[0])
meta = labels.loc[sid]

df = pd.read_pickle(DATA_DIR / f"{sid}_sample_hv_double_line_90kv.pkl")
cols = sorted(c for c in df.columns if c != "time_s")
dt = float(df["time_s"].iloc[1] - df["time_s"].iloc[0])
raw, cov = episode_to_windows(df[cols], float(meta["t_evnt_start"]), float(meta["t_evnt_end"]), dt=dt)
rawz = ((raw.reshape(-1, nc) - mean) / std).reshape(raw.shape).astype(np.float32)

k = raw.shape[0]
print(f"first train episode id={sid} sc_type={int(meta['sc_type'])} windows={k} coverage={cov.mean():.4f}")
print(f"stored y[0:{k}] labels={np.unique(np.asarray(y[:k])).tolist()}")

stored = np.asarray(X[:k], dtype=np.float32)
d = np.abs(stored - rawz)
print(f"max |stored - reconstructed| = {d.max():.6f}  mean = {d.mean():.6f}")
print(f"stored[0,0,:3]      = {stored[0,0,:3]}")
print(f"reconstructed[0,0,:3]= {rawz[0,0,:3]}")
print(f"stored window0 rms per ch (first 6) = {np.sqrt((stored[0]**2).mean(0))[:6]}")

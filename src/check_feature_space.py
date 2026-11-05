import struct
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from features import engineer_features


def open_split(stem):
    with open(stem + ".x.bin", "rb") as f:
        n, ws, nc = struct.unpack("<III", f.read(12))
    x = np.memmap(stem + ".x.bin", dtype=np.float32, mode="r", offset=12).reshape(n, ws, nc)
    with open(stem + ".y.bin", "rb") as f:
        _ = struct.unpack("<I", f.read(4))[0]
    y = np.memmap(stem + ".y.bin", dtype=np.int64, mode="r", offset=4)
    return x, y, n


mean = np.load("preprocessed/norm_mean.npy").astype(np.float32).reshape(1, 1, -1)
std = np.load("preprocessed/norm_std.npy").astype(np.float32).reshape(1, 1, -1)

xt, yt, nt = open_split("preprocessed/X_train")
xv, yv, nv = open_split("preprocessed/X_val")
rng = np.random.default_rng(0)
tr = rng.choice(nt, 8000, replace=False)
va = rng.choice(nv, 8000, replace=False)

Ztr = np.asarray(xt[tr], dtype=np.float32)
Zva = np.asarray(xv[va], dtype=np.float32)
ytr = np.asarray(yt[tr])
yva = np.asarray(yv[va])

for name, a, b in (
    ("z-scored", Ztr, Zva),
    ("physical (de-normalised)", Ztr * std + mean, Zva * std + mean),
):
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                class_weight="balanced", random_state=42, n_jobs=8)
    rf.fit(engineer_features(a), ytr)
    p = rf.predict(engineer_features(b))
    print(f"{name}: val macro_f1={f1_score(yva, p, average='macro'):.4f} "
          f"acc={(p == yva).mean():.4f} pred_dist={np.bincount(p, minlength=4).tolist()}")

# per-class mean RMS of a faulted current channel, physical vs z
ch = 3
p_rms = [round(float(np.sqrt(np.mean((Ztr[ytr == c][:, :, ch] * float(std[0, 0, ch]) + float(mean[0, 0, ch])) ** 2))), 1) for c in range(4)]
z_rms = [round(float(np.sqrt(np.mean(Ztr[ytr == c][:, :, ch] ** 2))), 3) for c in range(4)]
print("physical RMS by class:", p_rms)
print("z        RMS by class:", z_rms)

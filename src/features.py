# features.py
"""Feature extraction for the VIDYUT-KAVACHA CNN-Random Forest hybrid.

Two feature families are produced from the stored preprocessed windows:

1. Engineered electrical features (per window, per channel):
   RMS, standard deviation, peak-to-peak amplitude, maximum absolute amplitude.
   48 channels x 4 statistics = 192 features.

2. CNN embeddings (per window): 96-dimensional output of the convolutional
   feature extractor immediately before the linear classifier.

Both are computed batch-wise directly from memory-mapped arrays; the full
dataset is never copied into RAM.

The stored windows in preprocessed/*.x.bin are ALREADY per-channel z-scored
(fit on the training split by preprocess.py).  Nothing here re-applies
normalisation.

The engineered statistics are reported in PHYSICAL UNITS (amperes / volts),
recovered by inverting the saved train-only z-score (x = z * std + mean), so
the features are the electrical quantities the names imply.

Measured on 8000 random train / 8000 random validation windows of the corrected
splits (`python check_feature_space.py`):

    features on z-scored windows : val macro F1 0.9537
    features on physical units   : val macro F1 0.9546

Both are equivalent on correctly fault-aligned windows; physical units are used
for interpretability.  The CNN consumes the z-scored windows unchanged.
"""

import numpy as np
import torch

N_CHANNELS = 48
N_FEATURES_PER_CHANNEL = 4
N_ENG_FEATURES = N_CHANNELS * N_FEATURES_PER_CHANNEL
CNN_EMBED_DIM = 96

FEATURE_NAMES = []
for _c in range(N_CHANNELS):
    for _stat in ("rms", "std", "ptp", "maxabs"):
        FEATURE_NAMES.append(f"ch{_c:02d}_{_stat}")


def load_norm(preproc_dir="preprocessed"):
    """Train-fitted per-channel mean/std used by preprocess.py."""
    import os
    mean = np.load(os.path.join(preproc_dir, "norm_mean.npy")).astype(np.float32)
    std = np.load(os.path.join(preproc_dir, "norm_std.npy")).astype(np.float32)
    return mean, std


def engineer_features(X):
    """RMS, std, peak-to-peak and max-abs for each channel of a window batch.

    X: (N, window_size, n_channels) -> (N, n_channels * 4)
    """
    X = np.asarray(X, dtype=np.float32)
    N, _, nc = X.shape
    out = np.empty((N, nc * N_FEATURES_PER_CHANNEL), dtype=np.float32)
    for c in range(nc):
        ch = X[:, :, c]
        out[:, c * 4 + 0] = np.sqrt(np.mean(ch ** 2, axis=1))
        out[:, c * 4 + 1] = ch.std(axis=1)
        out[:, c * 4 + 2] = ch.max(axis=1) - ch.min(axis=1)
        out[:, c * 4 + 3] = np.abs(ch).max(axis=1)
    return out


def features_batched(x_mm, indices=None, batch_size=4096, mean=None, std=None):
    """Engineer features for a memory-mapped split without materialising it.

    If mean/std are given, the stored z-scored windows are converted back to
    physical units (x = z * std + mean) before the statistics are computed.
    """
    total = x_mm.shape[0] if indices is None else len(indices)
    out = np.empty((total, N_ENG_FEATURES), dtype=np.float32)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        sl = indices[start:stop] if indices is not None else slice(start, stop)
        X = np.asarray(x_mm[sl], dtype=np.float32)
        if mean is not None and std is not None:
            X = X * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
        out[start:start + X.shape[0]] = engineer_features(X)
    return out


@torch.no_grad()
def embeddings_batched(model, x_mm, indices=None, batch_size=2048, device="cpu"):
    """CNN embeddings for a memory-mapped split, computed in batches."""
    total = x_mm.shape[0] if indices is None else len(indices)
    out = np.empty((total, CNN_EMBED_DIM), dtype=np.float32)
    model.eval()
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        sl = indices[start:stop] if indices is not None else slice(start, stop)
        X = np.asarray(x_mm[sl], dtype=np.float32)
        out[start:start + X.shape[0]] = model.embed(torch.from_numpy(X).to(device)).cpu().numpy()
    return out

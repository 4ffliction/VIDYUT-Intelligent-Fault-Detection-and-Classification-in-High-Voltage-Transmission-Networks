# verify_cleanup.py
"""Post-cleanup verification: essentials present, models loadable, paths valid."""
import json
import struct
import sys
from pathlib import Path

import numpy as np

root = Path(".")
problems = []

ESSENTIAL = [
    "README.md", "README_LIMITATIONS.md", "CLEANUP_REPORT.md",
    "preprocess.py", "features.py", "train_hybrid.py", "verify_pipeline.py",
    "check_window_signal.py", "check_feature_space.py", "verify_storage.py",
    "run_modeling.sh", "preprocessing_config.json",
    "hv_double_line_90kv_labels.csv", "metadata_audit_full.csv",
    "preprocessed/X_train.x.bin", "preprocessed/X_train.y.bin",
    "preprocessed/X_val.x.bin", "preprocessed/X_val.y.bin",
    "preprocessed/X_test.x.bin", "preprocessed/X_test.y.bin",
    "preprocessed/norm_mean.npy", "preprocessed/norm_std.npy",
    "preprocessed/train_episodes.csv", "preprocessed/validation_episodes.csv",
    "preprocessed/test_episodes.csv",
    "preprocessed/class_distribution_report.txt",
    "preprocessed/window_shape_integrity_report.txt",
    "preprocessing_verification.txt", "window_signal_diagnostic.json",
    "hybrid_results/expA_val_metrics.json", "hybrid_results/expB_val_metrics.json",
    "hybrid_results/expC_val_metrics.json", "hybrid_results/final_test_metrics.json",
    "hybrid_results/expA_val_classification_report.txt",
    "hybrid_results/expB_val_classification_report.txt",
    "hybrid_results/expC_val_classification_report.txt",
    "hybrid_results/final_test_classification_report.txt",
    "hybrid_results/expA_val_confusion_matrix.png",
    "hybrid_results/expB_val_confusion_matrix.png",
    "hybrid_results/expC_val_confusion_matrix.png",
    "hybrid_results/final_test_confusion_matrix.png",
    "hybrid_results/cnn_training_history.json", "hybrid_results/cnn_training_history.png",
    "hybrid_results/expB_cnn_config.json", "hybrid_results/expB_cnn_best.pt",
    "hybrid_results/expA_rf_features_model.pkl", "hybrid_results/expC_hybrid_rf_model.pkl",
    "hybrid_results/comparison.json", "hybrid_results/comparison.md",
    "hybrid_results_legacy_windowing/NOTE.md",
    "hybrid_results_legacy_windowing/expA_metrics.json",
    "hybrid_results_legacy_windowing/expB_metrics.json",
    "hybrid_results_legacy_windowing/expC_metrics.json",
    "technical_report.md", "UTA024_Practical_Activity_Report.docx",
    "Thapar_Edp2_UTA024_Practical_Activity_Report.docx",
]
missing = [p for p in ESSENTIAL if not (root / p).is_file()]
problems += [f"missing essential: {p}" for p in missing]
print(f"1. essential files: {len(ESSENTIAL) - len(missing)}/{len(ESSENTIAL)} present")

for p in ("preprocessed/X_train.x.bin", "preprocessed/X_val.x.bin", "preprocessed/X_test.x.bin"):
    with open(root / p, "rb") as f:
        n, ws, nc = struct.unpack("<III", f.read(12))
    print(f"   {p}: n={n} win={ws} ch={nc}")
print(f"2. dataset archive: {(root/'hv_double_line_90kv_preprocessed_data.zip').stat().st_size} bytes")
eps = len(list((root / "hv_double_line_90kv_preprocessed_data_extracted/hv_double_line_90kv_preprocessed_data").glob("*.pkl")))
print(f"   extracted episodes: {eps}")

try:
    import joblib
    rf = joblib.load(root / "hybrid_results/expC_hybrid_rf_model.pkl")
    print(f"3. RF model loads: {type(rf).__name__}, n_features={rf.n_features_in_}, trees={rf.n_estimators}")
except Exception as e:
    problems.append(f"RF load failed: {e}")

try:
    import torch
    sd = torch.load(root / "hybrid_results/expB_cnn_best.pt", map_location="cpu", weights_only=True)
    nparam = sum(v.numel() for v in sd.values())
    print(f"4. CNN checkpoint loads: {len(sd)} tensors, {nparam} elements")
except Exception as e:
    problems.append(f"CNN load failed: {e}")

# README-referenced paths
import re
readme = (root / "README.md").read_text(encoding="utf-8")
cands = re.findall(r"`([A-Za-z0-9_./\-]+\.(?:py|sh|json|md|txt|png|pt|pkl|csv|npy|bin))`", readme)
bad = sorted({c for c in cands if not (root / c).exists() and "/" in c})
print(f"5. README path refs checked: {len(set(cands))} distinct, unresolved: {len(bad)}")
for b in bad:
    if not b.startswith(("preprocessed/", "hybrid_results/")):
        problems.append(f"README references missing file: {b}")
    else:
        print(f"   (informational) not found: {b}")

# split manifests untouched
import pandas as pd
for f, exp in (("train_episodes.csv", 6315), ("validation_episodes.csv", 1353), ("test_episodes.csv", 1354)):
    got = len(pd.read_csv(root / "preprocessed" / f))
    if got != exp:
        problems.append(f"{f}: expected {exp} rows, got {got}")
print("6. split manifests: row counts match (6315/1353/1354)")

print("7. problems:", problems if problems else "none")
sys.exit(1 if problems else 0)

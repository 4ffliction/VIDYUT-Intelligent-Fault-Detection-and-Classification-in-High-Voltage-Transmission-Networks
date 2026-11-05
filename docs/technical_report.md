# VIDYUT-KAVACHA Technical Report

> **STATUS (superseded in part).** The windowing description and the window
> counts below predate a correction: the original sampler placed ~98 % of
> windows in pre-fault steady state, so every model in the earlier run scored at
> chance. Current, verified pipeline and results are in `README.md`;
> root cause and audit trail are in `README_LIMITATIONS.md`. Verified window
> counts after the fix: train 105,728 / validation 22,908 / test 22,519.

## Objective
Build and evaluate a supervised window classifier for the PROTECT-90 90 kV double-circuit line dataset, covering a Random Forest baseline over engineered features and a compact 1D CNN over waveform windows.

## Dataset and preprocessing
- Source archive: `hv_double_line_90kv_preprocessed_data.zip`, manually supplied and verified this session.
- Archive validation: size matched, MD5 matched, `zipfile.testzip()` passed.
- Extraction: `hv_double_line_90kv_preprocessed_data_extracted/` with 9022 episode PKLs under `hv_double_line_90kv_preprocessed_data/`.
- Labels: `hv_double_line_90kv_labels.csv`, using `sc_type` as the fault class.
- Class mapping (verified from companion repo README):
  - 0 = LLL
  - 1 = LL
  - 2 = SLG
  - 3 = LLG

### Split design
- Split unit: episode.
- Strategy: stratified on `sc_type`, two-stage 70/15/15, seed 42.
- Episode counts: train 6315, validation 1353, test 1354.
- Leakage control: all windows from one episode stay in one split; split happens before windowing.

### Window design
- Window length: 256 samples = 40 ms at 6400 Hz.
- Stride: 64 samples.
- Fault alignment: windows anchored at `t_evnt_start` when present and clamped to episode bounds; episode windows inherit the episode `sc_type` label.
- Per-episode cap: `MAX_WINDOWS_PER_EPISODE=24`.
- Global window budget: train 500000, val 120000, test 120000.
- Channels: all 48 waveform channels across 8 measurement locations, column-name sorted order, no silent discarding.

### Normalization
- Per-channel z-score fit on train windows only.
- Applied to validation and test.
- Saved as `preprocessed/norm_mean.npy`, `preprocessed/norm_std.npy`.

### Outputs
- `preprocessed/X_train.x.bin`, `preprocessed/X_train.y.bin`
- `preprocessed/X_val.x.bin`, `preprocessed/X_val.y.bin`
- `preprocessed/X_test.x.bin`, `preprocessed/X_test.y.bin`
- `preprocessed/train_episodes.csv`, `preprocessed/validation_episodes.csv`, `preprocessed/test_episodes.csv`
- `preprocessed/class_distribution_report.txt`, `preprocessed/window_shape_integrity_report.txt`

### Verified window counts
- Train: 151536 windows
- Validation: 32472 windows
- Test: 32496 windows
- Shapes: (N, 256, 48), float32.
- No NaN, no Inf in train.
- Train label range 0..3.
- All classes present in all splits with similar balanced counts.
- Excluded sample during window generation: 7154 due to truncated pickle.

## Models

### Random Forest baseline
- Features: per-window engineered features from all 48 channels, including RMS, crest factor, mean abs, 5th/95th percentile, skew, kurtosis.
- Train-only feature normalization.
- RandomForest, n_estimators=400, class_weight="balanced", random_state=42.
- Saved model: `baseline/rf_model.pkl`.
- Outputs: `baseline/baseline_metrics.json`, `baseline/baseline_confusion_matrix.png`.

### 1D CNN
- Compact 1D CNN over waveform windows.
- Trained on normalized waveform windows.
- Validation-loss checkpointing and early stopping based on validation loss.
- Saved best weights: `cnn/best_model.pt`.
- Outputs: `cnn/training_history.json`, `cnn/cnn_test_metrics.json`.

## Evaluation
- Consolidated evaluation script: `evaluate.py`.
- Outputs:
  - `evaluation/evaluation_report.json`
  - `evaluation/evaluation_confusion_matrix.png`
- Metrics reported per split and per class: accuracy, macro F1, per-class precision/recall/f1, confusion matrix.

## Notes
- Train-only normalization and split-before-windowing were used throughout.
- No test-set tuning was performed; CNN selection was based on validation loss only.
- No waveform plots or metrics were fabricated; everything was read from the saved pipeline artifacts.
- On this host, full-window generation was kept practical with a per-episode cap and a global window budget rather than generating every possible window.

## Deliverables
- `preprocess.py`
- `preprocessing_config.json`
- `train_baseline.py`
- `train_cnn.py`
- `evaluate.py`
- `baseline/*`
- `cnn/*`
- `evaluation/*`
- `preprocessed/*`
- this report

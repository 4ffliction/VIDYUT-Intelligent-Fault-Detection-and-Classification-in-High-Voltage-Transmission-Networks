# VIDYUT-KAVACHA — CNN + Random Forest hybrid for PROTECT-90 fault classification

Four-class fault classification (LLL, LL, SLG, LLG) on the PROTECT-90 90 kV
double-circuit line dataset, comparing:

| Exp | Model |
|---|---|
| **A** | Random Forest on engineered electrical features |
| **B** | Compact 1D CNN classifier |
| **C** | CNN embedding + engineered features → Random Forest (hybrid) |

Test split is evaluated **once**, for the single model selected on validation.

---

## 1. Dataset

| Item | Value |
|---|---|
| Archive | `hv_double_line_90kv_preprocessed_data.zip` (12.0 GB, 9,022 episode pickles) |
| Extracted | `hv_double_line_90kv_preprocessed_data_extracted/hv_double_line_90kv_preprocessed_data/` |
| Labels | `hv_double_line_90kv_labels.csv` (9,022 rows, 49 columns) |
| Label field | `sc_type` → 0 = LLL, 1 = LL, 2 = SLG, 3 = LLG |
| Episode record | 6,400 samples @ 6,400 Hz = 1.0 s |
| Channels | 48 analogue channels (phase currents in A, phase voltages in V), 8 measurement locations × 3 phases × {current, voltage}; channel order = sorted column names |
| Class counts (episodes) | LLL 2,273 · LL 2,251 · SLG 2,279 · LLG 2,219 |
| Fault targets | Line_1_2_a 3,191 · Line_2_3_a 3,165 · Line_2_3_b 1,338 · Line_1_2_b 1,328 |
| Fault duration | mean ≈ 225 ms per class (class-independent); 828 episodes shorter than one 40 ms window |
| Corrupt file | `7154_*.pkl` (truncated pickle) — excluded, logged in `preprocessed/window_shape_integrity_report.txt` |

## 2. Splits (episode level, unchanged)

Split unit is the **episode**; a window never crosses splits.

| Split | Episodes | Windows |
|---|---|---|
| train | 6,315 | 105,728 |
| validation | 1,353 | 22,908 |
| test | 1,354 | 22,519 |

`python verify_pipeline.py` confirms zero episode overlap between all pairs of
splits. The split CSVs (`preprocessed/*_episodes.csv`) are reused verbatim by
`preprocess.py`, so regenerating windows cannot change split membership.

## 3. Preprocessing

* **Window** 256 samples = 40 ms, stride 64 samples = 10 ms, max 24 windows per
  episode, each window inherits the episode `sc_type`.
* **Placement (corrected).** Windows are drawn from the fault interval
  `[t_evnt_start, t_evnt_end]`: three onset-anchored windows first, then a
  stride sweep fully inside the fault span. Mean fault coverage per window is
  **0.964**, and **100 %** of windows overlap the fault in all three splits.
  See §7 and `README_LIMITATIONS.md` — the original sampler put ~98 % of
  windows in pre-fault steady state.
* **Normalisation.** Per-channel z-score, **fit on train windows only**, then
  applied to train/validation/test and stored in
  `preprocessed/norm_mean.npy`, `preprocessed/norm_std.npy`.
  Verified applied exactly once: train windows have per-channel
  `|mean|max = 0.0012`, `std mean = 1.0196` (`preprocessing_verification.txt`).
  The CNN consumes these normalised windows directly; nothing re-normalises.
* **Storage.** Raw binary `.x.bin`/`.y.bin` with headers, read through
  `numpy.memmap`. No full-dataset copies in RAM during training.

## 4. Models

**A — Random Forest on engineered features (192-D).** Per window and channel:
RMS, standard deviation, peak-to-peak, max absolute amplitude (48 × 4). These
are computed in **physical units** by inverting the saved z-score
(`x = z·std + mean`), so the numbers are amperes/volts as the names imply.
Measured on 8,000 train / 8,000 validation windows, z-scored and physical-unit
features are equivalent (0.9537 vs 0.9546 macro F1), so this choice costs
nothing and buys interpretability. RF: 400 trees, `min_samples_leaf=2`,
`class_weight="balanced"`, seed 42.

**B — Compact 1D CNN (37,316 parameters).** Conv1d(48→32, k5) → BN → ReLU →
MaxPool2 → Conv1d(32→64, k5) → BN → ReLU → MaxPool2 → Conv1d(64→96, k3) → BN →
ReLU → global average pool → linear(96→4). AdamW, lr 1e-3, weight decay 1e-4,
batch 128, `ReduceLROnPlateau`, max 20 epochs, early stopping on **validation
loss** (patience 5), best-validation-loss checkpoint kept.

**C — Hybrid.** 96-D CNN embedding (from B's best validation checkpoint) ⊕
192 engineered features = 288-D → Random Forest (same settings as A).

**Training subset.** All three experiments train on the same fixed-seed
class-balanced stratified subset: **16,384 windows** (seed 42) of the 105,728
available, per class `[4085, 4066, 4193, 4040]`. Validation and test use every
window. This is a deliberate runtime/statistics trade-off, recorded in every
metrics file.

## 5. Results

Model selection uses the validation split only. Values are from
`hybrid_results/*_val_metrics.json`.

| Experiment | Val accuracy | Val macro precision | Val macro recall | Val macro F1 | Params / model size | Train time |
|---|---|---|---|---|---|---|
| A: RF, engineered features | 0.9611 | 0.9608 | 0.9608 | **0.9608** | 400 trees, 30 MB | 33 s |
| B: compact 1D CNN | 0.9992 | 0.9992 | 0.9992 | **0.9992** | 37,316 params, 0.16 MB | 185 s |
| C: CNN embedding + features → RF | 0.9989 | 0.9989 | 0.9989 | **0.9989** | 400 trees, 1.0 MB | 9 s (+ B) |

**Selected on validation: B, the compact 1D CNN.**
Held-out test evaluation (single run, `hybrid_results/final_test_metrics.json`):

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| B: compact 1D CNN | 0.9990 | 0.9990 | 0.9990 | 0.9990 |

Test confusion matrix (22 errors / 22,519 windows):

|  | LLL | LL | SLG | LLG |
|---|---|---|---|---|
| **LLL** | 5795 | 0 | 2 | 1 |
| **LL** | 2 | 5497 | 0 | 0 |
| **SLG** | 2 | 0 | 5797 | 0 |
| **LLG** | 3 | 3 | 9 | 5408 |

Per-class validation F1:

| Experiment | LLL | LL | SLG | LLG |
|---|---|---|---|---|
| A | 0.9973 | 0.9271 | 0.9887 | 0.9301 |
| B | 0.9991 | 0.9998 | 0.9991 | 0.9987 |
| C | 0.9985 | 0.9997 | 0.9990 | 0.9985 |

### Which model performed best, and why

**B (compact 1D CNN) is the best model**, by a very small margin over C and a
clear margin over A.

* **The signal is temporal, not just amplitude.** A collapses each 40 ms window
  into four magnitude statistics per channel. That is enough to separate faults
  that differ in gross current asymmetry (LLL vs SLG: 0.997 vs 0.989), but it
  loses the waveform morphology that separates the two *two-phase* classes —
  LL (0.927) and LLG (0.930) are A's weak points, because both are two-phase
  faults whose distinction lives in the transient shape and ground-path
  response rather than in per-channel RMS. The CNN sees the full 256 × 48
  waveform and resolves them to 0.9998 / 0.9987.
* **The hybrid mostly inherits the CNN.** C reaches 0.9989 because its 288-D
  input is dominated by the 96 learned embedding dimensions; the 192 handcrafted
  features add no separable information the CNN has not already encoded. An RF
  over the embedding also quantises the CNN's smooth softmax decision into
  axis-aligned tree splits, which is why C lands marginally *below* B rather
  than above it.
* **B vs C is not a meaningful win.** 0.9992 vs 0.9989 on validation is 19 vs
  25 errors out of 22,908 windows — six windows. B is chosen because it is
  simpler (one model, 0.16 MB, direct softmax), not because it is decisively
  more accurate. A is decisively worse.
* **Where the hybrid would pay off** is a regime the compact CNN handled
  poorly — small data, few epochs, or heavy label noise. On this dataset with
  16k windows the CNN already saturates, so the fused RF has nothing to fix.

## 6. Reproducing

Environment used: Python 3.14.0, PyTorch 2.9.1 (CPU), scikit-learn 1.9.1,
NumPy 2.3.5, pandas 3.1.0, matplotlib 3.11.2, joblib 1.6.0. No GPU.
16 CPU cores.

```bash
python -m pip install -r requirements.txt        # core runtime (6 pinned packages)
python -m pip install -r requirements-dev.txt    # optional: .docx report builders
```

Pinned versions, per-package purpose and import locations are in
`DEPENDENCY_AUDIT.md`. The pins were verified by installing into a clean
virtual environment and running the saved CNN checkpoint and RF model.

```bash
# 1. Build the window store (~8 min, ~6.4 GB output)
python preprocess.py

# 2. Verify splits / shapes / normalisation (~30 s)
python verify_pipeline.py

# 3. Run all three experiments (~4 min)
python train_hybrid.py

# optional: also score A and B on test (documented deviation from the
# "one test evaluation" protocol)
python train_hybrid.py --test-all
```

Auxiliary checks used to diagnose the windowing defect:

```bash
python check_window_signal.py   # legacy vs fault-interval windows, 200 episodes
python check_feature_space.py   # z-scored vs physical-unit features
python verify_storage.py        # stored bins vs independent reconstruction
```

All scripts are deterministic given seed 42.

## 7. Artifacts

| Path | Contents |
|---|---|
| `features.py` | Feature extraction: engineered statistics + CNN embeddings, batch-wise |
| `preprocess.py` | Window generation, splits, train-only normalisation |
| `train_hybrid.py` | Experiments A/B/C, selection, single test evaluation |
| `verify_pipeline.py` | Split/shape/normalisation verification |
| `hybrid_results/exp{A,B,C}_val_metrics.json` | Validation metrics, per-class, CM, report |
| `hybrid_results/exp{A,B,C}_val_classification_report.txt` | Validation classification reports |
| `hybrid_results/exp{A,B,C}_val_confusion_matrix.png` | Validation confusion matrices |
| `hybrid_results/final_test_metrics.json` + `_classification_report.txt` + `_confusion_matrix.png` | Test, selected model only |
| `hybrid_results/cnn_training_history.{json,png}` | CNN loss/accuracy curves |
| `hybrid_results/expB_cnn_best.pt` | Best validation-loss CNN checkpoint |
| `hybrid_results/expA_rf_features_model.pkl`, `expC_hybrid_rf_model.pkl` | RF models |
| `hybrid_results/comparison.{json,md}` | Comparison table |
| `hybrid_results_legacy_windowing/` | Archived chance-level results from the original windowing (see limitations) |
| `window_signal_diagnostic.json` | Evidence for the windowing defect |

## 8. CNN training notes

20 epochs were run; early stopping (patience 5) never triggered because
validation loss kept improving. Best validation loss 0.00387 at epoch 18, which
is the checkpoint used for evaluation and for C's embeddings. Validation
accuracy rose from 0.9868 (epoch 1) to a peak of 0.9992.

## 9. Limitations

See **`README_LIMITATIONS.md`** for the honest ledger: the pre-fault windowing
defect and how it was found, failed experiments, subset usage, computational
constraints, and why 99.9 % accuracy on this dataset should not be read as
99.9 % accuracy on a real grid.

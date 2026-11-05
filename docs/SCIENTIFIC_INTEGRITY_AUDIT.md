# SCIENTIFIC_INTEGRITY_AUDIT.md — VIDYUT / PROTECT-90 pipeline

Independent review of the full experimental path: dataset → preprocessing →
features → training → selection → test evaluation.
Read-only: **no pipeline file was modified during this audit.**
Date: 4 November 2026.

Evidence is cited by file and function/line. Where a claim in a document could
not be reproduced from code or artifacts, that is stated rather than assumed.

---

## Verdict summary

| # | Check | Verdict |
|---|---|---|
| 1 | Episode-level splits, not window-level | **PASS** |
| 2 | No episode in more than one split | **PASS** |
| 3 | Correct time vector and sampling interval | **PASS** (after a real bug; see §3) |
| 4 | Every window overlaps the fault interval | **PASS** |
| 5 | Coverage from actual fault boundaries, not fallback values | **PASS now — was FAIL** (§5) |
| 6 | Normalisation fitted on training data only | **PASS** |
| 7 | Val/test use the same training statistics | **PASS** |
| 8 | No test data influences selection | **PASS**, with one disclosure (§8) |
| 9 | Engineered features free of label/future information | **PASS** |
| 10 | CNN embeddings consistent between training and evaluation | **PASS** (byte-exact reproduction) |
| 11 | RF and hybrid use the intended inputs | **PASS** |
| 12 | Training subset documented (sampling + class/episode spread) | **PASS now — was NEEDS REVIEW** (§12) |

Two checks failed at an earlier stage of the project and were corrected. Both
corrections are documented in `README_LIMITATIONS.md` §1 and are re-verified
here from code and artifacts — not from prose.

---

## 1. Episode-level splits, not random window-level splits — PASS

**Evidence.** `preprocess.py:296-325` splits **episode IDs**
(`ids_all = labels_df["sample_id"]`) with `train_test_split(..., stratify=y_all)`
(stratified on `sc_type`), producing `ids_train`, `ids_val`, `ids_test`. Only
afterwards does `build_split()` (`preprocess.py:200`) iterate those IDs and append
each episode's windows (`preprocess.py:207-210`). No window is ever assigned a
split independently.

**Verified.** `preprocessed/{train,validation,test}_episodes.csv` hold 6,315 /
1,353 / 1,354 distinct episodes; window totals (105,728 / 22,908 / 22,519) are
multiples of per-episode window counts (2–24 each).

**Why it matters.** Window-level splitting would put windows of the same
simulation in both train and validation, leaking the episode's specific
parameters and inflating every metric.

## 2. No episode appears in more than one split — PASS

**Evidence.** `preprocess.py:312-317` computes the three pairwise set overlaps and
raises `ValueError("episode leakage between splits detected")` if any is
non-empty. This runs on every preprocessing invocation, including when the split
CSVs are reused (`preprocess.py:296-311`).

**Verified.** `verify_pipeline.py` (rerun during this audit):
```
overlaps: {'train/val': 0, 'train/test': 0, 'val/test': 0}
```

**Independent cross-check.** For the train split, per-episode window counts were
re-derived from all 6,315 pickles and the resulting block boundaries were checked
against the stored label array: `counts_match = True` (sum = 105,728 exactly) and
`block_labels_consistent = True` (every window block carries exactly its
episode's `sc_type`). One corrupt pickle (`7154`) is skipped and contributes zero
windows. Artifact: `subset_stats.json`.

## 3. Correct time vector and sampling interval — PASS (after a real bug)

**Current code.** `build_windows_for_episode()` (`preprocess.py:136-147`) reads
`time_s` from the **full** frame, computes
`dt = time_s[1] - time_s[0]`, and passes it explicitly:
`episode_to_windows(df_feat, t_start, t_end, dt=dt)`. `episode_to_windows()`
(`preprocess.py:70`) uses `dt` to convert `t_evnt_start/end` into sample indices.

**Guard.** `preprocess.py:103`:
```python
if t_start is not None and not (dt and dt > 0):
    raise ValueError("fault alignment requested (t_evnt_start present) but no valid dt; refusing silent fallback")
```
So the historical failure mode cannot recur silently.

**Historical bug (disclosed).** The original code passed `df_feat` — a frame with
`time_s` already removed — so `dt` was always `None` and the alignment branch
never ran. See §5 and `README_LIMITATIONS.md` §1.

**Verified.** Across all 6,315 train episodes: `distinct n = [6400]`,
`distinct dt = [0.000156]` → 6,400 samples at 6,400 Hz = 1.0 s, matching the
dataset card (`docs`/`zenodo_README.md`: "6.4 kHz, 128 samples per 50 Hz cycle").
40 ms window = 256 samples, consistent with `WINDOW_SIZE=256`.

## 4. Every window overlaps the fault interval — PASS

**Evidence.** `episode_to_windows()` builds onset-centred windows
(`f0-128`, `f0`, `f1-256`) first, then a stride sweep **inside** `[f0, f1]`,
capped at 24 (`preprocess.py:105-124`). Coverage per window is computed against
those bounds (`preprocess.py:129-133`) and aggregated in `build_split`
(`preprocess.py:212`).

**Verified.** `preprocessed/window_shape_integrity_report.txt`:
```
window_placement = fault_interval
train mean fault coverage per window = 0.9642
train frac windows overlapping fault = 1.0000
val   mean fault coverage per window = 0.9638
test  mean fault coverage per window = 0.9639
```
`frac = 1.0000` in all three splits: **no window is entirely pre-fault.**

**Why it matters.** A fault-type label cannot be learned from pre-fault steady
state. The original sampler produced 7.0 % fault-overlapping windows and every
model scored at chance (see §5).

**Residual caveat.** 828 of 9,022 episodes have faults shorter than one 40 ms
window; those yield only 2–3 onset-centred windows whose coverage is partial by
construction (minimum 2 windows per episode is observed in the test set). They
are not dropped — that would bias the evaluation — but they carry less evidence
per episode.

## 5. Fault coverage from actual fault boundaries, not fallback values — PASS now, was FAIL

**The original failure.** Because `dt` was `None`, `episode_to_windows` fell to
its fallback branch, which set `f0 = 0`, `f1 = n`. Coverage was then computed
against those *fallback* bounds, so it reported **1.0000 while the windows were
entirely pre-fault**. The metric was self-referential: it could not detect its own
failure. The first attempted fix therefore appeared to succeed and still produced
105,728-era window counts with chance-level models.

**How it was caught.** Comparing samplers on one episode with a 19.2 ms fault:
`preprocess` returned 24 windows while the independent sampler returned 3. The
coverage number was correct-looking and wrong.

**Current state.** Two structural fixes: (a) `dt` is passed explicitly and a
missing `dt` raises (`preprocess.py:103`); (b) coverage is derived from the
`f0/f1` actually used to place windows, and a window list can only be non-empty
if alignment succeeded. A third, independent check now exists: for the test split,
all 1,354 episodes were re-windowed from raw PKLs and compared **window by
window** to the stored arrays — `max|diff| = 3.81e-06` (float32 rounding), 0
mismatches (`episode_evaluation/episode_level_metrics.json →
protocol.reconstruction_max_abs_diff`).

**Verdict.** PASS on the current pipeline. The historical failure is disclosed in
`README_LIMITATIONS.md` and the invalid results are retained in
`hybrid_results_legacy_windowing/` rather than deleted.

## 6. Normalisation fitted exclusively on training data — PASS

**Evidence.** `preprocess.py:336` calls
`build_split(ids_train, "train", ..., fit_norm=True)`; inside `build_split`
(`preprocess.py:231-236`) the per-channel mean/std are computed only when
`fit_norm` is true:
```python
if fit_norm:
    mean = X.reshape(-1, nc).mean(axis=0)
    std  = X.reshape(-1, nc).std(axis=0)
```
Validation (`preprocess.py:341`) and test (`preprocess.py:346`) receive
`norm_mean=mean, norm_std=std` and never fit.

**Verified.** Train windows are zero-mean and unit-variance by construction:
`|mean|max = 0.00122`, `std mean = 1.0196` (`preprocessing_verification.txt`,
rerun → RESULT: PASS).

**Why it matters.** Fitting statistics on val/test leaks distributional
information from the evaluation set into training.

## 7. Validation and test transformed with the same training statistics — PASS

**Evidence.** The same `mean`/`std` arrays are applied inside `build_split`
(`preprocess.py:237`) and persisted as `preprocessed/norm_mean.npy` /
`norm_std.npy`, which every downstream script reads (`features.load_norm`).
No script re-fits normalisation.

**Double-application check.** If the z-score were applied twice, train std would
collapse to ≈0.81 rather than 1.02. Observed `std mean = 1.0196`, so it was
applied exactly once. Reported automatically by `verify_pipeline.py`.

## 8. No test labels or metrics influence selection — PASS, one disclosure

**Evidence.**
* CNN checkpoint selection: best **validation loss** only —
  `train_hybrid.py:305-307` (`if vloss < best_loss`) with `vloss` computed over
  `val_split`; early stopping patience 5 (`train_hybrid.py:79, 310-311`).
* Model selection among A/B/C: validation macro-F1 only —
  `train_hybrid.py:434`
  `best_name = max(candidates, key=lambda k: candidates[k]["val"]["macro_f1"])`.
* Test is touched **after** selection, once: `train_hybrid.py:463-465`
  (`y_test = test_s.labels(); pred, prob = test_predict(best_name); resTest = evaluate(...)`).
* The test set is not used for RF hyperparameters (fixed a priori,
  `train_hybrid.py:76-80`), nor for normalisation, nor for window placement.

**Disclosure (deviations a reviewer should know).**
1. A `--test-all` flag exists (`train_hybrid.py:340`) that would score A and B on
   test as well. **It was not used** for the reported results; `comparison.json`
   records `"test_evaluations": "selected model only"`.
2. During development the *legacy* (defective-window) run's test metrics were
   inspected, and the pipeline was then changed. The change was a data-correctness
   fix (fault alignment), applied identically to all three experiments, not
   hyperparameter tuning against test performance. Nonetheless, the corrected
   model's test score arrived after that inspection, so this is not a
   "never looked at test" claim — it is "test never selected the model".
3. `hybrid_results/final_test_metrics.json` was produced in a single run and has
   not been overwritten; its mtime (12:59) predates all audit activity.

## 9. Engineered features correct, no label or future information — PASS

**Evidence.** `features.engineer_features` (`features.py:56-71`) computes, per
window and channel, RMS (`sqrt(mean(x²))`), standard deviation, peak-to-peak
(`max-min`) and maximum absolute amplitude, from the window's own 256 samples
only. No temporal neighbour, no episode-level statistic, no label.

**Units.** Features are computed in physical units by inverting the saved
train-only z-score: `X = Z * std + mean` (`features.py:86`). `mean`/`std` are
train-derived and label-independent, so this is a re-scaling, not leakage. It was
measured to be equivalent to computing on z-scored data (0.9537 vs 0.9546 macro
F1, `check_feature_space.py`), so no information is manufactured by the transform.

**Ordering.** Feature columns are per-channel blocks of 4
(`ch{c}_{rms,std,ptp,maxabs}` → 192 columns), matching `N_ENG_FEATURES = 48*4`.

**Not a feature.** `sc_type` appears only as the target `y`; a grep of
`features.py` shows no label access.

## 10. CNN embeddings consistent between training and evaluation — PASS

**Evidence.**
* The embedding is defined once, as `Small1DCNN.feature_net` (three conv blocks →
  global average pool → 96-D); the classifier is a separate `Linear(96, 4)`
  (`train_hybrid.py` model class). Training optimises the whole network;
  `embed()` returns exactly `feature_net` output.
* `features.embeddings_batched` (`features.py:92-106`) calls `model.eval()` before
  inference and runs under `torch.no_grad()` — so BatchNorm uses running
  statistics, matching inference mode during validation.
* Experiment C loads the **best validation-loss checkpoint** saved by B
  (`train_hybrid.py:314`, `torch.save`) — the same weights that produced B's
  reported metrics, not a later or averaged state.

**Verified empirically.** Reloading `expB_cnn_best.pt` and re-predicting the test
split reproduces `final_test_metrics.json` **exactly** (accuracy, macro F1 and the
full 4×4 confusion matrix compared programmatically; the evaluation script aborts
on mismatch). Embedding dimensionality on the same windows: `(64, 96)`, all finite.

**Residual note.** Embeddings are extracted in a single pass over z-scored
windows; no augmentation or dropout is active at extraction time, so train-time
and eval-time representations differ only by the intended BatchNorm mode change.

## 11. RF and hybrid use the intended inputs — PASS

**Evidence.**
* A: `RandomForestClassifier(400, min_samples_leaf=2, class_weight="balanced",
  random_state=42)` fitted on the **192 engineered features** of the training
  subset (`train_hybrid.py:363, 367-370`).
* C: fitted on `concat([96-D CNN embedding, 192 engineered features])` = 288
  columns (`train_hybrid.py:417-425`).
* Both are fitted on the training subset only; validation is used for scoring;
  test only after selection.

**Verified from the artifacts.** Loading `expC_hybrid_rf_model.pkl` gives
`RandomForestClassifier, n_features=288, n_estimators=400`; `expA_…pkl` gives 192.

**Sanity of the fusion.** C (0.9989) sits between A (0.9608) and B (0.9992) and
slightly below B — consistent with a fused model whose 288-D input is dominated by
the learned embedding. It is *not* reported as an improvement over B.

## 12. Training subset documented: sampling, class and episode representation — PASS now (was NEEDS REVIEW)

**What the code does.** `train_hybrid.py:113-123` (`stratified_subset`):
proportional stratified sampling **per class** from the training windows with
`np.random.default_rng(42)`; `SUBSET_TARGET = 16384` (`train_hybrid.py:76`).
Selection is by **window**, not by episode.

**What was recorded before this audit.** Window count, seed and per-class window
counts (`comparison.json → protocol`). **Episode representation was not recorded**
— that was the gap.

**Now measured** (`subset_stats.json`, reproducible from the pickles):

| Quantity | Value |
|---|---|
| Subset size | 16,384 windows (15.5 % of 105,728) |
| Seed / method | 42, proportional stratified per class |
| Windows per class | LLL 4,085 · LL 4,066 · SLG 4,193 · LLG 4,040 |
| Distinct episodes covered | **5,328 of 6,314 (84.4 %)** |
| Episodes per class | LLL 1,341 · LL 1,324 · SLG 1,351 · LLG 1,312 |
| Windows per episode in subset | min 1, **median 3**, mean 3.08, max 11 |

**Interpretation.** Despite being window-stratified, the subset is well spread
across episodes: 84 % of training episodes contribute at least one window, median
3 windows each, no episode contributes more than 11 of 16,384. It is therefore not
a "few episodes, many windows" subset — but the guarantee is statistical, not
structural: an episode-stratified subset would guarantee uniform episode coverage
by construction. **Recommendation:** switch `stratified_subset` to sample episodes
first, then windows within them, and record `subset_episodes` in the metrics file.

---

## Additional observations (not part of the 12 checks)

1. **Windows within an episode are not independent.** Every headline metric is
   per-window. The episode-level companion evaluation (deliverable of Prompt 3)
   gives 0.9963 macro F1 over 1,354 episodes — lower than 0.9990 per window, and
   the more honest number for "classify a new episode".
2. **Single seed, single split.** Seed 42 throughout; no repeated runs, no
   cross-validation, no confidence intervals. Patch-level differences (B vs C:
   19 vs 25 validation errors) are not statistically meaningful.
3. **Early stopping never fired** in the final run: all 20 epochs executed, best
   validation loss 0.00387 at epoch 18, `early_stopped = false`
   (`expB_cnn_config.json`). Reported as such rather than implied as active
   regularisation.
4. **Legacy artifacts are clearly labelled.** `hybrid_results_legacy_windowing/`
   contains chance-level results (A 0.2812, B 0.2332, C 0.2634) with a `NOTE.md`
   explaining they are invalid. They are not mixed into the final results.
5. **No fabricated metrics.** Every number in `hybrid_results/` and
   `episode_evaluation/` traces to a JSON artifact written by a script in this
   repository; window-level metrics were independently re-derived during this
   audit and matched exactly.

## Corrective actions recommended (none applied during the audit)

| Priority | Action |
|---|---|
| High | Switch the training subset to **episode-stratified** sampling and persist `subset_episodes` in the metrics file. |
| High | Persist an explicit `episode_index` per window in preprocessing so episode-level evaluation needs no reconstruction. |
| Medium | Run ≥3 seeds and report mean ± spread for A/B/C. |
| Medium | Replace `technical_report.md`'s stale window counts (151,536 / 32,472 / 32,496) — they predate the windowing fix; see `FINAL_CONSISTENCY_AUDIT.md`. |
| Low | Record `early_stopped` and best epoch in `comparison.json` for completeness. |

## Bottom line

The pipeline as it now stands is methodologically sound on all twelve checks:
episode-level splitting with machine-checked disjointness, train-only
normalisation applied once, fault-aligned windows verified window-by-window,
validation-only selection, and a single post-selection test evaluation.
The two failures that existed earlier (silent `dt` fallback; self-referential
coverage) were real, were capable of producing plausible-looking garbage, and are
documented rather than hidden. The main scientific caveat is not leakage but
**scope**: 99.6–99.9 % on simulated PROTECT-90 episodes says nothing about
behaviour on measured grid signals.

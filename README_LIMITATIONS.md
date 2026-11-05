# Limitations, failed experiments and resource notes

This file is the honest ledger for the VIDYUT-KAVACHA CNN–RF hybrid work. Every
number quoted here is produced by a script in this repository; nothing is
estimated or inferred.

---

## 1. The windowing defect (the most important finding)

The pre-existing preprocessing placed windows almost entirely **outside** the
fault.

`preprocess.py` called `episode_to_windows(df_feat, ...)` where `df_feat` had
already had its `time_s` column removed. The function inferred the sample
interval `dt` from that same frame, found no time column, and therefore *always*
fell through to a plain stride sweep starting at sample 0. Its
fault-alignment branch never executed.

Because the fault starts near t = 0.43 s of a 1.0 s record and the sweep fills
its 24-window cap from sample 0, the windows covered only the first ~0.25 s. On
the 200 sampled episodes only **7.0 % of windows overlapped the fault**, with a
mean fault coverage of **0.035** (`window_signal_diagnostic.json`); for episode
`sid=0` exactly 1 of 24 windows overlapped it, at 2.08 % total coverage.

Consequence: all three models scored at chance level on the original windows.

| Metric on original windows | Value |
|---|---|
| RF on engineered features, test macro F1 | 0.2812 |
| Compact 1D CNN, test macro F1 | 0.2332 |
| Hybrid, test macro F1 | 0.2634 |
| 4-class chance level | 0.25 |

These archived results are kept unmodified in `hybrid_results_legacy_windowing/`
so the failure is auditable rather than quietly deleted.

**Fix.** Windows are now drawn from `[t_evnt_start, t_evnt_end]`, and
`episode_to_windows` now *raises* if fault alignment is requested without a
valid `dt`, so this class of silent fallback cannot recur. Window size (256),
stride (64), per-episode cap (24), episode splits, and train-only per-channel
z-score are all unchanged.

### Second-order failure worth recording

The first attempt at the fix still failed, because the coverage metric was
computed against the *same* `f0`/`f1` that the broken code had set to
`0`/`n`. It self-reported `mean fault coverage = 1.0000` while the windows were
still the legacy pre-fault ones. The regeneration produced 151,536 windows
(identical to the broken run) and a placebo RF still scored 0.2482.

It was caught by comparing the two samplers side by side on real episodes
(`debug_sampler`, since removed): the preprocess sampler returned 24 windows for
an episode whose fault lasted only 19.2 ms. Lesson applied: the coverage metric
is now derived from the returned window starts and a genuine fault interval, and
the loader refuses to fall back silently.

### A false lead, for the record

During diagnosis, engineered features appeared to "collapse" on z-scored data
(macro F1 0.263) while physical units looked fine (0.810). That conclusion was
wrong — both measurements were taken on the *broken* windows, and the two
numbers came from different window sets. On correctly aligned windows the two
feature spaces are equivalent:

| Feature space (8,000 train / 8,000 val windows, corrected windows) | Val macro F1 |
|---|---|
| z-scored windows | 0.9537 |
| physical units (de-normalised) | 0.9546 |

Physical units are used in the final model because the features are meant to be
amperes/volts, not because they are more accurate. Normalisation itself was
never the problem.

## 2. Subset usage

All three experiments train on **16,384 windows** drawn as a fixed-seed
(42) class-balanced stratified subset of the 105,728 available training
windows — 15.5 % of the training split, per class
`[4085, 4066, 4193, 4040]`. Validation (22,908) and test (22,519) use every
window.

Consequences:

* The engineered-feature RF (A) and the hybrid RF (C) could probably gain a
  little from the full training split. A is at 0.961 and plausibly still
  improving with data; this was **not** measured, and no claim is made about it.
* The CNN saturates well before the subset size, so the subset is not the
  binding constraint for B or C.
* The subset is reproducible and its size/seed are written into every metrics
  file.

## 3. Computational constraints

* CPU only (no CUDA). PyTorch 2.9.1 CPU, 16 cores, ~16 GB RAM.
* Preprocessing re-reads all 9,022 pickles and writes ~6.4 GB of binary windows:
  ~8 minutes.
* CNN training: 20 epochs × ~9 s ≈ 185 s.
* RF (A) fit: 33 s. RF (C) fit + embedding extraction: 9 s.
* Total wall clock for the full pipeline from a clean window store: ~4 minutes.
* Because of that budget, no hyperparameter search was run. RF settings
  (400 trees, `min_samples_leaf=2`, balanced class weights) and CNN settings
  (3 conv blocks, lr 1e-3) were fixed a priori and never tuned on validation
  beyond early stopping.
* Data are read with `np.memmap` and processed batch-wise throughout. Peak
  training RAM is dominated by the 16,384 × 256 × 48 subset (~0.8 GB per batch
  working set), not by the 5.2 GB training split.

## 4. Protocol limitations

* **The test set was evaluated once**, for the model selected on validation
  (B). A and C have no test numbers in the headline results, by design. The
  `--test-all` flag exists for anyone who wants to spend two more test passes;
  doing so weakens the "untouched test set" claim and is documented as a
  deviation.
* **Windows within an episode are not independent.** The headline metrics are
  per-window. Because 16–24 windows come from the same simulation episode, the
  effective sample size is closer to the number of episodes (1,354 test
  episodes), and per-window metrics are optimistic relative to
  "classify a new, unseen episode". An episode-level metric (majority vote or
  mean logit per episode) was not computed and is the single most useful
  follow-up.
* Selection among A/B/C used a single validation split with no repeats, so the
  B-vs-C gap (19 vs 25 validation errors) is not statistically meaningful.
* B's early stopping (patience 5) never fired; all 20 epochs ran and the best
  validation-loss checkpoint (epoch 18) was used. Reporting "early stopping
  configured but not triggered" rather than implying it saved the run.

## 5. Dataset limitations

* **Synthetic simulations.** PROTECT-90 episodes are EMT simulations over a
  fixed 90 kV double-circuit topology with randomised line lengths, loads, grid
  strengths and fault parameters. Near-perfect separability is expected when
  the fault type directly determines the phase-current pattern; it does **not**
  transfer to field data with instrument transformers, noise, CT saturation,
  evolving faults or unknown topology.
* **The class is a clean 4-way label.** Real protection also needs fault
  location, resistance, phase selection and detection of non-fault disturbances
  (switching, inrush). None of that is addressed.
* **A 40 ms window inside the fault is an easy decision.** The model is
  evaluated on windows that are 96 % inside the fault interval by construction.
  Detection latency, fault inception detection and pre-fault-only operation are
  out of scope, and the reported accuracy says nothing about them.
* **Short faults.** 828 of 9,022 episodes have a fault shorter than one 40 ms
  window; those episodes yield only 1–3 onset-centred windows with lower fault
  coverage, and they are under-represented per episode even though their class
  balance is preserved.
* **Balanced classes.** The dataset is nearly balanced (2,219–2,279 episodes
  per class). No rare-class or open-set behaviour was tested.
* **One corrupt episode** (`7154`) fails to unpickle and is excluded; it is the
  only excluded sample and is listed in
  `preprocessed/window_shape_integrity_report.txt`.
* **Single seed.** Every result comes from seed 42. No variance across seeds,
  folds or data orderings is reported, so no confidence intervals accompany the
  numbers.

## 6. What was deliberately not done

* No data augmentation, no class re-weighting beyond `balanced`, no test-time
  augmentation.
* No architecture search, no transfer learning, no ensembling of A/B/C.
* No re-splitting or re-balancing of the episode splits — they are reused
  byte-identically.
* No feature-selection study; all 192 engineered features and all 96 embedding
  dimensions are used.

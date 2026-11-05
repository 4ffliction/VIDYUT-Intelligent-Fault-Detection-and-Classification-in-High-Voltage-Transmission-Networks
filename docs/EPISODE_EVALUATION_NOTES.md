# Episode-level evaluation — method and results

Adds an episode-level view of the selected model's test performance, alongside the
existing window-level numbers. **No model was retrained. The checkpoint and every
existing result file were opened read-only.**

Script: `evaluate_episode_level.py`
Outputs: `episode_evaluation/`

---

## 1. The missing artifact, stated plainly

`preprocessed/X_test.x.bin` stores **only** the label per window. No episode ID
was persisted per window, and no window→episode index file exists anywhere in the
project:

```
preprocessed/
  X_test.x.bin  X_test.y.bin  test_episodes.csv
  X_train..., X_val..., norm_mean.npy, norm_std.npy, *_report.txt
```

So the mapping had to be **reconstructed**, not read. What makes reconstruction
valid rather than invented:

1. `preprocess.build_split()` iterates the split's episode list in CSV order and
   appends each episode's windows contiguously, so blocks are ordered and
   contiguous — window `i` belongs to exactly one episode.
2. The number of windows an episode produces is a deterministic function of its
   fault interval: `episode_to_windows(t_evnt_start, t_evnt_end, dt)`.

## 2. How the reconstruction was verified (not assumed)

The script re-opens every test episode's raw PKL, re-runs the *same* windowing
function, z-scores the result with the saved train statistics, and compares it
byte-for-byte with the stored block at the hypothesised offset. Any mismatch
aborts the run.

```
verified 1354 episodes, 22519 windows, max|diff| = 3.814697265625e-06
episodes with 0 windows = 0
```

The residual `3.8e-06` is float32 rounding in the z-score (`(x-μ)/σ`), not a
misalignment; the tolerance is `1e-5`. Offsets sum to exactly 22,519 = the split
size, and no episode was skipped or dropped.

Additionally, recomputing window-level metrics from the loaded checkpoint
reproduces `hybrid_results/final_test_metrics.json` **exactly** (accuracy,
macro F1 and the full confusion matrix compared programmatically; the script
aborts otherwise). This proves the checkpoint, the stored windows and the
episode mapping are mutually consistent.

Nothing here uses test labels to alter the model or its predictions — labels are
read only after all predictions are made, for scoring.

## 3. Aggregation

**Primary — majority vote.** Each window casts one vote for its predicted class;
the episode takes the class with the most votes.

*Tie-breaking rule:* if two or more classes tie on votes, the tied class with the
higher **mean predicted probability** across the episode's windows wins; if still
tied, the lowest class index. **Ties encountered: 0** (with 2–24 windows per
episode, a 4-way tie never occurred).

**Secondary — mean probability.** Arithmetically average the per-window softmax
vectors and take the argmax. This uses confidence rather than counts.

Both aggregations produced **identical** predictions on this test set, so the
choice does not drive the result.

## 4. Results

Test episodes evaluated: **1,354** (all of them; 0 with zero windows).
Class distribution: LLL 341, LL 338, SLG 342, LLG 333.
Windows per episode: min 2, median 20, max 24.

| Level | n | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|---|
| Window-level | 22,519 | 0.9990 | 0.9990 | 0.9990 | 0.9990 |
| **Episode-level (majority vote)** | **1,354** | **0.9963** | **0.9964** | **0.9963** | **0.9963** |
| Episode-level (mean probability) | 1,354 | 0.9963 | 0.9964 | 0.9963 | 0.9963 |

Per-class F1:

| Level | LLL | LL | SLG | LLG |
|---|---|---|---|---|
| Window | 0.9991 | 0.9995 | 0.9989 | 0.9985 |
| Episode (majority) | 0.9985 | 0.9985 | 0.9942 | 0.9940 |

Episode-level confusion matrix (majority vote):

| | LLL | LL | SLG | LLG |
|---|---|---|---|---|
| **LLL** | 340 | 0 | 1 | 0 |
| **LL** | 0 | 338 | 0 | 0 |
| **SLG** | 0 | 0 | 342 | 0 |
| **LLG** | 0 | 1 | 3 | 329 |

Window-level confusion matrix (reproduced from the stored results):

| | LLL | LL | SLG | LLG |
|---|---|---|---|---|
| **LLL** | 5795 | 0 | 2 | 1 |
| **LL** | 2 | 5497 | 0 | 0 |
| **SLG** | 2 | 0 | 5797 | 0 |
| **LLG** | 3 | 3 | 9 | 5408 |

Every episode-level error lies inside the two confusable two-phase classes:
three LLG episodes were called SLG, one LLG was called LL, and one LLL was called
SLG. No SLG episode was misclassified.

## 5. Interpretation — and why the drop is the honest number

Episode accuracy is **0.35 percentage points lower** than window accuracy
(0.9963 vs 0.9990). That is the expected direction: a window that is 96 % inside
the fault is an easier decision than "what fault is this episode?", and an episode
is wrong if it has any systematically misvoted windows.

Put differently: **5 misclassified episodes out of 1,354**, versus **22
misclassified windows out of 22,519**. Quoting only the window figure overstates
how reliably the pipeline labels a *new* simulation episode, which is the
quantity a protection engineer cares about.

Both figures remain **simulation-only** performance on a synthetic dataset with a
fixed topology. Neither supports any claim about field data, measurement
transformers, noise, evolving faults, or protection-relay readiness.

## 6. Reproduce

```bash
python evaluate_episode_level.py
```

Runtime ≈ 5 min on CPU (dominated by re-reading 1,354 PKLs for verification).
Outputs, all new files — nothing overwritten:

| File | Contents |
|---|---|
| `episode_evaluation/episode_level_metrics.json` | window + both episode-level metric blocks, protocol, counts, verification residuals |
| `episode_evaluation/episode_level_classification_report.csv` | per-class precision/recall/F1/support for all three levels |
| `episode_evaluation/episode_level_confusion_matrix.png` | episode-level and window-level confusion matrices side by side |
| `episode_evaluation/episode_predictions.csv` | per-episode: id, true class, window count, vote counts, majority, mean-prob class, vote fraction |

`episode_predictions.csv` is included so a reviewer can inspect every individual
episode — including the five errors — without re-running anything.

## 7. Limitations of this evaluation

* The episode→window mapping is **reconstructed**, not stored. It is verified
  window-by-window here, but a future preprocessing run should persist an
  explicit `episode_index` column so this step becomes a lookup instead of a
  re-derivation from 1,354 pickles.
* Majority vote weights every window equally; episodes have 2–24 windows, so an
  episode with 24 windows is not more "confident" than one with 2. The
  mean-probability variant does not fix this either. A confidence-weighted vote
  was not attempted.
* 828 episodes in the full dataset have faults shorter than one 40 ms window;
  those contribute only 2–3 onset-centred windows, so their episode-level
  prediction rests on very little evidence. This is reported, not corrected —
  excluding them would flatter the numbers.
* One test split only; no cross-validation, no repeated seeds.

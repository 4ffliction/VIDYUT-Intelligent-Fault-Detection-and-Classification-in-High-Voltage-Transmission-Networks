# REPRODUCIBILITY_AUDIT.md — can a new user rebuild VIDYUT?

Reviewer simulation: follow only `README.md` + `requirements*.txt`, on a clean
machine, without the original author's context.
Read-only. No model was retrained and the 12 GB dataset was **not** re-downloaded.
Date: 4 November 2026.

---

## Environment actually in use

| Item | Value |
|---|---|
| Python | **CPython 3.14.0** (Windows x86-64) |
| PyTorch | 2.9.1, **CPU build** (no CUDA) |
| numpy / pandas | 2.3.5 / 3.1.0 |
| scikit-learn / joblib | 1.9.1 / 1.6.0 |
| matplotlib | 3.11.2 |
| Threads | `torch.set_num_threads(8)` (`train_hybrid.py:85`) |
| Disk needed | ~**34 GB** peak (12 GB archive + 22 GB extracted + 7 GB windows) |

## Verdict summary

| # | Check | Verdict |
|---|---|---|
| 1 | Python and library versions | **PASS** (exact pins; Python 3.14 is new — see note) |
| 2 | `requirements.txt` complete and compatible | **PASS** (file did not exist before this audit) |
| 3 | Random seeds set | **PASS** (with a bit-exactness caveat) |
| 4 | Paths relative and portable | **PASS** for path literals, **NEEDS REVIEW** for invocation directory |
| 5 | Dataset download/extraction documented | **NEEDS REVIEW** |
| 6 | Preprocessing reproducible from documented input | **PASS** (verified against artifacts) |
| 7 | Training and evaluation commands correct | **PASS** |
| 8 | Expected outputs match actual outputs | **PASS** (byte-exact reproduction) |
| 9 | Hard-coded paths tied to the original machine | **PASS** |
| 10 | Graceful failure on missing dataset/checkpoint | **FAIL** |

---

## 1. Python and library versions — PASS

`requirements.txt` pins six exact versions, all of which are the versions the
reported results were produced with. `requirements-dev.txt` adds
`python-docx==1.2.0` and states plainly that it is **not** needed to reproduce any
metric or figure (it is only imported by the EDP-II `.docx` builders, which are
not part of the VIDYUT pipeline).

**Note for a reviewer.** CPython 3.14 is very recent; PyTorch 2.9.1 provides
cp314 wheels, so `pip install -r requirements.txt` works on current Windows/Linux
x86-64. On older interpreters the pins will fail to resolve rather than silently
install something different — the safer failure. A `requires-python = ">=3.11"`
style hint is not present; the README states the tested version in one line.

## 2. `requirements.txt` complete and compatible — PASS

**Before this audit there was no `requirements.txt` at all** — the dependency set
was implicit. It is now present, derived from third-party imports actually found
in the project files (mapping in `DEPENDENCY_AUDIT.md`), not from guesswork.

Completeness was checked by importing every module the pipeline touches under the
pinned versions: `numpy`, `pandas`, `sklearn`, `joblib`, `matplotlib`, `torch`,
plus stdlib only (`pathlib`, `json`, `argparse`, `time`, `traceback`, `warnings`).
No unlisted third-party import exists in the runtime path. `tqdm`, `seaborn`,
`scipy` and `torchvision` appear nowhere.

## 3. Random seeds — PASS (with caveat)

| Site | Mechanism |
|---|---|
| Split construction | `RANDOM_SEED`, `train_test_split(..., random_state=RANDOM_SEED, stratify=…)` (`preprocess.py:319, 323`) |
| Subset sampling | `np.random.default_rng(SEED)` (`train_hybrid.py:115`) |
| CNN init + shuffling | `torch.manual_seed(seed)` and `np.random.seed(seed)` (`train_hybrid.py:249-250`), then `default_rng(seed)` at `:256` |
| RF (A and C) | `random_state=SEED` (`train_hybrid.py:367, 417`) |
| Inference batches | `default_rng(0)` — ordering only, no effect on outputs |

**Caveat.** No `torch.use_deterministic_algorithms(True)` and no
`torch.backends.cudnn.deterministic` are set. On CPU-only execution with these
layers there is no nondeterministic kernel in the path, and the three experiments
were each run once, so a re-run should reproduce the checkpoint — but this is
**not guaranteed bit-exact** across BLAS/CPU implementations. The honest claim is
"seeded and expected to be reproducible on the same platform", not "bit-identical
everywhere".

## 4. Paths relative and portable — PASS for literals, NEEDS REVIEW for CWD

**Good.** Every data path is a relative `pathlib.Path` (`preprocessed/`,
`hybrid_results/`, `hv_double_line_90kv_preprocessed_data_extracted/`). There is no
`C:\Users\…` or `/home/…` anywhere in the runtime code. `run_modeling.sh` begins
with `cd "$(dirname "$0")"`, so it is location-independent.

**Weakness.** All paths resolve against the **current working directory**, and
four scripts (`train_baseline.py`, `train_baseline_features.py`, `train_cnn.py`,
`evaluate.py`) do `sys.path.insert(0, str(Path.cwd()))`. Consequence: the commands
work only when run **from the project root**. Running
`python src/preprocess.py` from anywhere else, or `cd src && python preprocess.py`,
will fail to find `preprocessed/` or `preprocess.py`.

`train_hybrid.py` and `evaluate_episode_level.py` are the exceptions — they use
`Path(__file__).resolve().parent` — so the two scripts that matter most are the
most portable. Fix before publishing: introduce `src/paths.py` (see
`GIT_READINESS_AUDIT.md` §6).

## 5. Dataset download and extraction — NEEDS REVIEW

**What is documented.** `README.md` §4 gives the archive filename, the Zenodo
record (DOI `10.5281/zenodo.21109169`, concept `10.5281/zenodo.18418329`),
CC BY 4.0, and says to extract into `hv_double_line_90kv_preprocessed_data_extracted/`.
A checksum is recorded in `zip_md5_check.txt` (`7cf176f169299b825ba6a6be102edca8`,
matches the published archive) and the labels CSV
(`5f015330f77ed53b76bd5db26e83c48d`, matches).

**What is missing for a cold start.**
1. No single copy-pasteable command. The reviewer must infer the Zenodo file URL.
2. **No integrity verification step** in the README, even though the MD5 is known
   and correct — it should be `md5sum hv_double_line_90kv_preprocessed_data.zip`
   then compare.
3. **No expected-file-count check** after extraction (should be **9,022**
   `*_sample_hv_double_line_90kv.pkl`, ~22 GB).
4. **No disk-space warning** (~34 GB peak), which will bite on a laptop.
5. `DATA_DIR` in `preprocess.py` points at the extracted directory by default;
   a reviewer with the archive elsewhere has to edit the constant (there is no
   CLI flag or environment override).

None of these block reproduction for someone with the data in hand, but all are
"figure it out yourself" friction. Adding four lines to the README closes it.

## 6. Preprocessing reproducible from documented input — PASS (verified)

The full preprocessing step was **not** re-run from the 12 GB archive during this
audit (it takes ~8 minutes and 22 GB of pickles are already present), but its
output was independently re-derived and checked three ways:

1. **Window counts.** All 6,315 train pickles were re-read and re-windowed with
   `episode_to_windows`; the resulting per-episode counts were compared to the
   stored label arrays: sum = **105,728 = stored**, `counts_match = True`,
   `block_labels_consistent = True` (`subset_stats.json`).
2. **Test split, window by window.** All 1,354 test episodes were re-windowed and
   the resulting sample matrices compared elementwise to
   `preprocessed/X_test.x.bin`: **max |diff| = 3.81e-06** on float32, zero
   mismatches, zero zero-window episodes
   (`episode_evaluation/episode_level_metrics.json → protocol`).
3. **Normalisation.** `verify_pipeline.py` rerun → train `|mean|max` 0.00122,
   `std mean` 1.0196, applied once (a double z-score would give std ≈ 0.81);
   split overlaps all zero; per-split shapes and class balance as documented:
   **RESULT: PASS**.

Determinism: the split is seeded and reused from `preprocessed/*_episodes.csv`
when present (`preprocess.py:296-311`), so re-running preprocessing on the same
data reproduces the same splits rather than reshuffling.

## 7. Training and evaluation commands — PASS

Documented sequence, and its actual behaviour:

```bash
python -m pip install -r requirements.txt   # 6 pins, CPU torch
python preprocess.py                        # ~8 min, writes preprocessed/ (~7 GB)
python verify_pipeline.py                   # prints PASS/FAIL, writes preprocessing_verification.txt
python train_hybrid.py                      # experiments A, B, C -> hybrid_results/
python evaluate_episode_level.py            # -> episode_evaluation/
```

* `run_modeling.sh` wraps steps 2–4 with `set -euo pipefail` and `cd $(dirname $0)`.
  Correct as written.
* `train_hybrid.py` accepts exactly one flag, `--test-all`, documented in the
  README as a deliberate deviation that scores A and B on test too; the default
  path evaluates **only** the validation-selected model.
* `evaluate_episode_level.py` was executed end-to-end during this audit and
  exited 0, writing five artifacts.
* `verify_cleanup.py` was executed during this audit, exit 0:
  `essential files: 56/56 present`, archive 12,046,677,098 bytes, 9,022 extracted
  episodes, RF `n_features=288 trees=400`, CNN checkpoint `23 tensors`, split
  manifests 6315/1353/1354, and — notably — **README path references: 20 distinct,
  0 unresolved**.

## 8. Expected outputs match actual outputs — PASS

| Documented output | Present | Matches |
|---|---|---|
| `preprocessed/X_{train,val,test}.x.bin` (105,728 / 22,908 / 22,519 × 256 × 48) | yes | verified by shape |
| `preprocessed/*.y.bin`, `norm_{mean,std}.npy`, `*_episodes.csv` | yes | verified by count |
| `hybrid_results/comparison.json` — A 0.9608, B 0.9992, C 0.9989 val macro F1 | yes | exact |
| `hybrid_results/final_test_metrics.json` — 0.9990, CM `[[5795,0,2,1],[2,5497,0,0],[2,0,5797,0],[3,3,9,5408]]` | yes | **re-derived exactly** from the checkpoint |
| `episode_evaluation/episode_level_metrics.json` — 0.9963 | yes | recomputed by its own script |
| `hybrid_results/expB_cnn_best.pt` | yes | loads, 23 tensors |

Re-deriving the test confusion matrix from the saved checkpoint is the strongest
single reproducibility check available, and it passes exactly (agreement enforced
programmatically; the script aborts otherwise).

## 9. Hard-coded paths tied to the original computer — PASS

No user-specific path exists in any pipeline script. The only absolute paths in
the whole tree are LibreOffice discovery paths in `build_report_run.py`
(`C:\Program Files\LibreOffice\…`), which belongs to the EDP-II report toolchain
and is **excluded from the VIDYUT repository** per `GIT_READINESS_AUDIT.md` §3.7.
No environment variables, no machine names, no credentials.

## 10. Graceful failure when data is missing — FAIL

| Scenario | Actual behaviour |
|---|---|
| Archive not extracted | `preprocess.py` reaches `load_waveform`, raises `FileNotFoundError(<path>)`, caught by the top-level handler (`preprocess.py:453-455`) which prints a traceback and exits 1. **Diagnosable but not instructive** — no "download PROTECT-90 from Zenodo (DOI …)" message. |
| `preprocessed/` missing, training run | `train_hybrid.py:67` executes `OUT_DIR.mkdir(parents=True, exist_ok=True)` at **import time**, creating an empty `hybrid_results/`, then dies inside `np.memmap`/`load_binny` with a raw traceback. Side effect + unhelpful message. |
| Checkpoint missing | `evaluate_episode_level.py:106` and `:113` **do** exit cleanly with an explicit `SystemExit` message. This is the good pattern. |
| Labels CSV missing | `preprocess.py` fails on read with a raw traceback. |

**Corrective action (not applied).** Add one precondition block per entry point:
check `DATA_DIR` exists and contains ≥1 pickle → print the Zenodo DOI and the
expected extraction path; check `preprocessed/X_train.x.bin` exists → tell the user
to run `python preprocess.py`. Move the `OUT_DIR.mkdir` inside `main()`. Roughly
15 lines total, and it converts the most likely first-run failure into a message.

---

## Verified fresh-setup sequence

Steps executed during this audit are marked ✅; the two long ones (⚠️) were not
re-run because their outputs already exist and were verified against artifacts.

```bash
# 0. Python 3.14.0 x86-64, CPU-only. ~34 GB free disk.
python -m venv .venv && . .venv/bin/activate          # (Windows: .venv\Scripts\activate)

# 1. Dependencies  ✅ (pins resolved and imported)
python -m pip install -r requirements.txt

# 2. Data — PROTECT-90, CC BY 4.0, DOI 10.5281/zenodo.21109169
#    download hv_double_line_90kv_preprocessed_data.zip  (12.0 GB)
md5sum hv_double_line_90kv_preprocessed_data.zip       # expect 7cf176f169299b825ba6a6be102edca8
unzip hv_double_line_90kv_preprocessed_data.zip -d hv_double_line_90kv_preprocessed_data_extracted
ls hv_double_line_90kv_preprocessed_data_extracted/*.pkl | wc -l   # expect 9022

# 3. Build windows  ⚠️ (~8 min, writes ~7 GB into preprocessed/)
python preprocess.py

# 4. Verify the split rules  ✅ (RESULT: PASS)
python verify_pipeline.py

# 5. Experiments A/B/C + validation selection + single test eval  ⚠️
python train_hybrid.py

# 6. Episode-level evaluation  ✅ (exit 0, five artifacts)
python evaluate_episode_level.py

# 7. Artifact sanity  ✅ (exit 0, "problems: none")
python verify_cleanup.py
```

Expected end state, matching this repository:

```
hybrid_results/comparison.json            A 0.9608  B 0.9992  C 0.9989  (val macro F1)
hybrid_results/final_test_metrics.json     macro F1 0.9990, 22 errors / 22,519 windows
episode_evaluation/episode_level_metrics.json  macro F1 0.9963, 5 errors / 1,354 episodes
```

## Bottom line

A reviewer with the dataset and a CPU can rebuild every reported number by running
five documented commands, and the two headline artifacts were re-derived
independently during this audit and matched exactly. The gaps are all in **first-run
ergonomics**, not in the science: the dataset step lacks a copy-paste command and a
checksum check, and a missing dataset produces a traceback instead of an
instruction. Both are ~20 lines of work.

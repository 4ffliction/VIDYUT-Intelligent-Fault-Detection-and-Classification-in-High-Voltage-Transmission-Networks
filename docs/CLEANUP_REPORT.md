# CLEANUP_REPORT.md — VIDYUT-KAVACHA final safe cleanup

Date: 4 November 2026
Scope: `C:\Users\sharm\Downloads\Study_Material_Sem3`
Policy applied: delete only what is demonstrably regenerable or valueless;
preserve everything referenced, protected, or uncertain.

---

## 1. Summary

| Metric | Value |
|---|---|
| Files inspected (project, excluding the 9,022 extracted dataset pickles) | **264** |
| Files inspected (including extracted dataset) | 9,286 |
| Files deleted | **24** |
| Files moved to `cleanup_review/` | **1** |
| Files retained (239 in place + 1 relocated + 3 new) | **243** |
| Total storage recovered | **5,755,335 bytes (5.49 MiB)** |
| Arithmetic | 264 inspected − 24 deleted + 3 new = 243 (the moved file is retained, not removed) |
| Total project size before → after | 42,832,622,076 B → 42,826,866,741 B (39.89 GB) |
| Protected dataset data touched | **none** |

The directory is 39.89 GB, of which 39.0 GB is protected dataset data
(12.0 GB source archive + 22.0 GB extracted episodes + 7.0 GB preprocessed
windows). The cleanup therefore could not and should not have been large: the
recoverable volume among non-dataset files was small, and almost every
candidate turned out to be referenced by a retained document or to be the only
copy of a result.

## 2. Method

1. **Inventory.** Recursive listing of every file with exact byte size, plus
   directory-level `du` totals, empty-directory detection, and junk-pattern
   scan (`*.tmp`, `*.bak`, `*.swp`, `.DS_Store`, `Thumbs.db`) — no junk matched.
2. **Classification** into A (essential) / B (reproducibility) / C (safe to
   delete) / D (protected data).
3. **Dependency check.** `grep` over all `*.py`, `*.sh`, `*.md`, `*.json`,
   `*.txt` for each candidate's filename or path; MD5 hashing wherever
   duplicate content was suspected; archive validity testing with
   `zipfile` for the partial downloads.
4. **Execution** limited to candidates that passed all four deletion gates.
5. **Verification** via two independent scripts (§5).

## 3. Files deleted (24) and reasons

### 3.1 Regenerable bytecode cache (15 files, ~356 KB)

`__pycache__/*.cpython-314.pyc` for `build_report_body_part1`, `build_report_exp1`,
`build_report_exp2`, `build_report_exp3`, `build_report_helpers`,
`check_window_signal`, `evaluate`, `features`, `inspect_dataset`, `preprocess`,
`train_baseline`, `train_baseline_features`, `train_cnn`, `train_hybrid`,
`waveform_visualization`.

Reason: CPython 3.14 bytecode cache, recreated automatically on next import.
Contains no source, no results, no state. The now-empty `__pycache__/`
directory was removed too. Verification afterwards was run with
`PYTHONDONTWRITEBYTECODE=1` so the cache was not recreated.

### 3.2 Corrupt partial download scraps (2 files, 5,430,558 B)

| File | Size | Reason |
|---|---|---|
| `hv_partial_4MB.zip` | 4,194,304 B | `zipfile` raises `BadZipFile: File is not a zip file`. Truncated fragment from the dataset acquisition, superseded by the complete 12.0 GB archive. Zero references in any script or document. |
| `sample0_partial.zip` | 1,236,254 B | Same: `BadZipFile`, truncated fragment, superseded, zero references. |

These are **not** dataset archives — they are unusable byte fragments. The real
archives (`hv_double_line_90kv_preprocessed_data.zip` and the extracted tree)
were untouched; both were re-verified in §5.

### 3.3 Zero-byte files (7 files, 0 B)

`cnn_run.log`, `cnn_run_err.log`, `baseline_run.log`, `baseline_run_err.log`,
`hybrid_run_err.log`, `preprocessed_run_err.log`, `verify_err3.txt`.

Reason: exactly 0 bytes — they carry no information. The non-empty run logs
(`preprocessed_run.log`, `hybrid_run.log`, `download.log`) were **kept** as run
provenance. Note that `hybrid_run_err.log` and `preprocessed_run_err.log` were
recreated during the successful final run and were empty, which is itself
evidence the final pipeline produced no stderr output.

## 4. Files moved to `cleanup_review/` (1)

| File | Original path | Why moved rather than deleted |
|---|---|---|
| `scripts_run_modeling.stale.sh` | `scripts/run_modeling.sh` | Stale duplicate of the pre-hybrid runner. It `cd`s into `scripts/` and then calls `train_baseline.py` / `train_cnn.py` / `evaluate.py`, which do not exist in that directory, so it cannot run. Nothing references it, and the active runner `run_modeling.sh` in the project root now drives `preprocess → verify → train_hybrid`. Moved rather than deleted because two same-named runners is exactly the kind of ambiguity a reviewer should not have to guess about; the restore command is recorded in `cleanup_review/README.md`. The empty `scripts/` directory was then removed. |

## 5. Verification results

Ran after cleanup, with `PYTHONDONTWRITEBYTECODE=1`.

**`verify_cleanup.py` — 55/56 essential files present** (the 56th is this report,
which did not exist when the script ran):

```
1. essential files: 55/56 present
   preprocessed/X_train.x.bin: n=105728 win=256 ch=48
   preprocessed/X_val.x.bin:   n=22908  win=256 ch=48
   preprocessed/X_test.x.bin:  n=22519  win=256 ch=48
2. dataset archive: 12046677098 bytes ; extracted episodes: 9022
3. RF model loads: RandomForestClassifier, n_features=288, trees=400
4. CNN checkpoint loads: 23 tensors, 37703 elements
5. README path refs checked: 19 distinct, unresolved: 0
6. split manifests: row counts match (6315/1353/1354)
```

**`verify_pipeline.py` — RESULT: PASS**

```
overlaps: {'train/val': 0, 'train/test': 0, 'val/test': 0}
train: n=105728 labels=[26358, 26236, 27061, 26073]
val:   n=22908  labels=[5760, 5764, 5804, 5580]
test:  n=22519  labels=[5798, 5499, 5799, 5423]
NaN in train sample: False
train mean |max|=0.00122  std min=0.9719 mean=1.0196 max=1.0435
train windows are unit-variance and zero-mean: True
```

Checklist from the brief:

| Check | Result |
|---|---|
| Essential files still exist | PASS (55/56 + this report) |
| Final models and result artifacts intact | PASS — RF (288 features, 400 trees) and CNN checkpoint both load; all 20 result files in `hybrid_results/` present |
| README / experiment scripts reference valid paths | PASS — 19 distinct file references, 0 unresolved |
| Lightweight import / path checks | PASS — `features`, `preprocess`, `train_hybrid`, `verify_pipeline`, `verify_cleanup` all import and run |
| Dataset and split manifests unchanged | PASS — archive 12,046,677,098 B (unchanged), 9,022 extracted pickles, split rows 6315/1353/1354, zero episode overlap, window arrays unchanged |
| No final deliverable accidentally removed | PASS — see §6 |
| Cleanup report generated | this file |

## 6. Retained files (essential and reproducibility)

**Final pipeline (reproducibility, category B):**
`preprocess.py`, `features.py`, `train_hybrid.py`, `verify_pipeline.py`,
`verify_cleanup.py`, `check_window_signal.py`, `check_feature_space.py`,
`verify_storage.py`, `run_modeling.sh`, `preprocessing_config.json`.

**Final results (category A):** `hybrid_results/` (20 files, 31.4 MB) —
`exp{A,B,C}_val_metrics.json`, `exp{A,B,C}_val_classification_report.txt`,
`exp{A,B,C}_val_confusion_matrix.png`, `final_test_metrics.json`,
`final_test_classification_report.txt`, `final_test_confusion_matrix.png`,
`cnn_training_history.json`, `cnn_training_history.png`, `expB_cnn_config.json`,
`expB_cnn_best.pt`, `expA_rf_features_model.pkl`, `expC_hybrid_rf_model.pkl`,
`comparison.json`, `comparison.md`.

**Preprocessed data and reports (A/B/D):** `preprocessed/` (7.0 GB) — six
binary window splits, `norm_mean.npy`, `norm_std.npy`, three episode split
manifests, `class_distribution_report.txt`,
`window_shape_integrity_report.txt`. Plus `preprocessing_verification.txt`,
`window_signal_diagnostic.json`, `check_window_signal.py`, `technical_report.md`.

**Dataset (D):** `hv_double_line_90kv_labels.csv`,
`hv_double_line_90kv_preprocessed_data.zip` (12.0 GB),
`hv_double_line_90kv_preprocessed_data_extracted/` (22.0 GB, 9,022 pickles),
`metadata_audit_full.csv`, `zenodo_*.json`, `companion_repo_README.md`,
`gh_README.md`.

**Documentation (A):** `README.md`, `README_LIMITATIONS.md`,
`UTA024_Practical_Activity_Report.docx`,
`Thapar_Edp2_UTA024_Practical_Activity_Report.docx`, `dataset_audit.md`,
`preprocessing_investigation.md`, `preprocessing_validation_report.md`,
`training_readiness_report.md`, `full_dataset_validation.md`,
`dataset_readiness_report.md`, `changelog_task1_corrections.md`,
`recovery_log.md`, `report_verification.txt`, and the `build_report_*.py` /
`generate_report.py` builders that produce the `.docx` deliverables.

**Audit trail (deliberately kept):** `hybrid_results_legacy_windowing/`
(342 MB, 16 files) — the chance-level results from the defective windowing,
referenced by `README.md` and `README_LIMITATIONS.md`. `baseline/` (252 MB) —
retired-pipeline baseline, referenced by `technical_report.md`. `pkl_out/`
(17 MB, 7 episode pickles) — referenced by `dataset_audit.md`,
`inspect_dataset.py` and `metadata_audit_summary.json`. `plots/` (7.3 MB, 8 EDA
figures) — referenced by `preprocessing_investigation.md`.

## 7. Deliberately *not* deleted, and why

| Candidate | Size | Decision |
|---|---|---|
| `baseline/rf_model.pkl` | 252,047,161 B | Chance-level model from the retired (defective-window) pipeline. Kept because `technical_report.md` explicitly references `baseline/rf_model.pkl` as its saved model, and the report is a retained deliverable. Removing it would break a live reference. |
| `hybrid_results_legacy_windowing/*.pkl` | 341 MB | Audit evidence for the windowing defect; cited in `README_LIMITATIONS.md` §1 and `hybrid_results_legacy_windowing/NOTE.md`. |
| `metadata_audit_full.csv` | 7,327,363 B | MD5 `5f015330f77ed53b76bd5db26e83c48d` — byte-identical to `hv_double_line_90kv_labels.csv`. Kept: `dataset_audit.md` documents it as the audit copy. |
| `.cloud-attachments/c1f35ae…bin` | 7,327,363 B | Same MD5: a chat-uploaded copy of the labels CSV. Client-managed attachment store — not the agent's to delete. |
| `.cloud-attachments/*.json` (2) | 772 KB | Imported conversation records, client-managed. |
| `pkl_out/`, `plots/`, `pkl_audit_output.json`, `waveform_*.json/csv` | ~25 MB | All referenced by retained audit documents. |
| Old-pipeline scripts (`train_baseline.py`, `train_cnn.py`, `train_baseline_features.py`, `evaluate.py`) | 39 KB | Referenced by `technical_report.md`; document the retired approach. Trivial size. |
| `download.log` | 1.8 MB | Dataset acquisition provenance; non-empty. |
| Unrelated directories: `ARKA_DECK`, `DEADPOOL_MEASUREMENT_HANDBOOK`, `Ele`, `OMNITRIX_TD`, `OPERATION20`, `TD_5x5_Handbook`, `WAYNE_TD_DOSSIER`, `src` | ~19 MB | **Not part of VIDYUT-KAVACHA.** Separate coursework/decks. Out of scope; untouched. |

## 8. Unresolved concerns (not caused by this cleanup)

1. **`technical_report.md` references artifacts that do not exist.**
   It cites `cnn/best_model.pt`, `cnn/training_history.json` and
   `evaluation/evaluation_report.json`. `cnn/` is empty and there is no
   `evaluation/` directory. This is a pre-existing gap from the retired
   pipeline (its run log was 0 bytes, so it never completed). The file already
   carries a "SUPERSEDED IN PART" banner pointing to `README.md`. Left as-is
   because a cleanup must not rewrite reports; the empty `cnn/` directory was
   also left in place rather than silently removed, so the gap stays visible.
2. **~590 MB could be reclaimed later** by dropping
   `baseline/rf_model.pkl` (252 MB) and the two legacy RF pickles (341 MB), if
   the academic audit trail is ever judged unnecessary. Not done — both are
   cited by retained documents.
3. **Environment pinning — since resolved.** At cleanup time only `README.md`
   §6 described the environment. `requirements.txt` and `requirements-dev.txt`
   were added afterwards from an import audit, with the six direct dependencies
   pinned and verified in a clean virtual environment; see
   `DEPENDENCY_AUDIT.md`. Transitive versions remain unlocked (documented in
   that audit, §8).
4. `cleanup_review/` holds one deliberately-quarantined file; it can be deleted
   outright once the reviewer agrees the stale runner is not wanted.

## 9. Files added by this cleanup

| File | Purpose |
|---|---|
| `CLEANUP_REPORT.md` | This report. |
| `cleanup_review/README.md` | Explains the quarantined file and how to restore it. |
| `verify_cleanup.py` | Re-runnable post-cleanup verification (essentials present, models load, README paths resolve, split row counts). |

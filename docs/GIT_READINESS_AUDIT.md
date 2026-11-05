# GIT_READINESS_AUDIT.md — VIDYUT repository architecture

Read-only audit. **Nothing was moved, renamed, deleted or overwritten.**
Date: 4 November 2026.

---

## 1. Current state

| | |
|---|---|
| Git repository | **None** — no `.git` directory; the project has never been initialised |
| Directory size | **39.89 GB** |
| Files on disk | 9,264 (of which 9,022 are extracted dataset pickles) |
| Project files (excluding dataset pickles) | 264 at last count + 6 created during these audits |
| Python files | **33** at project root, flat, no packages |
| Structure | Flat root + `preprocessed/`, `hybrid_results/`, `episode_evaluation/`, `data archives` |

The root mixes three unrelated efforts: the VIDYUT pipeline, an **EDP-II course
report toolchain** (`build_report_*.py`, `generate_report.py`,
`verify_report.py`, and the two `.docx` files), and generic utilities plus
unrelated coursework directories. Only the first belongs in a VIDYUT repository.

## 2. Proposed structure

```
vidyut/
├── README.md                     # entry point, setup, reproduce commands
├── README_LIMITATIONS.md         # honest ledger
├── requirements.txt              # 6 pinned core packages
├── requirements-dev.txt          # python-docx (report tooling only)
├── .gitignore
├── LICENSE                       # MISSING — see §8
├── CITATION.cff                  # recommended: dataset + code citation
├── src/
│   ├── paths.py                  # NEW: repo-root resolution (see §6)
│   ├── preprocess.py
│   ├── features.py
│   ├── train_hybrid.py
│   ├── evaluate_episode_level.py
│   ├── verify_pipeline.py
│   ├── check_window_signal.py
│   ├── check_feature_space.py
│   ├── verify_storage.py
│   ├── verify_cleanup.py
│   ├── inspect_dataset.py
│   └── legacy/
│       ├── train_baseline.py
│       ├── train_baseline_features.py
│       ├── train_cnn.py
│       ├── evaluate.py
│       ├── plot_fault_waveforms.py
│       └── waveform_visualization.py
├── configs/
│   └── preprocessing_config.json
├── data/
│   ├── README.md                 # dataset card: DOI, checksums, extraction
│   ├── labels/hv_double_line_90kv_labels.csv
│   ├── raw/                      # .gitkeep only — 12 GB archive NOT committed
│   └── processed/                # .gitkeep only — 7.0 GB regenerated
├── models/
│   ├── expB_cnn_best.pt          # 159 KB  → Git LFS
│   ├── expA_rf_features_model.pkl# 30 MB   → Git LFS
│   └── expC_hybrid_rf_model.pkl  # 1.0 MB  → Git LFS
├── results/
│   ├── window_level/             # from hybrid_results/
│   ├── episode_level/            # from episode_evaluation/
│   └── legacy_windowing/         # from hybrid_results_legacy_windowing/
├── docs/
│   ├── technical_report.md
│   ├── audits/                   # these six audit documents
│   ├── dataset/                  # dataset_audit.md, validation reports, provenance
│   └── figures/                  # plots/ (8 EDA PNGs)
└── reports/                      # VIDYUT report + deck — DO NOT EXIST YET
```

## 3. File mapping table

### 3.1 VIDYUT core pipeline → `src/`

| Current path | Proposed | Size | Notes |
|---|---|---|---|
| `preprocess.py` | `src/preprocess.py` | 21 KB | owns `DATA_DIR`, `LABEL_CSV`, `OUT_DIR` |
| `features.py` | `src/features.py` | 4 KB | imported by `train_hybrid`, `evaluate_episode_level` |
| `train_hybrid.py` | `src/train_hybrid.py` | 23 KB | experiments A/B/C |
| `evaluate_episode_level.py` | `src/evaluate_episode_level.py` | 13 KB | new |
| `verify_pipeline.py` | `src/verify_pipeline.py` | 4 KB | split/normalisation verification |
| `check_window_signal.py` | `src/check_window_signal.py` | 5.5 KB | windowing defect evidence |
| `check_feature_space.py` | `src/check_feature_space.py` | 2 KB | z-scored vs physical features |
| `verify_storage.py` | `src/verify_storage.py` | 2 KB | bin ↔ PKL reconstruction |
| `verify_cleanup.py` | `src/verify_cleanup.py` | 4 KB | post-cleanup essentials check |
| `inspect_dataset.py` | `src/inspect_dataset.py` | 24 KB | dataset audit CLI |

### 3.2 Retired pipeline → `src/legacy/` (kept for provenance, clearly marked)

| Current path | Proposed | Size |
|---|---|---|
| `train_baseline.py` | `src/legacy/train_baseline.py` | 4.8 KB |
| `train_baseline_features.py` | `src/legacy/train_baseline_features.py` | 6.8 KB |
| `train_cnn.py` | `src/legacy/train_cnn.py` | 14 KB |
| `evaluate.py` | `src/legacy/evaluate.py` | 15 KB |
| `plot_fault_waveforms.py` | `src/legacy/plot_fault_waveforms.py` | 20 KB |
| `waveform_visualization.py` | `src/legacy/waveform_visualization.py` | 19 KB |

### 3.3 Configuration → `configs/`

| Current path | Proposed | Size |
|---|---|---|
| `preprocessing_config.json` | `configs/preprocessing_config.json` | 13 KB |

### 3.4 Data (labels committed, waveforms not) → `data/`

| Current path | Proposed | Size | Git? |
|---|---|---|---|
| `hv_double_line_90kv_labels.csv` | `data/labels/…` | 7.0 MB | **Yes** (7 MB, under a 10 MB threshold; or LFS) |
| `metadata_audit_full.csv` | `data/labels/metadata_audit_full.csv` | 7.0 MB | byte-identical duplicate (MD5 `5f015330…`) — commit one, or both with a note |
| `hv_double_line_90kv_preprocessed_data.zip` | `data/raw/` (not committed) | 12.0 GB | **No** — Zenodo, DOI 10.5281/zenodo.21109169 |
| `hv_double_line_90kv_preprocessed_data_extracted/` | `data/raw/extracted/` (not committed) | 22.0 GB | **No** |
| `preprocessed/` (6 bins + norm + manifests) | `data/processed/` (not committed) | 7.0 GB | **No** — regenerated by `preprocess.py` (~8 min) |
| `split_ids.csv` | `data/labels/split_ids.csv` | 161 KB | optional; superseded by `preprocessed/*_episodes.csv` |
| `pkl_out/` (7 episode pickles) | `data/samples/` or drop | 17 MB | **No** — duplicates of dataset members; only for the inspect demo |

### 3.5 Results → `results/`

| Current path | Proposed | Size | Git? |
|---|---|---|---|
| `hybrid_results/` (20 files) | `results/window_level/` | 31 MB | Yes, **except** the two RF `.pkl` (31 MB total) → `models/` + LFS |
| `episode_evaluation/` (5 files) | `results/episode_level/` | ~60 KB | Yes |
| `hybrid_results_legacy_windowing/` (16 files) | `results/legacy_windowing/` | 342 MB | Metrics + PNGs yes; the two chance-level RF `.pkl` (341 MB) **no** → keep local or drop |
| `baseline/` (3 files) | `results/legacy_windowing/baseline/` | 252 MB | `baseline_metrics.json` + PNG yes; `rf_model.pkl` (252 MB) **no** |
| `cnn/` | — | empty | Remove; nothing in it |
| `plots/` (8 PNGs) | `docs/figures/` | 7.3 MB | Yes |

### 3.6 Documentation → `docs/` and `reports/`

| Current path | Proposed | Size |
|---|---|---|
| `README.md` | `README.md` (root) | 11 KB |
| `README_LIMITATIONS.md` | `README_LIMITATIONS.md` (root) | 8.4 KB |
| `technical_report.md` | `docs/technical_report.md` | 4.5 KB |
| `CLEANUP_REPORT.md` | `docs/audits/` | 11 KB |
| `DEPENDENCY_AUDIT.md` | `docs/audits/` | 12 KB |
| `GIT_READINESS_AUDIT.md`, `SCIENTIFIC_INTEGRITY_AUDIT.md`, `REPRODUCIBILITY_AUDIT.md`, `FINAL_CONSISTENCY_AUDIT.md`, `SUBMISSION_READINESS_REPORT.md` | `docs/audits/` | — |
| `EPISODE_EVALUATION_NOTES.md` | `docs/` | 6 KB |
| `dataset_audit.md` (23 KB), `preprocessing_investigation.md` (27 KB), `preprocessing_validation_report.md` (12 KB), `training_readiness_report.md` (11 KB), `full_dataset_validation.md` (8.5 KB), `dataset_readiness_report.md` (2.8 KB), `changelog_task1_corrections.md` (4.8 KB), `recovery_log.md` (3.8 KB) | `docs/dataset/` | ~93 KB |
| `companion_repo_README.md`, `gh_README.md`, `zenodo_README.md`, `zenodo_files.json`, `zenodo21109169_files.json` | `docs/dataset/provenance/` | ~40 KB |
| `dataset_availability_report.csv` (748 KB), `dataset_availability_summary.{json,md}`, `metadata_audit_summary.json`, `metadata_derived.json`, `inspected_metadata.json`, `pkl_audit_output.json`, `audit_run_output.json` | `docs/dataset/audit_outputs/` | ~1.0 MB |
| `waveform_inspection.json` (1.0 MB), `waveform_inventory.csv`, `waveform_validation_summary.csv`, `waveform_offsets.json`, `waveform_rng.json`, `event_duration_short_counts.json` | `docs/dataset/audit_outputs/` | ~1.0 MB |
| `preprocessing_verification.txt`, `window_signal_diagnostic.json`, `subset_stats.json`, `report_verification.txt` | `docs/evidence/` | ~5 KB |
| `UTA024_Practical_Activity_Report.docx` (empty), `Thapar_Edp2_UTA024_Practical_Activity_Report.docx` | **exclude** — EDP-II course report | 89 KB |

### 3.7 Excluded — not part of VIDYUT

| Group | Files | Reason |
|---|---|---|
| EDP-II report toolchain | `build_report_all.py`, `build_report_body_part1.py`, `build_report_cover_toc.py`, `build_report_exp1/2/3.py`, `build_report_helpers.py`, `build_report_run.py`, `generate_report.py`, `verify_report.py` | Build a **circuit-design lab report**, not VIDYUT. `generate_report.py` targets `Thapar_Edp2_…docx`; `build_report_run.py` targets the empty `UTA024_…docx`. |
| Generic utilities | `open-roadmap.py`, `print-checklist.py`, `print-poster.py`, `serve-checklist.py`, `serve-posters.py`, `serve.py` | Serve/print unrelated HTML decks; 0 occurrences of "VIDYUT" |
| Dataset helper | `zekeparse.py` | Reads `metadata_audit_summary.json` to render audit text; keep in `docs/tools/` if wanted |
| Unrelated directories | `ARKA_DECK/` (CAN-bus vehicle security pitch), `DEADPOOL_MEASUREMENT_HANDBOOK/`, `Ele/`, `OMNITRIX_TD/`, `OPERATION20/`, `TD_5x5_Handbook/`, `WAYNE_TD_DOSSIER/`, `src/` | Different courses. **Must not enter the VIDYUT repo.** |
| App state | `.cloud-attachments/`, `.freebuff/`, `cleanup_review/` | Chat/tooling state |

## 4. Size inventory — what cannot go in Git

| Item | Size | Destination |
|---|---|---|
| `hv_double_line_90kv_preprocessed_data.zip` | 12.0 GB | Zenodo (external); document DOI + expected MD5 |
| `hv_double_line_90kv_preprocessed_data_extracted/` | 22.0 GB | Local only; reproducible from the archive |
| `preprocessed/*.bin` | 7.0 GB | Local only; `preprocess.py` regenerates in ~8 min |
| `baseline/rf_model.pkl` | 252 MB | Local only (chance-level artifact) |
| `hybrid_results_legacy_windowing/*.pkl` | 341 MB | Local only (audit metrics retained separately) |
| `hybrid_results/expA_rf_features_model.pkl` | 30 MB | **Git LFS** — the real experiment-A model |
| `hybrid_results/expC_hybrid_rf_model.pkl` | 1.0 MB | **Git LFS** (or plain Git) |
| `hybrid_results/expB_cnn_best.pt` | 159 KB | Plain Git |

**Recommended policy:** GitHub hard-blocks files >100 MB, warns >50 MB. Only
`expA_rf_features_model.pkl` (30 MB) approaches that; LFS covers it cleanly.
Everything above 30 MB should never be committed — it is either published data
(Zenodo) or regenerable output.

## 5. `.gitignore`

```gitignore
# ---- datasets (published on Zenodo; see data/README.md) ----
hv_double_line_90kv_preprocessed_data.zip
hv_double_line_90kv_preprocessed_data_extracted/
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
pkl_out/

# ---- regenerable preprocessed windows (~7 GB) ----
preprocessed/
X_*.bin

# ---- large / obsolete model binaries ----
baseline/
hybrid_results_legacy_windowing/*.pkl
cnn/
**/*.pt.tmp

# ---- python ----
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
env/
.pytest_cache/
.ipynb_checkpoints/
*.egg-info/

# ---- OS / editor ----
.DS_Store
Thumbs.db
desktop.ini
.vscode/
.idea/
*.swp

# ---- logs & scratch ----
*.log
episode_eval_run.log
zip_md5_check.txt
report_text_dump.txt
cleanup_review/

# ---- local secrets (none present; defensive) ----
.env
.env.*
*.pem

# ---- NOT ignored: the .docx in this folder belongs to another course ----
# (it is simply not added to this repository)
```

Note: `*.log` is ignored above, which would also hide the useful run logs
`preprocessed_run.log` / `hybrid_run.log`. Either whitelist those two
(`!preprocessed_run.log`) or move them to `docs/evidence/` and let them commit.

## 6. Broken paths and imports after restructuring

The pipeline is **CWD-relative**. Every path literal resolves from the process
working directory, and four scripts depend on `Path.cwd()` being the directory
that contains `preprocess.py`.

Verified counts across the current tree:

| Pattern | Files affected |
|---|---|
| literal `preprocessed` | **16** `.py` files |
| literal `hybrid_results` | 3 (`train_hybrid.py`, `evaluate_episode_level.py`, `verify_cleanup.py`) |
| literal `hv_double_line_90kv` | 8 (dataset paths/label CSV) |
| local imports (`from preprocess/features/train_hybrid import …`) | 8 |

Per-file detail:

| File | Import mechanism | Breaks if moved to `src/`? |
|---|---|---|
| `train_hybrid.py` | `sys.path.insert(0, Path(__file__).parent)` then `from features import …` | **No**, as long as `features.py` moves with it |
| `evaluate_episode_level.py` | same pattern | **No** — same condition |
| `evaluate.py` | `sys.path.insert(0, Path.cwd())` then `import preprocess` | **Yes** — running from repo root no longer finds `preprocess` |
| `train_baseline.py` | `Path.cwd()` | **Yes** |
| `train_baseline_features.py` | `Path.cwd()` | **Yes** |
| `train_cnn.py` | `Path.cwd()` | **Yes** |
| `verify_storage.py` | none; `from preprocess import …` | **No** (script dir is auto-added to `sys.path`) but fragile |
| `check_feature_space.py` | none; `from features import …` | **No**, but fragile |
| `check_window_signal.py` | none (imports `preprocess` indirectly? no — self-contained) | **No** |
| `verify_cleanup.py` | none | **No** |

Path literals that must change (all of these):

```
preprocess.py:            EXTRACT_DIR, LABEL_CSV, OUT_DIR="preprocessed", CONFIG_PATH
features.py:              load_norm(preproc_dir="preprocessed")
train_hybrid.py:          PREPROC=Path("preprocessed"), OUT_DIR=Path("hybrid_results")
evaluate_episode_level.py:PREPROC, RESULTS, OUT, "hv_double_line_90kv_labels.csv"
verify_pipeline.py:       PREPROC=Path("preprocessed")
verify_storage.py:        "preprocessed/…", "hv_double_line_90kv_labels.csv"
verify_cleanup.py:        all ESSENTIAL paths, "hv_double_line_90kv_preprocessed_data.zip"
inspect_dataset.py:       CLI defaults
plot_fault_waveforms.py:  dataset + plots dirs
waveform_visualization.py:dataset + plots dirs
check_window_signal.py:   LABEL_CSV, PREPROC
check_feature_space.py:   "preprocessed/…"
```

**Minimal-risk migration recipe** (recommended over "move and hope"):

1. Add `src/paths.py`:
   ```python
   from pathlib import Path
   ROOT = Path(__file__).resolve().parents[1]
   DATA_RAW = ROOT / "data" / "raw"
   DATA_LABELS = ROOT / "data" / "labels"
   DATA_PROCESSED = ROOT / "data" / "processed"
   MODELS = ROOT / "models"
   RESULTS = ROOT / "results"
   CONFIGS = ROOT / "configs"
   ```
2. Replace every CWD-relative literal with a `paths` constant, and change the
   four `Path.cwd()` insertions to `Path(__file__).resolve().parent`.
3. Move files, then run the verification gate:
   `python src/verify_pipeline.py && python src/evaluate_episode_level.py`
   — the second is the strongest regression test available: it re-derives the
   window→episode mapping and aborts if any byte disagrees, and it aborts if
   window metrics differ from the stored results.
4. `run_modeling.sh` already does `cd "$(dirname "$0")"`; update its three
   commands to `python src/….py` and it keeps working.

## 7. Will the README stay valid?

**Partly.** `README.md` §6 currently tells the reader to run
`python preprocess.py` etc. from the repository root, and references
`hybrid_results/…` and `preprocessed/…`. After restructuring, these all move:

| README reference | Breaks? | Fix |
|---|---|---|
| `python preprocess.py` / `verify_pipeline.py` / `train_hybrid.py` | yes | `python src/preprocess.py …` |
| `python check_window_signal.py`, `check_feature_space.py`, `verify_storage.py` | yes | `src/…` |
| artifact table paths (`hybrid_results/…`) | yes | `results/window_level/…` |
| `run_modeling.sh` | no (self-locating) | update inner commands |
| Dataset section (Zenodo DOI, archive name) | no | keep; add `data/README.md` |
| §5 results tables and §8/§9 notes | no | unchanged |
| `requirements.txt` / `requirements-dev.txt` | no | unchanged |

So the README needs a paths pass, not a rewrite. Budget ~20 line edits.

## 8. Gaps found (not fixed — audit only)

1. **No `.git`, no `.gitignore`, no `LICENSE`, no `CITATION.cff`.** The dataset is
   CC BY 4.0 and the companion repo is MIT; a derived-work repository should
   state its own code licence and carry the dataset attribution.
2. **No VIDYUT report or presentation exists.** The two `.docx` files in the root
   are an EDP-II circuit lab report, and `UTA024_Practical_Activity_Report.docx`
   is an empty shell (0 paragraphs, 0 tables, `document.xml` = 1,548 bytes).
   `reports/` would be empty on day one.
3. **`zenodo_record.json` is the wrong record** — it describes an unrelated
   transport-economics paper (DOI 10.5281/zenodo.15348976), not PROTECT-90. The
   correct provenance is in `zenodo_README.md`, `companion_repo_README.md` and
   `zenodo21109169_files.json`. Do not commit the misleading file, or replace it.
4. **`cnn/` is empty** while `technical_report.md` cites `cnn/best_model.pt`.
5. **`scripts/` was removed in cleanup**; `cleanup_review/` holds one quarantined
   stale runner — exclude it from Git.
6. Report-title terminology drift: the project is called **VIDYUT** in the brief
   and *VIDYUT-KAVACHA* in `README.md`, but the dataset is **PROTECT-90**. Pick
   one framing before publishing.

## 9. Uncertain files — manual review before committing

| File | Why uncertain |
|---|---|
| `metadata_audit_full.csv` | Byte-identical duplicate of the labels CSV; keeping both doubles a 7 MB data file in Git |
| `split_ids.csv` | 161 KB, 4 columns; appears superseded by `preprocessed/*_episodes.csv` but nothing documents that |
| `docs/dataset/*` (8 markdown reports) | Historical audit trail from earlier pipeline versions; contains numbers that predate the windowing fix |
| `baseline_metrics.json` + PNG | Document a model trained on the *defective* windows; keep only as a negative result |
| `hybrid_results_legacy_windowing/*.pkl` | 341 MB of chance-level models; the metrics are the useful part |
| `waveform_inspection.json` (1.0 MB) | Large audit blob; referenced only by its own audit docs |
| `project-freebuck-roadmap.html` / `-checklist.html` | 0 VIDYUT mentions; likely unrelated tooling output |

## 10. Recommendation

Do **not** restructure and publish in one step. Sequence:

1. Create `mirror/` staging (or a fresh clone target) and copy the ~60 VIDYUT
   files listed in §3, leaving all data directories behind.
2. Add `src/paths.py` and `configs/`, apply the path edits from §6.
3. Run the regression gate (`verify_pipeline.py` + `evaluate_episode_level.py`)
   inside the new layout before the first commit.
4. Commit code + docs + small results; add `models/` via Git LFS in a second
   commit; never add datasets.
5. Only then rewrite README §6 for the new paths.

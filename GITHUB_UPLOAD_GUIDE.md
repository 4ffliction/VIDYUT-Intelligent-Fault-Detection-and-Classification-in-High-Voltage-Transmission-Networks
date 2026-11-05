# GitHub Upload Guide for VIDYUT-KAVACHA

This guide explains exactly what is included in `VIDYUT_GITHUB_RELEASE/`,
what has been deliberately excluded, and how to publish it to GitHub.

---

## 1. Location of the completed release folder

**Path:** `VIDYUT_GITHUB_RELEASE/` in the project root
(`C:/Users/sharm/Downloads/Study_Material_Sem3/`).

**Total size:** ~8.2 MB on disk (71 files).

**Structure:**
```
VIDYUT_GITHUB_RELEASE/
├── README.md                          # Main project readme
├── README_LIMITATIONS.md              # Honest limitations ledger
├── LICENSE                            # Apache License 2.0
├── .gitignore                         # Comprehensive Git ignore rules
├── requirements.txt                   # 6 pinned Python dependencies
├── GITHUB_UPLOAD_GUIDE.md            # This file
├── configs/
│   └── preprocessing_config.json     # Preprocessing configuration
├── data/
│   └── README.md                     # Data directory readme (placeholder)
├── docs/
│   ├── CLEANUP_REPORT.md
│   ├── DEPENDENCY_AUDIT.md
│   ├── EPISODE_EVALUATION_NOTES.md
│   ├── GIT_READINESS_AUDIT.md
│   ├── REPRODUCIBILITY_AUDIT.md
│   ├── SCIENTIFIC_INTEGRITY_AUDIT.md
│   ├── changelog_task1_corrections.md
│   ├── companion_repo_README.md
│   ├── dataset_audit.md
│   ├── dataset_availability_summary.md
│   ├── dataset_readiness_report.md
│   ├── figures/                      # 8 PNG figures (EDA + confusion matrices)
│   │   ├── class_comparison.png
│   │   ├── waveform_sample0000.png
│   │   ├── waveform_sample1000.png
│   │   ├── waveform_sample1001.png
│   │   ├── waveform_sample1002.png
│   │   ├── waveform_sample1287.png
│   │   ├── waveform_sample1807.png
│   │   └── waveform_sample2338.png
│   ├── full_dataset_validation.md
│   ├── gh_README.md
│   ├── preprocessing_investigation.md
│   ├── preprocessing_validation_report.md
│   ├── recovery_log.md
│   ├── technical_report.md
│   ├── training_readiness_report.md
│   └── zenodo_README.md
├── models/
│   └── README.md                     # Models directory readme
├── reports/
│   └── README.md                     # Reports directory readme
├── results/
│   ├── classification_reports/       # 4 validation + 1 final test reports
│   │   ├── expA_val_classification_report.txt
│   │   ├── expB_val_classification_report.txt
│   │   ├── expC_val_classification_report.txt
│   │   ├── final_test_classification_report.txt
│   │   └── episode_level_classification_report.csv
│   ├── confusion_matrices/           # 6 confusion matrix PNGs
│   │   ├── baseline_confusion_matrix.png
│   │   ├── episode_level_confusion_matrix.png
│   │   ├── expA_val_confusion_matrix.png
│   │   ├── expB_val_confusion_matrix.png
│   │   ├── expC_val_confusion_matrix.png
│   │   └── final_test_confusion_matrix.png
│   ├── metrics/                      # 6 metric JSON files
│   │   ├── cnn_training_history.json
│   │   ├── comparison.json
│   │   ├── expA_val_metrics.json
│   │   ├── expB_val_metrics.json
│   │   ├── expC_val_metrics.json
│   │   └── final_test_metrics.json
│   └── training_curves/
│       └── cnn_training_history.png  # CNN training curve
└── src/
    ├── check_feature_space.py
    ├── check_window_signal.py
    ├── evaluate.py
    ├── evaluate_episode_level.py
    ├── features.py
    ├── inspect_dataset.py
    ├── plot_fault_waveforms.py
    ├── preprocess.py
    ├── run_modeling.sh
    ├── train_baseline.py
    ├── train_baseline_features.py
    ├── train_cnn.py
    ├── train_hybrid.py
    ├── verify_cleanup.py
    ├── verify_pipeline.py
    ├── verify_storage.py
    └── waveform_visualization.py
```

---

## 2. What is included

### Core Python source code (17 files in `src/`)
All the final, clean Python scripts that implement the full pipeline:
- `preprocess.py` — data preprocessing and window extraction
- `features.py` — feature engineering
- `train_hybrid.py` — training orchestration for experiments A/B/C
- `train_baseline.py` — baseline Random Forest on raw windows
- `train_baseline_features.py` — baseline Random Forest on engineered features
- `train_cnn.py` — standalone CNN training script
- `evaluate.py` — consolidated evaluation (legacy, for reference)
- `evaluate_episode_level.py` — episode-level evaluation (new)
- `verify_pipeline.py` — verification of preprocessing
- `verify_storage.py` — verification of stored windows
- `verify_cleanup.py` — verification that essential files are intact
- `check_window_signal.py` — diagnostic for windowing defect
- `check_feature_space.py` — diagnostic for feature space
- `inspect_dataset.py` — dataset inspection and verification
- `plot_fault_waveforms.py` — fault waveform plotting
- `waveform_visualization.py` — waveform visualization utilities
- `run_modeling.sh` — shell script to run the full pipeline

### Configuration (1 file)
- `configs/preprocessing_config.json` — full preprocessing configuration

### Documentation (19 Markdown files in `docs/`)
- `README.md` — main project description
- `README_LIMITATIONS.md` — honest limitations ledger
- `CLEANUP_REPORT.md` — cleanup audit report
- `DEPENDENCY_AUDIT.md` — dependency audit
- `EPISODE_EVALUATION_NOTES.md` — episode evaluation notes
- `GIT_READINESS_AUDIT.md` — Git readiness audit
- `REPRODUCIBILITY_AUDIT.md` — reproducibility audit
- `SCIENTIFIC_INTEGRITY_AUDIT.md` — scientific integrity audit
- `changelog_task1_corrections.md` — changelog of corrections
- `companion_repo_README.md` — companion repository readme
- `dataset_audit.md` — dataset audit
- `dataset_availability_summary.md` — dataset availability summary
- `dataset_readiness_report.md` — dataset readiness report
- `full_dataset_validation.md` — full dataset validation
- `gh_README.md` — GitHub-specific readme
- `preprocessing_investigation.md` — preprocessing investigation
- `preprocessing_validation_report.md` — preprocessing validation
- `recovery_log.md` — recovery log
- `technical_report.md` — technical report
- `training_readiness_report.md` — training readiness report
- `zenodo_README.md` — Zenodo documentation

### Figures (8 PNG files in `docs/figures/`)
- `class_comparison.png` — class comparison plot (1.4 MB)
- `waveform_sample0000.png` — waveform sample 0 (0.9 MB)
- `waveform_sample1000.png` — waveform sample 1000 (0.7 MB)
- `waveform_sample1001.png` — waveform sample 1001 (0.9 MB)
- `waveform_sample1002.png` — waveform sample 1002 (0.9 MB)
- `waveform_sample1287.png` — waveform sample 1287 (0.8 MB)
- `waveform_sample1807.png` — waveform sample 1807 (0.8 MB)
- `waveform_sample2338.png` — waveform sample 2338 (0.7 MB)

### Results (17 files in `results/`)
- 4 classification reports (text) + 1 episode-level report (CSV)
- 6 confusion matrix PNGs
- 6 metric JSON files
- 1 training curve PNG

### Legal and packaging (4 files in root)
- `LICENSE` — Apache License 2.0
- `.gitignore` — comprehensive Git ignore rules
- `requirements.txt` — 6 pinned Python dependencies
- `GITHUB_UPLOAD_GUIDE.md` — this file

### Placeholder READMEs (3 files)
- `data/README.md` — explains external dataset hosting
- `models/README.md` — explains model checkpoints are NOT included
- `reports/README.md` — explains reports are NOT ready yet

---

## 3. What has deliberately been excluded

### Dataset files (EXCLUDED — too large)
- `hv_double_line_90kv_preprocessed_data.zip` (12 GB) — hosted on Zenodo
- `hv_double_line_90kv_preprocessed_data_extracted/` (22 GB) — extracted waveforms
- `pkl_out/` (17 MB) — sample PKL files for inspection
- `preprocessed/` (7 GB) — binary window stores (regenerable)

### Model checkpoints (EXCLUDED — see below)
- `expB_cnn_best.pt` (159 KB) — best CNN checkpoint
- `expA_rf_features_model.pkl` (30 MB) — Random Forest model (Experiment A)
- `expC_hybrid_rf_model.pkl` (1.0 MB) — Random Forest model (Experiment C, hybrid)

### Large results (EXCLUDED — regenerable)
- `hybrid_results/` (31 MB) — full experiment results directory
- `episode_evaluation/` (112 KB) — episode evaluation results (regenerable)
- `baseline/` (252 MB) — baseline results (too large)
- `basel

# DEPENDENCY_AUDIT.md — VIDYUT-KAVACHA

Audit of every Python file in the VIDYUT-KAVACHA project, the third-party
packages they actually import, and the pinned versions verified to install and
run in a clean virtual environment.

Date: 4 November 2026
Interpreter audited: **CPython 3.14.0** (`C:\Python314\python.exe`, Windows
x86-64)
Method: AST parse of all 32 project `.py` files (`ast.Import` /
`ast.ImportFrom`, including function-level and conditional imports), filtered
against `sys.stdlib_module_names` and against the set of local modules
(`preprocess`, `features`, `train_hybrid`, …). Versions taken from
`pip freeze` / package metadata — none guessed.

---

## 1. Summary

| | |
|---|---|
| Python files inspected | 32 |
| Files with third-party imports | 24 |
| Files using only the standard library | 8 |
| Distinct third-party packages required | **6** core + 1 dev |
| Packages installed in the interpreter but excluded | 24 |

Core runtime: `numpy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`,
`torch`.
Optional (document generation only): `python-docx`.

## 2. Core runtime dependencies (`requirements.txt`)

These six are required to run `preprocess.py` → `verify_pipeline.py` →
`train_hybrid.py` and therefore to reproduce every reported metric and figure.

| Package | Import name | Pinned | Purpose in this project | Imported by (count) |
|---|---|---|---|---|
| numpy | `numpy` | `2.3.5` | Window arrays, feature statistics (RMS/std/ptp/max-abs), memmap loading of the `.bin` splits, metric arithmetic, normalisation stats | 15 files |
| pandas | `pandas` | `3.1.0` | Reading episode pickles (`read_pickle`), the labels CSV, episode split manifests, and result CSV/JSON tables | 8 files |
| scikit-learn | `sklearn` | `1.9.1` | `RandomForestClassifier` (experiments A and C), `train_test_split` (episode-level 70/15/15 split), `GroupKFold` (diagnostics), metrics (`f1_score`, `accuracy_score`, `confusion_matrix`, `precision_recall_fscore_support`, `classification_report`) | 8 files |
| joblib | `joblib` | `1.6.0` | Persisting and reloading the two RF models (`expA_rf_features_model.pkl`, `expC_hybrid_rf_model.pkl`) | 5 files |
| matplotlib | `matplotlib` | `3.11.2` | `Agg`-backend confusion-matrix and training-history figures | 7 files |
| torch | `torch` | `2.9.1` | `Small1DCNN` (`nn.Module`, `Conv1d`, `BatchNorm1d`, `AdaptiveAvgPool1d`), `AdamW`, `ReduceLROnPlateau`, `CrossEntropyLoss`, checkpoint save/load, batched embedding extraction | 5 files |

### Exact import statements found

**numpy** — 15 files: `preprocess.py`, `features.py`, `train_hybrid.py`,
`verify_pipeline.py`, `verify_storage.py`, `verify_cleanup.py`,
`check_window_signal.py`, `check_feature_space.py`, `inspect_dataset.py`,
`plot_fault_waveforms.py`, `waveform_visualization.py`, `evaluate.py`,
`train_baseline.py`, `train_baseline_features.py`, `train_cnn.py`
(all plain `import numpy`).

**pandas** — 8 files: `preprocess.py`, `check_window_signal.py`,
`verify_pipeline.py`, `verify_storage.py`, `verify_cleanup.py`,
`inspect_dataset.py`, `plot_fault_waveforms.py`, `waveform_visualization.py`
(all plain `import pandas`).

**scikit-learn** — 8 files:

| File | Import |
|---|---|
| `preprocess.py` | `from sklearn.model_selection import train_test_split` |
| `train_hybrid.py` | `from sklearn.ensemble import RandomForestClassifier`; `from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support` |
| `check_window_signal.py` | `from sklearn.ensemble import RandomForestClassifier`; `from sklearn.metrics import f1_score`; `from sklearn.model_selection import GroupKFold` |
| `check_feature_space.py` | `from sklearn.ensemble import RandomForestClassifier`; `from sklearn.metrics import f1_score` |
| `evaluate.py`, `train_cnn.py` | `from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score, accuracy_score` |
| `train_baseline.py`, `train_baseline_features.py` | `from sklearn.ensemble import RandomForestClassifier`; `from sklearn.metrics import confusion_matrix`; `from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix` |

**torch** — 5 files:

| File | Import |
|---|---|
| `features.py` | `import torch` |
| `train_hybrid.py` | `import torch`; `import torch.nn` |
| `train_cnn.py` | `import torch`; `import torch.nn`; `from torch.utils.data import DataLoader, Dataset` |
| `evaluate.py` | `import torch`; `import torch.nn`; `from torch.utils.data import DataLoader, TensorDataset` |
| `verify_cleanup.py` | `import torch` |

**joblib** — 5 files: `train_hybrid.py`, `evaluate.py`, `train_baseline.py`,
`train_baseline_features.py`, `verify_cleanup.py`.

**matplotlib** — 7 files: `train_hybrid.py`, `train_cnn.py`,
`train_baseline.py`, `train_baseline_features.py`, `evaluate.py`,
`plot_fault_waveforms.py`, `waveform_visualization.py`
(`import matplotlib` + `import matplotlib.pyplot`, always with
`matplotlib.use("Agg")` where figures are written headlessly).

## 3. Optional / development dependency (`requirements-dev.txt`)

| Package | Import name | Pinned | Purpose | Imported by |
|---|---|---|---|---|
| python-docx | `docx` | `1.2.0` | Builds and verifies the `.docx` submission reports | 9 files |

Files: `build_report_all.py`, `build_report_body_part1.py`,
`build_report_exp1.py`, `build_report_exp2.py`, `build_report_exp3.py`,
`build_report_helpers.py`, `build_report_run.py`, `generate_report.py`,
`verify_report.py` — all via `from docx import Document`,
`docx.shared`, `docx.enum.*`, `docx.oxml`.

This is kept separate because **no experiment, metric or figure depends on it**:
`requirements.txt` alone reproduces the full reported result set.

**Non-pip system dependency (optional).** `build_report_run.py` shells out to
LibreOffice (`soffice`) to convert DOCX → PDF. It is an external binary, not a
pip package, and the `.docx` deliverables build without it.

## 4. Files using only the standard library (8)

`build_report_cover_toc.py`, `open-roadmap.py`, `print-checklist.py`,
`print-poster.py`, `serve-checklist.py`, `serve-posters.py`, `serve.py`,
`zekeparse.py`.

They import `os`, `sys`, `json`, `pathlib`, `subprocess`, `http.server`,
`shutil`, `re`, `importlib.util` — all standard library, so they impose no
requirement. (`subprocess` calls in the poster/serve scripts invoke the OS print
and browser, not Python packages.)

## 5. Excluded — installed but imported by no VIDYUT file

These appear in the system interpreter's `pip freeze` but are not used by this
project (they belong to other coursework on the same machine):

`docx2pdf`, `paddleocr`, `opencv-python`, `transformers`, `huggingface-hub`,
`tokenizers`, `safetensors`, `playwright`, `pyee`, `pymupdf`, `pypdf`,
`requests`, `brotli`, `xlsxwriter`, `pywin32`, `regex`, `tqdm`, `colorama`,
`greenlet`, `certifi`, `charset-normalizer`, `idna`, `urllib3`, `PyYAML`.

Evidence: none of these names occurs in any `import`/`from … import` statement
across the 32 files (AST scan), **and** the clean virtual environment built from
`requirements.txt` alone resolved without any of them (see §7). In particular
`docx2pdf` is *not* used even by the report builders — PDF export goes through
LibreOffice instead.

Also excluded, though present in the system freeze: `lxml`, `scipy`, `sympy`,
`networkx`, `contourpy`, `fonttools`, `pillow`, `Jinja2`, etc. These are
**transitive** dependencies of the six core packages, not direct imports, so
they are correctly left unpinned in `requirements.txt` (pip resolves them; the
verified resolution is recorded in §8).

## 6. Python-version compatibility

* Target: **CPython 3.14.0**. Every pinned version has a `cp314` wheel for
  Windows x86-64 — proven by a successful from-scratch install (§7), not
  assumed.
* `torch==2.9.1` resolves to the CPU build (`2.9.1+cpu`) from PyPI on this
  platform; CUDA is not required. The code guards with
  `torch.device("cuda" if torch.cuda.is_available() else "cpu")`, so the same
  pin works on CPU-only and CUDA hosts.
* `pandas==3.1.0` is a major-version line; the project uses only stable APIs
  (`read_csv`, `read_pickle`, `DataFrame.values`, `to_numeric`).
* `numpy==2.3.5` is the 2.x ABI the code was run against (`np.memmap`,
  `np.frombuffer`, `np.bincount`, `default_rng`).

## 7. Verification performed (clean virtual environment)

A throwaway venv was created inside the project, used, and then deleted. The
system interpreter and its site-packages were **not** modified; nothing was
uninstalled.

```
python -m venv .venv_verify
.venv_verify/Scripts/python -m pip install -r requirements.txt      # exit 0
.venv_verify/Scripts/python -m pip install -r requirements-dev.txt  # exit 0
```

Reported versions inside the clean venv (exactly the pins):

```
numpy 2.3.5 | pandas 3.1.0 | sklearn 1.9.1 | joblib 1.6.0
matplotlib 3.11.2 | torch 2.9.1+cpu | python-docx 1.2.0
```

End-to-end smoke test with the real saved artifacts, no training:

| Check | Result |
|---|---|
| `torch.load("hybrid_results/expB_cnn_best.pt")` + `Small1DCNN.load_state_dict` | OK |
| CNN forward on 64 real test windows → 96-D embeddings | OK, shape (64, 96), all finite |
| `joblib.load("hybrid_results/expC_hybrid_rf_model.pkl")` + predict on fused features | OK, 4-class output |
| `features.features_batched` + `features.load_norm` | OK |
| matplotlib `Agg` figure render | OK |

## 8. Transitive set as resolved during verification

Recorded for the record; **not** pinned in `requirements.txt` because they are
not direct imports. This is the exact closure the clean venv produced on
CPython 3.14:

```
Jinja2==3.1.6            MarkupSafe==3.0.4        cloudpickle==3.1.2
contourpy==1.4.0         cycler==0.12.1           filelock==4.2.8
fonttools==4.68.0        fsspec==2026.9.0         kiwisolver==1.5.1
mpmath==1.3.0            narwhals==2.28.0         networkx==3.7
packaging==26.3          pillow==13.0.0           pyparsing==3.3.3
python-dateutil==4.0.0   scipy==1.18.1            setuptools==84.2.1
sympy==1.14.0            threadpoolctl==3.7.0     typing_extensions==4.16.0
tzdata==2026.6
```

Note these resolved to slightly *newer* patch versions than the system
interpreter had (e.g. `packaging` 25.0 → 26.3, `python-dateutil`
2.9.0.post0 → 4.0.0, `pillow` 12.3.0 → 13.0.0) and the pipeline still ran, which
shows the results do not depend on those exact transitive builds. If a
bit-identical environment is ever required, generate a lock with
`pip freeze > requirements.lock.txt` inside a resolved venv.

## 9. Commands

```bash
# core (reproduces all reported metrics and figures)
python -m pip install -r requirements.txt

# optional (only for building the .docx reports)
python -m pip install -r requirements-dev.txt

# reproduce
python preprocess.py
python verify_pipeline.py
python train_hybrid.py

# sanity-check an installed environment
python verify_cleanup.py
```

## 10. Caveats

1. The six pins are exact for the environment in which the results in
   `hybrid_results/` were produced, so this file documents a *tested* set rather
   than a merely compatible one.
2. `python-docx` is separated as dev-only by design. If the `.docx` submission
   files are considered part of "runtime", merge that single line into
   `requirements.txt` — nothing else changes.
3. Optional acceleration is unused: no package here is installed with CUDA,
   MKL-specific or GPU-only extras, so the file behaves identically on a
   CPU-only machine.

# Training Readiness Report

## VIDYUT-KAVACHA: Can We Proceed to CNN Training Today?

**Date:** 2026-11-02
**Dataset:** PROTECT-90 v1.0.0 (Zenodo 21109169)

---

## Executive Summary

**VERDICT: NOT READY**

The project **cannot proceed to CNN training today** with scientifically defensible results. The complete waveform archive is not available locally — only **7 of 9,022 waveform files** (0.078%) are present. The local ZIP is a truncated partial download (3.76 GB of expected 12.05 GB, 31.2%) with no valid central directory or EOCD.

**Required action:** Download the complete 12 GB waveform archive from Zenodo record 21109169.

---

## Detailed Assessment

### Dataset Source Verification

| Item | Status | Details |
|------|--------|---------|
| Zenodo record 21109169 | ✅ VERIFIED | Primary record for PROTECT-90 v1.0.0, published 2026-07-01 |
| Zenodo record 18418330 | ✅ VERIFIED | Earlier v0.1.0 release (March 2026), same file contents |
| Zenodo concept DOI 18418329 | ✅ VERIFIED | Concept record grouping all versions |
| Local metadata CSV MD5 | ✅ VERIFIED | Matches Zenodo: `5f015330f77ed53b76bd5db26e83c48d` |
| Local waveform ZIP | ❌ INCOMPLETE | 3.76 GB vs 12.05 GB expected (31.2%), no central directory |
| Local ZIP checksum | ❌ DOES NOT MATCH | MD5 `7cf176f169299b825ba6a6be102edca8` not achieved |

### Source Discrepancy Resolution

| ID/DOI | Nature | Status |
|--------|--------|--------|
| 21109169 | Current stable release v1.0.0 | **CORRECT — USE THIS** |
| 18418330 | Earlier alpha/beta v0.1.0 | Legitimate PROTECT-90 record, but not current |
| 18418329 | Zenodo concept DOI (series) | Not for direct download, but resolves to latest version |
| 15348976 (in local zenodo_record.json) | Unrelated paper | INCORRECT — copy-paste error in local files |

The earlier reference to "18418330" was correct but pointed to the v0.1.0 release. The v1.0.0 release is 21109169. The concept DOI 18418329 is the series identifier.

### Waveform Availability

| Split | Usable Episodes | Notes |
|-------|-----------------|-------|
| Train | **0** | Cannot create meaningful training set |
| Validation | **0** | Cannot create meaningful validation set |
| Test | **0** | Cannot create meaningful test set |
| **Total usable** | **0** | **Cannot proceed to training** |

With only 7 episodes total (5 train, 2 val), a stratified train/val/test split is mathematically impossible:
- Train: 5 episodes (1 LLL, 1 LL, 2 SLG, 1 LLG) — not enough for 4-class training
- Validation: 2 episodes (1 LL, 1 SLG) — 2 samples cannot represent 4 classes
- Test: 0 episodes — no held-out evaluation possible

### Why 0 Usable Episodes Per Split

The 7 available episodes are distributed across classes as:
- LLL: 1 episode
- LL: 2 episodes
- SLG: 3 episodes
- LLG: 1 episode

Attempting a 70/15/15 split on these 7 results in:
- Train: 5 (not enough samples per class for CNN training)
- Val: 2 (not enough to validate)
- Test: 0 (cannot evaluate)

Even if one were to attempt training on all 7, the result would be **scientifically indefensible**:
1. No held-out test set
2. 1-3 samples per class is orders of magnitude too few for any learning
3. Cannot assess generalization
4. Cannot compare against any baseline

---

## What Would Be Needed

### Minimum Viable Dataset

For a scientifically defensible four-class fault classifier:

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Episodes per class | 500+ | 2,000+ |
| Total episodes | 2,000+ | 9,022 (full dataset) |
| Train/val/test split | Possible with 2000+ | 70/15/15 on 9022 |
| Test set size | 300+ (75 per class minimum) | ~1,354 (full 15%) |
| Short-duration coverage | All 404 episodes present | All 404 + all 5 ultra-short |

### With Current 7 Episodes

| Requirement | Current State | Verdict |
|-------------|---------------|---------|
| Episodes per class | 1-3 | ❌ FAR BELOW MINIMUM |
| Total episodes | 7 | ❌ FAR BELOW MINIMUM |
| Train/val/test split | 5/2/0 | ❌ IMPOSSIBLE |
| Test set | 0 episodes | ❌ NO EVALUATION POSSIBLE |
| Short-duration coverage | 0 of 404 | ❌ NONE |

---

## Data Completeness Analysis

### What's Missing

The local ZIP contains 2,812 local file headers, each corresponding to a waveform file. However, only **7** have been extracted and are usable (in `pkl_out/`). The other ~2,805 headers in the ZIP may or may not be extractable — they were never tested for validity.

**Critical gap:** The ZIP is missing its central directory and EOCD entirely. This means:
- The last ~6210 files (9022 - 2812) were never written to the archive
- The remaining 2812 headers in the archive may include valid files, but extraction was never completed
- The 7 files in `pkl_out/` are the only verified-accessible waveforms

### Short-Duration Episodes

All 404 episodes with fault duration < 128 samples (1 cycle) are **missing** from available waveforms. This is a significant gap because:
- These 404 episodes represent 4.48% of the dataset
- They include the 5 ultra-short events (< 1 sample)
- They span all fault classes (not concentrated in any single class)
- They test the edge cases of the window extraction policy

### Fault Target Coverage

No episodes from `Line_2_3_b` are available. This means:
- 0% of that fault target's 1,338 episodes are available
- Any model trained on available data would never see this fault scenario

---

## Pipeline Readiness

**The preprocessing pipeline itself is READY** (after bug fixes documented in `preprocessing_validation_report.md`):

| Component | Status | Notes |
|-----------|--------|-------|
| Fault label mapping | ✅ READY | sc_type → 0-3 labels, correct |
| Episode-wise splitting | ✅ READY | StratifiedShuffleSplit with seed=42 |
| No leakage between splits | ✅ READY | One window per episode design |
| Train-only normalization | ✅ READY | Fit on train, apply to all |
| Timestamp alignment | ✅ READY | time_s → sample index with clamping |
| Channel selection | ✅ READY | 6 channels from Bus_1_Line_01_02A |
| Short-duration handling | ✅ READY | No silent exclusion; clamped windows |
| Fixed random seeds | ✅ READY | seed=42 throughout |
| Output dimensions/dtypes | ✅ READY | float32 X, int64 y |

**The pipeline can execute** once the full waveform archive is available.

---

## Window Extraction Validation

Based on the 7 available episodes and metadata analysis:

| Window Length | Samples | Cycles | Episodes Captured (full fault) | Validity |
|---------------|---------|--------|-------------------------------|----------|
| 128 samples | 128 | 1 cycle (20 ms) | ~95.5% of all 9,022 episodes | Viable for most events, but misses inception transient for late faults |
| 256 samples | 256 | 2 cycles (40 ms) | ~90.8% of all episodes | **RECOMMENDED** — captures inception + steady-state for most faults |
| 512 samples | 512 | 4 cycles (80 ms) | ~82.2% of all episodes | More context but may extend beyond clearing for short faults |

**For the recommended 256-sample window:**
- 828 episodes (9.18%) have fault duration < 256 samples → window will contain partial fault
- 404 episodes (4.48%) have fault duration < 128 samples → window may have no fault-active samples
- 5 episodes (0.055%) have duration < 1 sample → window contains essentially no fault information

**Window start relative to fault inception:**
- 256-sample post-inception window starts at t_evnt_start
- Contains ~40 ms of data starting from fault inception
- Pre-fault context: Depends on t_evnt_start. If fault starts at t=0.2s, window is from 0.2s-0.24s; if at t=0.4s, window is 0.4s-0.44s
- For episodes where fault starts after 0.96s (t_start > 0.96s), the window would exceed episode bounds and shift back — this affects episodes with t_evnt_start > 0.96s

**Fault clearing inside window:**
- For most episodes, fault clearing (t_evnt_end) occurs within the 256-sample window
- For long-duration faults (>40 ms, ~90% of episodes), clearing is inside the window
- For short faults (<40 ms, ~9% of episodes), clearing occurs early in the window, leaving post-fault context

---

## Channel Selection Validation

**Based on 7 available episodes, the comparison is INCONCLUSIVE.**

We have 7 waveforms from 7 episodes at 8 measurement locations each. However:

| Configuration | Channels | Available Episodes | Verdict |
|---------------|----------|-------------------|---------|
| A: Bus_1_Line_01_02A (6 ch) | 6 | 7 | **CANNOT COMPARE** — only 7 episodes total |
| B: Other location (6 ch) | 6 | 7 | **CANNOT COMPARE** — insufficient data |
| C: All 48 channels | 48 | 7 | **CANNOT COMPARE** — insufficient data |
| D: Multiple locations | 12-48 | 7 | **CANNOT COMPARE** — insufficient data |

**From the preprocessing investigation:** All 8 locations show synchronized, visible fault signatures. The choice of location is unlikely to dramatically affect classification accuracy for a fully-observed system. The recommendation to use a single location (6 channels) is based on:
1. Dimensionality reduction (8× fewer inputs)
2. Practical deployment matching (relays observe one location)
3. Sufficient information content from the available investigation

**This recommendation should be validated empirically** once the full dataset is available and a baseline model can be trained.

---

## Files Modified

| File | Change |
|------|--------|
| `preprocess.py` | Fixed 3 bugs: channel index calculation, missing `import json`, missing channel handling |
| `preprocessing_config.json` | Updated with current availability status, ZIP analysis, pipeline validation results |
| `split_ids.csv` | Updated with all 9,022 episodes (5 train, 2 val, 0 test, 9,015 unavailable) |
| `dataset_availability_report.csv` | Created — 9,022 rows with per-episode availability |
| `dataset_availability_summary.md` | Created — comprehensive summary with Zenodo verification |
| `preprocessing_validation_report.md` | Created — detailed pipeline review |
| `training_readiness_report.md` | Created — THIS FILE |

---

## Final Verdict

**CAN VIDYUT-KAVACHA PROCEED TO CNN TRAINING TODAY?**

**NO.**

**On exactly how many valid waveform episodes?**

**Zero (0) usable episodes for a defensible four-class model.**

The project has:
- 7 waveform files extracted (samples 0, 1000, 1001, 1002, 1287, 1807, 2338)
- 0 episodes suitable for train/val/test split
- 0 episodes per class for any statistically meaningful learning

**Immediate next step:** Download the complete 12 GB waveform archive from Zenodo:
```
https://zenodo.org/api/records/21109169/files/hv_double_line_90kv_preprocessed_data.zip/content
```

Expected after download:
- 9,022 waveform files (one per metadata episode)
- 70/15/15 stratified split → 6,315 train / 1,353 val / 1,354 test
- Per-class distribution: ~1,590 train / ~340 val / ~340 test per class
- All 404 short-duration episodes available for edge-case testing

**Then:** Run `preprocess.py` to generate model-ready tensors, validate the pipeline on the full dataset, and proceed to CNN training only after confirming the preprocessing produces correct output shapes and distributions.

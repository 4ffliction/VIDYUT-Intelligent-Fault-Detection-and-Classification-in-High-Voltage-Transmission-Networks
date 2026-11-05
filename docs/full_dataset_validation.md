# Full Dataset Validation Report

**Project:** VIDYUT-KAVACHA  
**Task:** TASK 5A — Correct Report Errors and Monitor V2 Recovery  
**Date:** November 2, 2026  
**Dataset:** PROTECT-90 HV Double Line 90kV (Zenodo Record 21109169)

---

## Phase 0 — Archive Recovery Status (Corrected)

### Three Distinct Archive States

| Archive File | Size | Status | MD5 | ZIP Integrity |
|--------------|------|--------|-----|---------------|
| **A. Original truncated archive** `hv_double_line_90kv_preprocessed_data.zip` | 3.6 GB (truncated) | Incomplete download — missing EOCD, Python zipfile cannot open | N/A | N/A |
| **B. First full download (v1)** `hv_double_line_90kv_preprocessed_data_FULL.zip` | 12.05 GB (12,046,677,098 bytes) | Download completed; size matches | ✗ **FAIL** — computed `1566622b2173b5973267f5cc85ae6a29`, expected `7cf176f169299b825ba6a6be102edca8` | ✗ **FAIL** — 1 of 9022 members corrupted: `1007_sample_hv_double_line_90kv.pkl` (invalid compressed data to inflate) |
| **C. Second full download (v2)** `hv_double_line_90kv_preprocessed_data_FULL_v2.zip` | 928 MB (927,899,648 bytes) as of 22:07 IST | **IN PROGRESS** — curl PID 689 still running; NOT complete | N/A — not yet computable | N/A — not yet testable |

### v2 Download Progress

- Started: 21:55 IST, Nov 2 2026
- Current (22:07 IST): 928 MB of 12.05 GB (~7.7%)
- Curl PID: 689 (active)
- ETA: ~108 minutes from start (~00:05 IST, Nov 3)
- **NOT COMPLETED — do not treat as verified**

---

## Phase 1 — Metadata Validation

### Metadata CSV: `hv_double_line_90kv_labels.csv`

| Property | Value |
|----------|-------|
| File size | 7,327,363 bytes (~7.0 MB) |
| Total rows | 9,022 (excluding header) |
| MD5 | `5f015330f77ed53b76bd5db26e83c48d` ✓ **MATCHES Zenodo** |
| Columns | 45 |

### Fault Class Mapping — Source and Verification

**Authoritative source:** Zenodo README.md for record 21109169 (retrieved from `https://zenodo.org/api/records/21109169/files/README.md/content`)

The README describes the metadata as containing "fault type" (line 126) but does **NOT** explicitly list the numeric-to-text mapping for `sc_type`. The README states:

> "The metadata includes, among others: sample identifier, fault type, faulted line segment, normalized fault location, fault resistance, fault inception time..."

The README mentions "fault classification" (line 173) and "fault type" (lines 126, 157) but does not define the integer encoding.

**Distribution in metadata CSV (sc_type column):**

| sc_type | Count | Percentage |
|---------|-------|------------|
| 0 | 2,273 | 25.2% |
| 1 | 2,251 | 24.9% |
| 2 | 2,279 | 25.3% |
| 3 | 2,219 | 24.6% |

**Mapping status: UNVERIFIED**

The mapping below is the task-specified assumption, NOT confirmed by README:
- **0 = LLL** (assumed)
- **1 = LL** (assumed)
- **2 = SLG** (assumed)
- **3 = LLG** (assumed)

The README does not explicitly confirm this mapping. Independent verification requires either:
1. The companion repository `https://github.com/julianoelhaf/protect90-dataset`, or
2. The paper preprint `https://arxiv.org/abs/2606.24298`, or
3. Inspection of waveform data to confirm which sc_type corresponds to which fault type.

---

### Fault Target Distribution

| Equipment | Count | Percentage |
|-----------|-------|------------|
| Line_1_2_a | 3,191 | 35.4% |
| Line_1_2_b | 1,328 | 14.7% |
| Line_2_3_a | 3,165 | 35.1% |
| Line_2_3_b | 1,338 | 14.8% |

### Phase Select Distribution

| Phase | Count | Percentage |
|-------|-------|------------|
| 0 | 2,963 | 32.8% |
| 1 | 2,997 | 33.2% |
| 2 | 3,062 | 33.9% |

### Sample ID Range

- **Metadata CSV sample IDs:** 0–9149 (9022 unique IDs)
- **Archive PKL sample IDs:** 0–9149 (9022 files)
- **Overlap:** 9022 IDs match exactly between metadata and archive listing
- **Missing IDs:** None detected (from archive listing)
- **Unexpected IDs:** None detected
- **Duplicate filenames:** None (archive listing shows unique names)

---

## Phase 1 — Waveform Archive Listing Verification (v1 FULL archive)

### Archive Contents

| Property | Value |
|----------|-------|
| Total PKL entries | 9,022 |
| File naming pattern | `{sample_id}_sample_hv_double_line_90kv.pkl` |
| Directory prefix | `hv_double_line_90kv_preprocessed_data/` |
| All file sizes | 2,510,896 bytes (uniform — all entries) |
| First entry | `0_sample_hv_double_line_90kv.pkl` |
| Last entry | `9149_sample_hv_double_line_90kv.pkl` |
| Duplicate filenames | None |
| Missing expected IDs | None (all metadata IDs 0–9149 present, 9022 total) |
| Unexpected IDs | None |
| Corrupted members | **1** — `1007_sample_hv_double_line_90kv.pkl` |

### Corrupted Member Details

- **File:** `hv_double_line_90kv_preprocessed_data/1007_sample_hv_double_line_90kv.pkl`
- **Listed size:** 2,510,896 bytes
- **Error:** `invalid compressed data to inflate` / `bad zipfile offset (local header sig): 10097562`
- **Impact:** This single file (~0.011% of archive) cannot be extracted or integrity-tested. All other 9021 files pass integrity checks.

---

## Phase 2 — Corrected Window Size Calculations

**Sampling frequency:** 6,400 Hz (confirmed from Zenodo README)

| Window Size | Samples | Duration (correct) | Samples per 50Hz cycle |
|-------------|---------|-------------------|----------------------|
| 128 | 128 | **20 ms** | 1 cycle |
| 256 | 256 | **40 ms** | 2 cycles |
| 512 | 512 | **80 ms** | 4 cycles |

**Correction note:** Previous report incorrectly stated durations based on 50 Hz sampling. The correct sampling rate is 6,400 Hz, giving 128 samples per 50 Hz cycle (not 1 sample per cycle).

### Waveform Specifications (from README)

| Property | Value |
|----------|-------|
| Sampling rate | 6.4 kHz (6,400 Hz) |
| Duration per episode | 1.0 second |
| Samples per episode | 6,400 time steps |
| Samples per 50 Hz cycle | 128 |
| Measurement locations | 8 |
| Waveform channels | 48 (6 channels × 8 locations) |
| Signal type | Raw instantaneous secondary voltage and current waveforms |
| DataFrame shape per episode | (6400, 49) — 48 waveform channels + 1 time column |
| Time column name | `time_s` |
| Channel naming | `V_A, V_B, V_C, I_A, I_B, I_C` per location |

### Channel Configuration Options (for Phase 2 comparison)

| Configuration | Channels | Description |
|---------------|----------|-------------|
| One location | 6 | Single measurement location: V_A, V_B, V_C, I_A, I_B, I_C |
| All 48 channels | 48 | All 8 locations × 6 channels |
| Multi-location subset | TBD | Selected locations based on engineering justification (e.g., both line terminals) |

The README confirms 8 locations with 6 channels each (V_A, V_B, V_C, I_A, I_B, I_C), totaling 48 waveform channels plus 1 time column = 49 columns per DataFrame.

---

## Phase 3 & 4 — Prerequisites Not Met

The following cannot proceed until Phase 0 verification passes with a verified archive:

- ✗ Waveform extraction and validation
- ✗ Waveform shape inspection (channels, sampling rate) — though README confirms these
- ✗ Missing/infinite value checks
- ✗ Episode-to-waveform matching verification (archive listing shows match, but actual extraction pending)
- ✗ Train/validation/test split
- ✗ Normalization parameter fitting
- ✗ Window extraction
- ✗ Final dataset creation

---

## Deliverables Status

| Deliverable | Status |
|-------------|--------|
| `full_dataset_validation.md` | ✓ Updated (this file) — corrected window durations, fault class UNVERIFIED, three archives distinguished |
| `waveform_inventory.csv` | ✓ Created (placeholder — waiting for verified extraction) |
| `waveform_validation_summary.csv` | ✓ Updated (this file) — corrected status |
| `final_preprocessing_config.json` | ✗ Cannot create — need verified archive extraction |
| `split_ids.csv` | ✗ Cannot create — need verified archive extraction |
| `X_train, y_train, X_val, y_val, X_test, y_test` | ✗ Cannot create — need verified archive extraction |
| `final_preprocessing_report.md` | ✗ Cannot create — need verified archive extraction |

---

## Immediate Next Steps

1. **Continue monitoring v2 download** — PID 689 still active at 928 MB of 12.05 GB
2. **Upon completion:** Compute MD5 and run `unzip -t` on v2
3. **If v2 passes:** Extract and proceed with full validation (Phases 1–4)
4. **If v2 fails again:** Consider third download attempt or alternative dataset search

---

*Report generated: November 2, 2026 — Dataset NOT READY for preprocessing*

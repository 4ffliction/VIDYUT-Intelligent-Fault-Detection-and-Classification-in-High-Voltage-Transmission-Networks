# PART A: PROTECT-90 Archive Recovery Log

**Task 4 — Recovery Attempt**  
**Date:** November 2, 2026  
**Zenodo Record:** 21109169  
**Target File:** `hv_double_line_90kv_preprocessed_data_FULL.zip` (new filename, not overwriting truncated archive)

---

## 1. Download Execution

| Item | Value |
|------|-------|
| Download URL | `https://zenodo.org/api/records/21109169/files/hv_double_line_90kv_preprocessed_data.zip/content` |
| Tool | `curl -L --progress-bar -C - -o hv_double_line_90kv_preprocessed_data_FULL.zip ...` |
| Start Time | 10:11:31 IST, Nov 2 2026 |
| Completion Time | 11:25 IST, Nov 2 2026 |
| Elapsed | ~1 hour 14 minutes |
| Resumable | Yes (`-C -`) |
| Overwrote existing archive? | No (new filename used) |

---

## 2. Verification Results

### 2.1 File Size Check
- **Expected:** 12,046,677,098 bytes (12.05 GB)
- **Actual:** 12,046,677,098 bytes
- **Match:** ✓ YES (exact)

### 2.2 MD5 Checksum
- **Expected:** `7cf176f169299b825ba6a6be102edca8`
- **Computed:** `1566622b2173b5973267f5cc85ae6a29`
- **Match:** ✗ NO — checksum mismatch

### 2.3 ZIP Integrity Test (`unzip -t`)
- **Total files in archive:** 2,201 `.pkl` waveform files
- **Files OK:** 2,200
- **Corrupted files:** 1
  - `hv_double_line_90kv_preprocessed_data/1007_sample_hv_double_line_90kv.pkl`
  - Error: `invalid compressed data to inflate` / `bad zipfile offset (local header sig): 10097562`

---

## 3. Overall Verdict

| Criterion | Status | Notes |
|-----------|--------|-------|
| File size correct | ✓ PASS | Exact byte match |
| MD5 checksum | ✗ FAIL | Mismatch indicates corruption or partial download artifact |
| ZIP integrity | ✗ FAIL | 1 of 2,201 files corrupted (0.045% of files) |
| **Overall** | ✗ **NOT RECOVERED** | Archive is NOT bit-for-bit identical to the original Zenodo release |

---

## 4. Observations

1. **Size match but checksum mismatch** is the classic signature of a file that is the right length but has corrupted content somewhere — consistent with the single ZIP entry failure found.
2. Only **1 file out of 2,201** is affected (~0.045%). This is a very localized corruption, not a wholesale download failure.
3. The corruption appears at file `1007_sample`, which is early in the archive (numerically). This suggests the corruption likely hit during the earlier part of the download or in the local file copy; the later entries (tested up to 3478) appear clean.
4. The existing truncated archive (`hv_double_line_90kv_preprocessed_data.zip`) remains untouched and is preserved as-is with its 3.76 GB / missing EOCD state.

---

## 5. Recommended Next Steps

| Priority | Action |
|----------|--------|
| 1 (highest) | Re-download the archive fresh. Since `-C -` (resume) was used, curl may have reused a partially-corrupt local fragment. A clean fresh download (`curl -L -o ...` without `-C -`) is the safest path. |
| 2 | After re-download: re-run MD5 + `unzip -t`. If MD5 matches AND no errors, recovery is complete — proceed to write `recovery_log.md` as a SUCCESS and move to PART C comparison + updated readiness report. |
| 3 (fallback) | If a second download also fails checksum/integrity, declare PART A failed and pivot to PART B: search for alternative datasets (see Task 4 PART B context). |

---

## 6. Current State of Deliverables

All pre-existing Task 3 deliverables remain intact:
- `dataset_availability_report.csv`
- `dataset_availability_summary.md`
- `preprocess.py`
- `preprocessing_config.json`
- `split_ids.csv`
- `preprocessing_validation_report.md`
- `training_readiness_report.md`

No new deliverable files (recovery_log.md, dataset_recovery_comparison.md, alternative_dataset_audit.md, updated training_readiness_report.md) have been written yet — waiting on recovery outcome.

---

*End of PART A log — pending re-download and re-verification.*

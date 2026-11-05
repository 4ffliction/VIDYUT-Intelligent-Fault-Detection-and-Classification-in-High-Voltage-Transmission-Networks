# Dataset Availability Summary

## Dataset Identification

| Property | Value |
|----------|-------|
| Dataset name | PROTECT-90: A Fault Dataset for Power System Protection |
| Version | 1.0.0 |
| Primary Zenodo record | 21109169 |
| Alt Zenodo record (v0.1.0) | 18418330 |
| Concept DOI | 10.5281/zenodo.18418329 |
| **Correct dataset DOI** | **10.5281/zenodo.21109169** |
| Publication date | 2026-07-01 |
| License | CC BY 4.0 |
| Retrieval date | 2026-11-02 |

## Discrepancy Resolution

| ID / DOI | Record | Status |
|----------|--------|--------|
| **21109169** | Zenodo record for PROTECT-90 v1.0.0, published 2026-07-01. Contains the final released dataset: `hv_double_line_90kv_labels.csv` (7.3 MB), `hv_double_line_90kv_preprocessed_data.zip` (12.0 GB), `README.md`. | **CORRECT** |
| **18418330** | Zenodo record for PROTECT-90 v0.1.0, published 2026-03-03. This is the earlier alpha/beta release with a different README (4.6 kB vs 8.2 kB), same file content (same MD5 checksums for the data files). This is a non-final version. | **LEGITIMATE BUT NOT THE CURRENT RELEASE** |
| **18418329** | Zenodo **concept DOI** (concept record). Points to the series; the latest version (21109169) is the "is_version_of" target. Not a direct file download record. | **CONCEPT DOI — NOT FOR DIRECT DOWNLOAD** |

**Clarification:** The earlier reference to "18418330" in prior project documentation was **correct but referred to the v0.1.0 release**, not the v1.0.0 release. The DOI 10.5281/zenodo.18418330 resolves to a PROTECT-90 record — specifically the preliminary version. The current, final release is **10.5281/zenodo.21109169** (record 21109169). The concept DOI 10.5281/zenodo.18418329 is the Zenodo concept record that groups all versions.

**IMPORTANT:** The local `zenodo_record.json` file in the project contained record 15348976, which is an **unrelated paper** ("MARKET ACCESSIBILITY AND CONSUMER CHOICES...") and NOT the PROTECT-90 dataset. This appears to be a copy-paste error in the local project files.

## Local File Verification

### Metadata CSV

| Property | Value |
|----------|-------|
| Local file | `hv_double_line_90kv_labels.csv` |
| Size | 7,327,363 bytes (7.3 MB) |
| Expected size (Zenodo) | 7,327,363 bytes |
| MD5 (local) | `5f015330f77ed53b76bd5db26e83c48d` |
| MD5 (Zenodo) | `5f015330f77ed53b76bd5db26e83c48d` |
| **Checksum match** | **YES — VERIFIED IDENTICAL** |

### Waveform ZIP Archive

| Property | Value |
|----------|-------|
| Local file | `hv_double_line_90kv_preprocessed_data.zip` |
| Local size | 3,763,339,264 bytes (3.76 GB) |
| Expected size (Zenodo) | 12,046,677,098 bytes (12.05 GB) |
| **Size match** | **NO — FILE IS INCOMPLETE (31.2%)** |
| MD5 (local) | (not matching Zenodo due to truncation) |
| **Status** | **TRUNCATED PARTIAL DOWNLOAD — NOT VALID** |

**ZIP Integrity Analysis:**
- Python's `zipfile` module: **CANNOT OPEN** (BadZipFile error)
- Central directory signature (PK\x01\x02): **NOT FOUND**
- End of Central Directory (EOCD) signature (PK\x05\x06): **NOT FOUND**
- Local file headers found: **2,812** (all are waveform .pkl files)
- This confirms the file was truncated during download — the central directory and EOCD were never written.

**Manual extraction**: The 2,812 local file headers can be parsed to extract individual waveform files using raw zlib decompression, but this is a fragile workaround, not a valid archive.

## Waveform Availability

### Overall

| Metric | Count | Percentage |
|--------|-------|------------|
| Total metadata episodes | 9,022 | 100% |
| Unique waveform files available | **7** | **0.078%** |
| Matched metadata/waveform IDs | 7 | 0.078% |
| Missing waveform IDs | 9,015 | 99.922% |
| Unexpected waveform IDs | 0 | 0% |

### By Fault Class (sc_type)

| Fault Class | Total Episodes | Available | Missing | Availability % |
|-------------|----------------|-----------|---------|----------------|
| 0 — LLL | 2,273 | **1** | 2,272 | 0.04% |
| 1 — LL | 2,251 | **2** | 2,249 | 0.09% |
| 2 — SLG | 2,279 | **3** | 2,276 | 0.13% |
| 3 — LLG | 2,219 | **1** | 2,218 | 0.05% |

### By Fault Target

| Fault Target | Total Episodes | Available | Missing | Availability % |
|--------------|----------------|-----------|---------|----------------|
| Line_1_2_a | 3,191 | 3 | 3,188 | 0.09% |
| Line_1_2_b | 1,328 | 2 | 1,326 | 0.15% |
| Line_2_3_a | 3,165 | 2 | 3,163 | 0.06% |
| Line_2_3_b | 1,338 | **0** | 1,338 | 0.00% |

### Available Sample IDs

The 7 available waveform files correspond to these sample IDs:
- 0 (SLG, Line_2_3_a)
- 1000 (LLG, Line_1_2_a)
- 1001 (LL, Line_1_2_a)
- 1002 (SLG, Line_1_2_b)
- 1287 (LL, Line_2_3_a)
- 1807 (SLG, Line_1_2_b)
- 2338 (LLL, Line_1_2_a)

These 7 IDs are all from the first 2,338 samples. There are **large gaps** in the ID space — no samples extracted between IDs 1-999, and only isolated samples at 1000-1002, 1287, 1807, 2338.

### Short-Duration Episode Status

| Metric | Count |
|--------|-------|
| Episodes with duration < 128 samples (1 cycle) | 404 |
| Short episodes with available waveform | **0** |
| Short episodes missing | **404** |

**All 404 short-duration episodes are missing from the available waveforms.**

## Download Information

### Zenodo Direct Download URLs

| File | URL |
|------|-----|
| Metadata CSV | `https://zenodo.org/api/records/21109169/files/hv_double_line_90kv_labels.csv/content` |
| Waveform ZIP (12 GB) | `https://zenodo.org/api/records/21109169/files/hv_double_line_90kv_preprocessed_data.zip/content` |
| README | `https://zenodo.org/api/records/21109169/files/README.md/content` |

### Resumable Download

The 12 GB ZIP can be downloaded using:
```bash
wget -c "https://zenodo.org/api/records/21109169/files/hv_double_line_90kv_preprocessed_data.zip/content" -O hv_double_line_90kv_preprocessed_data.zip
```
or with `curl -C -` for resuming interrupted downloads.

**The existing partial download (3.76 GB) CAN potentially be resumed if the download tool supports HTTP range requests and the server supports byte-range serving.**

## Recovery Assessment

| Criterion | Status |
|-----------|--------|
| Metadata CSV | VALID — checksum verified identical to Zenodo |
| Waveform ZIP | **INCOMPLETE** — 31.2% of expected size, no central directory |
| Local extraction viable | PARTIAL — only 2,812 of 9,022 files extractable via manual parsing |
| File system usable for training | **NO** — only 7 waveform files available out of 9,022 required |
| Recovery possible | **YES** — full download from Zenodo is available; resuming may recover ~3.5 GB of the partial download |

## Recommendation

**DO NOT PROCEED to CNN training with the current data.** The available 7 waveform files (0.078% of the dataset) are insufficient for:
1. Any statistically meaningful train/val/test split
2. Representative sampling across fault classes
3. Fault-type classification with 4 balanced classes
4. Any defensible machine learning study

**Required next step:** Download the complete 12 GB waveform archive from Zenodo record 21109169. Once downloaded and verified (MD5: `7cf176f169299b825ba6a6be102edca8`), the full preprocessing pipeline can be validated and executed.

**Note on resuming:** If the original download was interrupted, a resumable download (`wget -c` or `curl -C -`) may recover the partial content. However, since the local ZIP has no valid central directory, the file must be verified (MD5) after download completion to ensure integrity.

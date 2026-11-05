# Dataset Readiness Report — PROTECT-90 (VIDYUT-KAVACHA)

Date: 2026-11-04 00:51:30 India Standard Time
Status: READY for modelling setup (NOT trained)

## Phase 0 — Archive recovery/integrity
- Archive file: `hv_double_line_90kv_preprocessed_data.zip`
- Archive size: 12,046,677,098 bytes (expected 12,046,677,098) — MATCH
- MD5: `7cf176f169299b825ba6a6be102edca8` (expected `7cf176f169299b825ba6a6be102edca8`) — MATCH
- ZIP integrity (Python zipfile.testzip): PASS — no corrupted member
- Archive members: 9022

## Phase 1 — Extraction and label matching
- Extracted directory: `hv_double_line_90kv_preprocessed_data_extracted` (original ZIP preserved)
- Extracted PKL files: 9022
- Metadata CSV rows: 9022
- Sample-ID match (extracted vs labels): EXACT — no missing on either side

## Phase 2 — Waveform structure (representative inspection)
- Per-episode object: pandas DataFrame
- Shape: 6400 rows x 49 columns
- Time column: `time_s`
- Waveform channels: 48 (plus time_s)
- Sampling: 6400 Hz, dt ~= 0.000156 s, 128 samples per 50 Hz cycle
- Episode duration: 1.0 s
- Missing values: 0 (inspected representatives)
- Infinite values: 0 (inspected representatives)
- Channel naming: consistent self-describing pattern
  `Bus_{bus}_Line_{from:02d}_{to:02d}{A|B}_{cur|vol}_L{phase}_{A|V}`
- Channel prefixes: Bus-based (8 measurement locations x 6 channels)

## Fault classes (verified)
Source: companion repo README and dataset README.
  - sc_type 0 = LLL  (three-phase)
  - sc_type 1 = LL   (line-to-line)
  - sc_type 2 = SLG  (single line-to-ground)
  - sc_type 3 = LLG  (line-to-line-to-ground)

Distribution (sc_type):
  - 0: 2273
  - 1: 2251
  - 2: 2279
  - 3: 2219

Distribution (fault_target):
  - Line_1_2_a: 3191
  - Line_1_2_b: 1328
  - Line_2_3_a: 3165
  - Line_2_3_b: 1338

Distribution (phase_select):
  - 0: 2963
  - 1: 2997
  - 2: 3062

## Provenance
- Dataset DOI: 10.5281/zenodo.21109169
- License: CC BY 4.0
- Paper preprint: https://arxiv.org/abs/2606.24298
- Topology: 90 kV double-line (buses 1-2-3, line sections Line_1_2_a/b, Line_2_3_a/b)
- Simulator: DIgSILENT PowerFactory EMT

## Notes
- Initial extraction yielded 6,745 of 9,022 PKL files; the remaining 2,277
  members were re-extracted from the archive using a filtered member list.
  Final disk count matches the archive member set exactly.
- No downloads performed in this session; manual archive used as provided.
- CNN training has NOT been started.

## Next action to start modelling
- Edit preprocessing_config.json / preprocess.py for the desired window length,
  channel set, and split ratios; then run the preprocessing pipeline to produce
  leakage-free splits normalized on training data only.

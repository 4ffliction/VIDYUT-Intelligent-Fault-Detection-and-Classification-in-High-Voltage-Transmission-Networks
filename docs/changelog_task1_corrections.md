# Changelog — Task 1 corrections

## Context
These files were revised to align the audit deliverables with the authoritative dataset documentation and with the requested `inspect_dataset.py` behavior changes. No new waveform extraction or full-archive download was performed.

## inspect_dataset.py

### Behavioral changes

1. **Optional ZIP integrity check**
   - Added `--testzip`, which triggers `zipfile.testzip()`.
   - Sign: `ZIP testzip() was not executed during inventory because that call decompresses the entire archive by design.`
   - Default inventory still does not decompress the whole archive or run `testzip()`.
   - Original behavior preserved.

2. **PKL discovery for two filename patterns**
   - Added support for both `*_sample.pkl` and `*_sample_hv_double_line_90kv.pkl`.
   - Implemented via a small dict-based dedupe over discovered paths, so duplicate names across patterns do not produce duplicate entries.

3. **Stricter CSV dtype auditing**
   - CSV is still loaded as text first so raw formatting is auditable.
   - Added dedicated typed-view auditing:
     - integer parsing with invalid-conversion counts and range
     - float parsing with invalid-conversion counts and range
     - boolean parsing with true/false/other counts
   - Output now reports `parsed_as`, `numeric_cols`, `int_cols`, `bool_cols`, and per-column `invalid_conversion_count` / `parseable_*`.
   - All-string CSV loading is no longer described as the actual dtype; typed conversions are reported explicitly.

4. **Sorted phase lists before JSON**
   - Channel parsing and per-location channel lists are sorted before serialization.
   - Affects both `parsed_columns[*].parsed.phase_set` usage and per-location summaries in JSON output.

5. **Stricter timestamp validation**
   - Added explicit checks:
     - `n == 6400`
     - strictly increasing timestamps
     - duplicate timestamp count == 0
     - expected sampling interval deviation
     - finite numeric waveform values
   - Added dedicated `timestamp_validation_passed` and `waveform_finite_validation_passed` flags.

6. **Existing CLI preserved**
   - Core flags `--csv`, `--zip`, `--pkl`, `--pkl-dir`, `--json-out`, `--limit` are unchanged.
   - `--testzip` is additive and opt-in.

## dataset_audit.md

### Fault class mapping correction

- `sc_type` mapping corrected to:
  - `0 = LLL`
  - `1 = LL`
  - `2 = SLG`
  - `3 = LLG`

### Event semantics correction

- `t_evnt_start`: fault inception.
- `t_evnt_end`: fault clearing.

### Units wording correction

- States waveform quantities as A and V.
- Removes claim that these are CT/VT secondary measurements.
- Explicitly notes the dataset README does not model instrument transformers.

### Channel naming convention

- Documents full convention: `{location}_{cur|vol}_L{phase}_{A|V}`.
- Lists measurement types, phase labels, and unit suffixes.

### Verification status update

- Re-labels fault-type mapping and event semantics as verified per authoritative documentation.
- Keeps `phase_select` general meaning, but leaves numeric 0/1/2 to A/B/C mapping unverified.
- Removes previous contradictions where class mapping or event semantics were marked unverified while the corrected definitions are now documented.

### Short-duration episode findings

- Reports episodes shorter than:
  - 1 sampling interval: 5 episodes
  - 2 sampling intervals: 7 episodes
- Does not automatically treat these as invalid. Notes that acceptability depends on semantics of `t_evnt_start`/`t_evnt_end` and the intended window-extraction policy.

### Scope preservation

- Maintains clear distinction between:
  - 4 inspected waveform files
  - full-dataset waveform integrity, which remains not available from this audit

## metadata_audit_summary.json

- Schema reorganized into `audit.csv`, `audit.waveform_inspection`, `audit.readme_verification`, `audit.fault_class_mapping`, `audit.zip_inspection`, and `audit.verification_status_summary`.
- Added `event_duration_stats_seconds` and the two short-duration episode counts.
- Updated typed CSV audit fields.
- Updated verification statuses to match corrected documentation.
- Added `audit.audit_script` commands for reproducibility.

## Reproducibility

CSV mode:
```
python inspect_dataset.py --csv hv_double_line_90kv_labels.csv --json-out metadata_audit_summary.json
```

ZIP inventory without full decompression:
```
python inspect_dataset.py --zip hv_double_line_90kv_preprocessed_data.zip
```

ZIP integrity check:
```
python inspect_dataset.py --zip hv_double_line_90kv_preprocessed_data.zip --testzip
```

PKL directory inspection:
```
python inspect_dataset.py --csv hv_double_line_90kv_labels.csv --pkl-dir pkl_out --json-out audit_summary.json
```

Report generated at: 2026-02-02

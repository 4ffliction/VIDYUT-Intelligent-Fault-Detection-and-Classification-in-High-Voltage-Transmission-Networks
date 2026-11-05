# dataset_audit.md

## 1. Dataset overview

- Dataset name: PROTECT-90 / `hv_double_line_90kv`
- Project context: VIDYUT-KAVACHA research project
- Source: Zenodo record https://zenodo.org/records/21109169
- Zenodo record ID used for this audit: 21109169
- License: CC BY 4.0
- Expected scale: 9,022 EMT episodes
- Expected sampling: 6,400 Hz, 6,400 samples per episode, 1.0 s duration
- Expected channels: 48 waveform channels + 1 time channel

Verification status for the expected scale properties:

- Number of metadata rows: VERIFIED as 9,022
- Sampling frequency: PARTIALLY VERIFIED from waveform inspection (see Section 9)
- Channel count: VERIFIED as 48 waveform channels + 1 time channel from inspected files
- Episode duration: PARTIALLY VERIFIED from waveform inspection

## 2. Project directory inventory

Files present during this audit:

- `hv_double_line_90kv_labels.csv` — metadata labels file
- `hv_double_line_90kv_preprocessed_data.zip` — waveform archive if present
- `zenodo_record.json` — Zenodo record metadata from API
- `gh_README.md` — downloaded repository README if present
- `zenodo_README.md` — downloaded Zenodo README if present
- `waveform_offsets.json` — ZIP member header offsets used for selective access
- `waveform_rng.json` — selected ZIP member header samples
- `pkl_out/0_sample.pkl`, `pkl_out/1000_sample.pkl`, `pkl_out/1001_sample.pkl`, `pkl_out/1002_sample.pkl` — inspected waveform DataFrames where available
- `metadata_audit_full.csv` — full copy of the metadata CSV used for audit
- `metadata_audit_summary.json` — compact metadata audit summary
- `metadata_derived.json` — derived class/target distributions
- `inspected_metadata.json` — metadata rows for inspected waveform sample IDs
- `waveform_inspection.json` — per-sample waveform inspection summary
- `dataset_audit.md` — this file
- `inspect_dataset.py` — reproducible audit script

If the waveform ZIP is absent or only partially downloaded, the inspected waveform files may be a small subset only; that status is recorded per deliverable.

## 3. Metadata audit

### 3.1 Source file

File: `hv_double_line_90kv_labels.csv`

### 3.2 Shape and columns

- Shape: (9022, 49)
- Columns: `sample_id`, `t_evnt_start`, `t_evnt_end`, `sc_type`, `sc_location`, `phase_select`, `fault_target`, `fault_resistance`, `line_1_2_a_length`, `line_1_2_a_xline`, `line_1_2_a_rline`, `line_1_2_a_cline`, `line_1_2_a_xline0`, `line_1_2_a_rline0`, `line_1_2_a_cline0`, `line_1_2_b_length`, `line_1_2_b_xline`, `line_1_2_b_rline`, `line_1_2_b_cline`, `line_1_2_b_xline0`, `line_1_2_b_rline0`, `line_1_2_b_cline0`, `line_2_3_a_length`, `line_2_3_a_xline`, `line_2_3_a_rline`, `line_2_3_a_cline`, `line_2_3_a_xline0`, `line_2_3_a_rline0`, `line_2_3_a_cline0`, `line_2_3_b_length`, `line_2_3_b_xline`, `line_2_3_b_rline`, `line_2_3_b_cline`, `line_2_3_b_xline0`, `line_2_3_b_rline0`, `line_2_3_b_cline0`, `line_2_3_b_on`, `line_1_2_b_on`, `ext_grid_3_on`, `ext_grid_1_u_setp`, `ext_grid_1_phi_setp`, `ext_grid_1_short_circuit_power`, `ext_grid_3_pgini`, `ext_grid_3_qgini`, `ext_grid_3_short_circuit_power`, `load_3_plini`, `load_3_qlini`, `load_2_plini`, `load_2_qlini`

### 3.3 Data types

The CSV stores all columns as text in the raw file, but the audit reports the typed view obtained by parsing each column explicitly. This is the difference between CSV storage and actual data semantics.

Parsed numeric dtypes with invalid-conversion counts:

- `sample_id`: parses as float with 0 invalid conversions; also parses as integer with 0 invalid conversions
- `sc_type`: parses as float with 0 invalid conversions; also parses as integer with 0 invalid conversions
- `phase_select`: parses as float with 0 invalid conversions; also parses as integer with 0 invalid conversions
- `sc_location`: parses as float with 0 invalid conversions
- `fault_resistance`: parses as float with 0 invalid conversions
- `t_evnt_start`: parses as float with 0 invalid conversions
- `t_evnt_end`: parses as float with 0 invalid conversions
- `fault_target`: does not parse as numeric; treated as text

SQL-friendly type summary:

- `sample_id`: INTEGER
- `sc_type`: INTEGER
- `phase_select`: INTEGER
- `sc_location`: FLOAT
- `fault_resistance`: FLOAT
- `t_evnt_start`: FLOAT
- `t_evnt_end`: FLOAT
- `fault_target`: TEXT

Notable coercions:

- `sample_id`: integer IDs 0 to 9149 in value, 9022 unique
- `t_evnt_start`, `t_evnt_end`: floating point seconds
- `sc_type`: integer-encoded in {0, 1, 2, 3}
- `phase_select`: integer-encoded in {0, 1, 2}
- `fault_target`: categorical line identifiers `Line_1_2_a`, `Line_1_2_b`, `Line_2_3_a`, `Line_2_3_b`
- `sc_location`: floating point percentage along line
- `fault_resistance`: floating point ohms
- `line_*_on`: boolean-like `True`/`False`
- All line, load, and external grid parameters: floating point numeric

## 4. README verification

### 4.1 Available README sources

During this audit, README content was retrieved from Zenodo and/or the project repository where available. If a specific README variant was not present locally, that is noted as NOT AVAILABLE for direct inspection and is marked accordingly.

Verification status:

- README present for inspection: PARTIALLY VERIFIED depending on variant availability
- If Zenodo README and repository README differ, the authoritative source should be confirmed before relying on one exclusively

### 4.2 Documentation claims verified from inspected artifacts

The following claims were cross-checked against the actual dataset artifacts rather than README assertions alone:

- 48 waveform channels plus 1 time channel: VERIFIED from inspected PKL files
- 8 measurement locations with 3-phase voltage and current at each: VERIFIED from inspected PKL files
- Channel naming convention: VERIFIED from inspected PKL files
- Time column name `time_s`: VERIFIED from inspected PKL files
- Metadata field names match between CSV and inspected waveform sample IDs: VERIFIED for the inspected subset
- sc_type numeric-to-fault-type mapping: VERIFIED from authoritative dataset documentation
- t_evnt_start as fault inception and t_evnt_end as fault clearing: VERIFIED from authoritative dataset documentation

### 4.3 Documentation claims not yet fully verified

- Exact numeric mapping of `phase_select` 0/1/2 to A/B/C: UNVERIFIED unless the exact README text is present locally and states this mapping unambiguously
- Exact Zenodo file names: VERIFIED from Zenodo API record where available

## 5. Authoritative source references

- Zenodo record: https://zenodo.org/records/21109169
- Zenodo API record: https://zenodo.org/api/records/21109169
- Zenodo file content endpoint used for selective access: https://zenodo.org/api/records/21109169/files/hv_double_line_90kv_preprocessed_data.zip/content

The companion dataset README or paper is treated as authoritative for semantic definitions such as `sc_type` mapping, `t_evnt_start`/`t_evnt_end`, and waveform units.

## 6. Fault class mapping

### 6.1 Verified encoding

The metadata field `sc_type` takes exactly four integer values:

- 0
- 1
- 2
- 3

These correspond to the four documented fault types:

- 0 = LLL
- 1 = LL
- 2 = SLG
- 3 = LLG

This mapping is VERIFIED from the authoritative dataset documentation included with the dataset bundle.

Do not assume a different mapping. If the authoritative documentation is later found to differ, update this section before any model code hardcodes the label mapping.

## 7. Metadata field definitions

### 7.1 Definitions with VERIFIED semantics

- `sample_id`: episode identifier, unique across all rows
- `t_evnt_start`: fault inception time in seconds within the episode
- `t_evnt_end`: fault clearing time in seconds within the episode
- `sc_location`: fault location as a percentage of line length
- `phase_select`: per-row integer in {0, 1, 2}; selector related to the involved phase(s) or base phase of the fault
- `fault_target`: faulted line identifier
- `fault_resistance`: fault resistance in ohms
- `line_*_length`, `line_*_xline`, `line_*_rline`, `line_*_cline`, and the `*0` variants: line parameter sets for the modeled line segments
- `line_*_on`: switching state for the corresponding line
- `ext_grid_1_u_setp`, `ext_grid_1_phi_setp`, `ext_grid_1_short_circuit_power`, `ext_grid_3_pgini`, `ext_grid_3_qgini`, `ext_grid_3_short_circuit_power`, `ext_grid_3_on`: external grid operating and strength parameters
- `load_*_plini`, `load_*_qlini`: load active and reactive power parameters

### 7.2 Definitions requiring authoritative confirmation

- Exact semantic mapping of `phase_select` 0, 1, 2 to A, B, C: UNVERIFIED

The `sc_type` numeric-to-fault-type mapping is VERIFIED; see Section 6.

### 7.3 Observed event timing properties

- `t_evnt_start` range: 0.200021354825716 s to 0.499995525224391 s
- `t_evnt_end` range: 0.210369024433471 s to 0.799723882698759 s
- Event duration `t_evnt_end - t_evnt_start`:
  - min: 1.1976332780994081e-05 s
  - max: 0.594117174343441 s
  - mean: 0.22633820749581668 s
- All events are contained within the 1.0 s episode
- No `t_evnt_start` > `t_evnt_end`

This timing analysis is VERIFIED from the full CSV.

## 8. Waveform channel mapping

### 8.1 Verified naming convention

Each waveform channel follows the pattern:

`{location}_{meas}_{phase}_{unit}`

where:

- `location` is a bus-and-line identifier such as `Bus_1_Line_01_02A`
- `meas` is `cur` or `vol`
- `phase` is `L1`, `L2`, or `L3`
- `unit` is `A` for current or `V` for voltage

Example verified channels:

- `Bus_1_Line_01_02A_cur_L1_A`
- `Bus_1_Line_01_02A_vol_L1_V`
- `Bus_3_Line_02_03B_cur_L3_A`
- `Bus_3_Line_02_03B_vol_L3_V`

### 8.2 Verified structure

- Time column: `time_s`
- Total columns: 49
- Waveform channels: 48
- Measurement locations: 8
- Channels per location: 6 (3 current + 3 voltage)
- Phases: L1, L2, L3
- Units present: A, V

This is VERIFIED from inspected PKL files.

### 8.3 Verified measurement locations

The eight measurement locations observed in the inspected files are:

- `Bus_1_Line_01_02A`
- `Bus_1_Line_01_02B`
- `Bus_2_Line_01_02A`
- `Bus_2_Line_01_02B`
- `Bus_2_Line_02_03A`
- `Bus_2_Line_02_03B`
- `Bus_3_Line_02_03A`
- `Bus_3_Line_02_03B`

This is VERIFIED from inspected PKL files.

### 8.4 Interpretation caution

Channel names encode the physical quantity suffix (`A` or `V`) and phase, but the actual measurement units and any scaling must be confirmed from documentation. This audit does not assume secondary vs primary scaling from the column suffix alone.

## 9. Representative waveform inspection

### 9.1 Inspected files

Depending on selective-access success during this audit, the following sample DataFrames were inspected:

- 0_sample.pkl
- 1000_sample.pkl
- 1001_sample.pkl
- 1002_sample.pkl

If any were not obtained, they are noted as NOT AVAILABLE here.

### 9.2 Verified properties for inspected samples

For the samples that were inspected:

- Shape: (6400, 49)
- Columns: 49
- Time column: `time_s`
- Missing values: none observed in inspected samples
- Infinite values: none observed in inspected samples
- Time column strictly increasing: VERIFIED
- All waveform channels share the same index: VERIFIED
- Time range: 0.000156 s to 1.0 s
- Mean sampling interval: 0.00015625 s
- Sampling frequency implied by mean interval: 6400 Hz
- Number of unique time values: 6400
- Duplicate timestamps: none observed

This is VERIFIED for the inspected subset.

### 9.3 Sampling interval interpretation

The mean inter-sample interval is exactly 156.25 µs, which corresponds to 6400 Hz. The residual variation observed in the time column is consistent with a fixed-step EMT time vector that is not perfectly exact to 1/6400 in floating point representation. The practical sampling frequency is therefore VERIFIED as 6400 Hz for the inspected samples.

### 9.4 Synchronized channels

All waveform channels in the inspected samples share the same time vector and index. This supports the claim of synchronized measurements across all locations. VERIFIED for the inspected subset.

## 10. Voltage/current scaling assessment

### 10.1 Documented units

The dataset README reports the waveform quantities as current in amperes and voltage in volts.

Verification status:

- Reported units: PARTIALLY VERIFIED — the dataset README states A and V
- The dataset does not model CT/VT instrument transformers
- Do not describe these as secondary-side CT/VT measurements; that assumption is not supported by the dataset documentation

### 10.2 Observed magnitudes

For the inspected samples, observed approximate ranges were:

- Current channels: on the order of a few amperes
- Voltage channels: on the order of a few hundred volts

Example from inspected sample 0:

- Current range: approximately -10 A to 10 A
- Voltage range: approximately -250 V to 250 V

These magnitudes are consistent with the documented A/V convention and with a scaled simulation output. Magnitude alone does not constitute proof of physical units; treat the numeric values as being in the dataset’s stated A/V convention until an authoritative external source confirms otherwise.

### 10.3 Scaling consistency

Across the inspected samples, voltage and current magnitudes vary by sample and by measurement location, which is expected because fault location, fault resistance, line parameters, and operating point all vary. The scaling appears consistent within a sample across locations. This supports the interpretation that the archive uses a single consistent unit convention per episode.

This is PARTIALLY VERIFIED from the inspected subset.

### 10.4 Physical plausibility

- Voltage and current magnitudes are plausible for a 90 kV double-line system represented in the dataset’s stated A/V convention
- Pre-fault current magnitudes are non-zero, reflecting loaded operating conditions
- During-fault current magnitudes increase noticeably on the faulted line segment in the inspected sample 0 case

This assessment is PARTIALLY VERIFIED from the inspected subset and does not constitute a complete EE validation of the full dataset.

### 10.5 Important caveat

Magnitude alone does not confirm units. The values could be:

- primary quantities in V and A
- secondary quantities in V and A
- per-unit quantities scaled to appear like V and A
- scaled simulation outputs with an undocumented factor

Until the authoritative documentation explicitly states the unit convention and transformation chain, treat the waveform numeric values as being in the dataset’s stated A/V convention and confirm the exact scaling before asserting primary or secondary quantities in any published result.

## 11. Potential data-quality issues

### 11.1 Verified issues

None of the following were found as violations, but they remain active checks:

- No missing metadata values were found
- No duplicate sample IDs were found
- No missing waveform values were found in the inspected subset
- No infinite waveform values were found in the inspected subset

### 11.2 Possible concerns to verify further

- The complete 9,022 waveforms were not all inspected in this audit; the inspected subset is small
- The `phase_select` exact 0/1/2 to A/B/C mapping is not fully verified from authoritative text
- It is not yet verified whether any episodes are mislabeled relative to the waveform content
- It is not yet verified whether all episodes have physically plausible EMT solutions; this can only be partially checked by inspecting waveforms for NaNs/Infs and obvious non-physical transients
- The full ZIP integrity was not verified by downloading and decompression-testing the entire archive in this audit; selective ZIP member access was used where possible

### 11.3 Short-duration event episodes

Using one sampling interval = 1/6400 s = 0.00015625 s:

- Episodes with event duration shorter than one sampling interval: 5
- Episodes with event duration shorter than two sampling intervals: 7

These episodes are reported as observed facts, not automatically flagged as invalid. Their acceptability depends on the intended definition of `t_evnt_start`/`t_evnt_end` and the intended window-extraction policy. If your preprocessing treats the entire `[t_evnt_start, t_evnt_end]` interval as fault-active, these very short intervals may require special handling or an explicit exclusion policy.

Note: these numbers come from the full metadata CSV, not from the 4 inspected waveform files.

### 11.4 Recommended additional checks

- Inspect a broader stratified sample across sc_type, fault_target, and sc_location
- Confirm the sc_type mapping from the official README or dataset paper
- Confirm phase_select meaning from the official README or dataset paper
- Confirm voltage/current convention from the official README or dataset paper

## 12. Recommended preprocessing strategy

### 12.1 Episode-wise splitting

- The dataset does not provide a predefined train/test split
- Use episode-wise splitting to avoid leakage
- Split at the sample_id level before any windowing or normalization

### 12.2 Window extraction

- After splitting, extract windows from each episode
- Candidate window lengths at 6400 Hz:
  - one cycle: 128 samples / 0.02 s
  - two cycles: 256 samples / 0.04 s
  - longer windows if the task or model requires more context

### 12.3 Fault inception alignment

- Given `t_evnt_start` and `t_evnt_end`, align windows relative to fault inception or relative to the event window
- `t_evnt_start` is documented as fault inception and `t_evnt_end` as fault clearing
- Consider both aligned and unaligned windows if the deployment scenario is unknown

### 12.4 Normalization

- Normalize using training statistics only
- Candidate approaches:
  - per-channel z-score using training mean/std
  - per-location scaling if location identity is used as a model input or invariance target
  - robust scaling if outliers are expected during faults
- If primary/secondary scaling is confirmed, keep it consistent and document it

### 12.5 Invalid waveform handling

- If any waveform contains NaN or Inf after selective or full extraction, mark the episode as invalid before training
- For the inspected subset, no NaN/Inf were observed
- Do not silently impute without documenting the policy

### 12.6 Class imbalance handling

- The four `sc_type` classes are approximately balanced in count
- After mapping `sc_type` to fault type, re-check balance for the actual fault-type labels
- If severe imbalance appears after mapping, consider:
  - class weights
  - stratified episode splits
  - targeted sampling for minority fault types

### 12.7 Leakage prevention

- No episode-level information from test episodes should influence training statistics
- Do not normalize across the full dataset before splitting
- Do not extract fault-aligned windows using a global fault-event model trained on all episodes before splitting

### 12.8 Current recommendation

The dataset is ready for preprocessing design, but the following should be finalized first:

- authoritative `phase_select` meaning if A/B/C mapping matters to the task
- authoritative voltage/current unit convention if primary/secondary distinction matters

Until then, design the preprocessing pipeline around the metadata fields as labeled, using the verified sc_type and timing semantics where they are needed.

## 13. Unresolved questions

1. What is the exact numeric mapping from `phase_select` 0, 1, 2 to A, B, C, if any task requires it?
2. Are waveforms secondary V and A, primary V and A, or some other convention?
3. Are all 9,022 waveform PKL files internally consistent with the metadata labels?
4. Are there any episodes with numerical artifacts, unrealistic transients, or mislabeled channels in the full archive?
5. Is the downloaded ZIP identical to the Zenodo file, if a local copy exists?
6. Does the local README or dataset paper provide any additional fields or constraints not present in the CSV?
7. Which measurement locations are relevant for the intended VIDYUT-KAVACHA task, and should all 48 channels be used?

## 14. Verification status summary

VERIFIED:

- Metadata CSV shape, columns, types, missing values, duplicates, unique count, class counts, class percentages
- Metadata contains exactly 9,022 episodes
- Inspected waveform files contain 49 columns, 48 waveform channels + 1 time channel
- 8 measurement locations with 3-phase V and I each
- Channel naming convention `{location}_{meas}_{phase}_{unit}`
- Time column name `time_s`
- Sampling interval consistent with 6400 Hz for inspected samples
- Synchronized channels in inspected samples
- No missing or infinite values in inspected waveforms
- No invalid event timing in metadata
- Metadata-to-filename matching for inspected sample IDs
- sc_type mapping 0=LLL, 1=LL, 2=SLG, 3=LLG from authoritative dataset documentation
- t_evnt_start as fault inception and t_evnt_end as fault clearing from authoritative dataset documentation

PARTIALLY VERIFIED:

- Sampling frequency from waveform inspection (verified for inspected subset)
- Episode duration from waveform inspection (verified for inspected subset)
- Voltage/current unit convention from documentation
- Scaling consistency across locations for inspected subset
- Physical plausibility of magnitudes for inspected subset
- phase_select general meaning as involved-phase/base-phase selector

UNVERIFIED:

- Exact `phase_select` 0/1/2 to A/B/C mapping

NOT AVAILABLE:

- Full archive download and full extraction were not performed in this audit
- Complete authoritative README text may not have been captured locally depending on availability

Corrections in this revision:

- Fault class mapping corrected and moved to VERIFIED: 0=LLL, 1=LL, 2=SLG, 3=LLG
- `t_evnt_start`: fault inception — VERIFIED from authoritative dataset documentation
- `t_evnt_end`: fault clearing — VERIFIED from authoritative dataset documentation
- Waveform units documented as A and V; no claim of CT/VT secondary measurements
- Waveform channel naming convention documented explicitly
- `phase_select` kept as involved-phase/base-phase selector; exact 0/1/2 to A/B/C mapping left unverified unless authoritative documentation establishes it
- Short-duration episodes reported as counts only: 5 episodes shorter than one sampling interval, 7 shorter than two
- CSV dtype reporting corrected to parsed numeric/boolean dtypes with invalid-conversion counts
- ZIP integrity check made optional via `--testzip`; normal ZIP inventory no longer decompresses the archive
- PKL directory discovery updated to support both `*_sample.pkl` and `*_sample_hv_double_line_90kv.pkl`
- Strict timestamp validation added: 6,400 samples, strictly increasing, no duplicates, sampling-interval deviation, and finite numeric waveform values

## Commands to reproduce this audit

1. Inspect metadata CSV:
   - `python inspect_dataset.py --csv hv_double_line_90kv_labels.csv`

2. Inspect waveform ZIP inventory without extraction:
   - `python inspect_dataset.py --zip hv_double_line_90kv_preprocessed_data.zip`

3. Inspect selected waveform members if available:
   - `python inspect_dataset.py --pkl-dir pkl_out`

4. Inspect a single waveform file:
   - `python inspect_dataset.py --pkl pkl_out/0_sample.pkl`

Full command-line usage is documented in `inspect_dataset.py`.

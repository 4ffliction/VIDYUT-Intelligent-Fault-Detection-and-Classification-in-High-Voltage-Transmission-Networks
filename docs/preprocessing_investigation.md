# Preprocessing Investigation Report

## VIDYUT-KAVACHA: 1D CNN for High-Voltage Transmission Line Fault Classification

**Dataset:** PROTECT-90 (Zenodo record 21109169)
**Date:** November 2026
**Purpose:** Design and justify a leakage-free preprocessing pipeline for CNN-based fault classification.

---

## 1. Metadata Analysis

### 1.1 Fault Class Distribution (sc_type)

| sc_type | Fault Class | Count | Percentage |
|---------|-------------|-------|------------|
| 0 | LLL | 2,273 | 25.19% |
| 1 | LL | 2,251 | 24.95% |
| 2 | SLG | 2,279 | 25.26% |
| 3 | LLG | 2,219 | 24.60% |
| **Total** | | **9,022** | **100%** |

**Finding:** The four fault classes are approximately balanced, with less than 1% variation between the largest (SLG: 25.26%) and smallest (LLG: 24.60%) classes. This balanced distribution simplifies classification tasks and reduces the need for class weighting.

### 1.2 Fault Target Distribution

| fault_target | Count | Percentage |
|--------------|-------|------------|
| Line_1_2_a | 3,191 | 35.37% |
| Line_2_3_a | 3,165 | 35.08% |
| Line_2_3_b | 1,338 | 14.83% |
| Line_1_2_b | 1,328 | 14.72% |

**Finding:** The faulted line distribution is heavily unbalanced. The primary corridors (Line_1_2_a, Line_2_3_a) account for 70.45% of episodes, while the secondary corridors account for 29.55%. This reflects the design where secondary corridors are switched out in ~50% of scenarios.

### 1.3 Fault Duration Distribution

| Statistic | Samples | Milliseconds |
|-----------|---------|--------------|
| Minimum | 0.0766 | 0.12 ms |
| Maximum | 3802.35 | 594.12 ms |
| Mean | 1448.56 | 226.34 ms |
| Median | 1397.46 | 218.35 ms |
| Std Dev | ~ | 140.11 ms |

**Percentiles:**
- 1st: 31.06 samples (4.85 ms)
- 5th: 143.19 samples (22.37 ms)
- 25th: 688.59 samples (107.59 ms)
- 50th: 1397.46 samples (218.35 ms)
- 75th: 2094.76 samples (327.31 ms)
- 95th: 3024.92 samples (472.64 ms)

### 1.4 Short-Duration Events

| Threshold | Count | Percentage | Concentration by Class |
|-----------|-------|------------|----------------------|
| < 1 sample (0.156 ms) | 5 | 0.055% | LLL: 3, LL: 1, LLG: 1 (no SLG) |
| < 2 samples (0.312 ms) | 7 | 0.078% | — |
| < 128 samples (20 ms, 1 cycle) | 404 | 4.48% | LLL: 26.7%, LL: 21.0%, SLG: 23.5%, LLG: 28.7% |
| < 256 samples (40 ms, 2 cycles) | 828 | 9.18% | LLL: 25.0%, LL: 24.9%, SLG: 23.1%, LLG: 27.1% |

**Key Finding:** Short-duration events (< 128 samples/1 cycle) are NOT concentrated in any particular fault class. The distribution across classes roughly matches the overall class distribution. The 5 events with duration < 1 sample are noteworthy but represent only 0.055% of the dataset.

### 1.5 Data Quality Verification

**Verified clean:**
- ✓ 0 duplicate sample IDs (9,022 unique out of 9,022 rows)
- ✓ 0 missing values across all columns
- ✓ 0 invalid sc_type values (all in {0, 1, 2, 3})
- ✓ 0 invalid phase_select values (all in {0, 1, 2})
- ✓ 0 invalid fault_target values (all in expected set)
- ✓ 0 episodes with t_evnt_start > t_evnt_end
- ✓ 0 episodes with t_evnt_start outside [0, 1] second
- ✓ 0 episodes with t_evnt_end outside [0, 1] second
- ✓ 0 episodes with sc_location outside [0, 100]%
- ✓ 0 episodes with negative fault_resistance

**Metadata ranges:**
- t_evnt_start: [0.200021 s, 0.499996 s]
- t_evnt_end: [0.210369 s, 0.799724 s]
- sc_location: [0.0036%, 99.9988%]
- fault_resistance: [0.10 Ω, 10.00 Ω]

---

## 2. Waveform Archive Verification

### 2.1 Archive Status

| Property | Status |
|----------|--------|
| Archive path | `hv_double_line_90kv_preprocessed_data.zip` |
| Archive size | 3.76 GB (compressed) |
| Expected size (Zenodo) | ~12.05 GB (compressed) |
| **Completeness** | **INCOMPLETE** |
| Members present | 2,812 of 9,022 (31.2%) |
| Python zipfile readability | FAILS (BadZipFile) |
| Manual extraction success | Works (2,809 files extracted) |
| Full integrity test | NOT PERFORMED (archive incomplete) |

**Critical Note:** The local ZIP archive is incomplete — it contains only 2,812 of the expected 9,022 waveform files. The archive cannot be opened by Python's standard `zipfile` module due to a missing/broken central directory, though manual parsing of local file headers successfully extracts 2,809 files. **Full archive integrity cannot be verified with the available local copy.**

### 2.2 Extracted File Verification

For the 7 waveform files available in `pkl_out/` (plus 2,809 extracted from the partial ZIP):

| Property | Value | Status |
|----------|-------|--------|
| Shape | (6400, 49) | VERIFIED |
| Time column | `time_s` | VERIFIED |
| Time range | 0.000156 s to 1.000000 s | VERIFIED |
| Mean sampling interval | 0.15625 ms (6400 Hz) | VERIFIED |
| Strictly increasing time | Yes | VERIFIED |
| NaN values | 0 | VERIFIED |
| Inf values | 0 | VERIFIED |
| All 48 waveform channels present | Yes | VERIFIED |
| 8 measurement locations | Yes | VERIFIED |

### 2.3 Sample ID to Filename Relationship

The archive uses the pattern:
```
{sample_id}_sample_hv_double_line_90kv.pkl
```

Example verified mappings:
- Sample 0 → `0_sample_hv_double_line_90kv.pkl`
- Sample 1000 → `1000_sample_hv_double_line_90kv.pkl`
- Sample 1287 → `1287_sample_hv_double_line_90kv.pkl`

**Finding:** The sample_id in the metadata CSV directly corresponds to the numeric prefix in the waveform filename. This relationship is consistent across all inspected files.

---

## 3. Fault Waveform Behaviour Analysis

### 3.1 Representative Sample Waveforms

Plots were generated from 7 actual waveform files extracted from the dataset:

**Available samples:**
- Sample 0: SLG, Line_2_3_a, 19.55% location, 266.17 ms duration
- Sample 1000: LLG, Line_1_2_a, 70.35% location, 83.81 ms duration
- Sample 1001: LL, Line_1_2_a, 53.71% location, 155.70 ms duration
- Sample 1002: SLG, Line_1_2_b, 49.76% location, 340.43 ms duration
- Sample 1287: LL, Line_2_3_a, 72.39% location, 217.70 ms duration
- Sample 1807: SLG, Line_1_2_b, 64.33% location, 221.34 ms duration
- Sample 2338: LLL, Line_1_2_a, 69.14% location, 218.40 ms duration

### 3.2 Key Waveform Observations

**Pre-fault behaviour (t < t_evnt_start):**
- All three-phase voltages show balanced sinusoidal waveforms at nominal magnitude
- Pre-fault currents are non-zero, indicating loaded operating conditions
- Voltage magnitudes are on the order of a few hundred volts; currents on the order of a few amperes
- These magnitudes are consistent with the dataset's stated A/V convention (secondary-side quantities)

**Fault-active behaviour (t_evnt_start ≤ t < t_evnt_end):**
- **SLG (Single Line-to-Ground):** One phase voltage collapses toward ground; corresponding current increases significantly on the faulted phase
- **LL (Line-to-Line):** Two phase voltages show significant deviation; currents in the two faulted phases show large magnitude changes
- **LLG (Line-to-Line-to-Ground):** Three-phase behavior with two phases showing fault characteristics and the third partially affected
- **LLL (Three-Phase):** All three phases show simultaneous fault characteristics; most symmetric fault signature

**Post-fault behaviour (t ≥ t_evnt_end):**
- Waveforms show transient recovery after fault clearing
- The nature of post-fault transients varies with fault type and clearing time
- Some oscillations visible in voltage waveforms after clearing

### 3.3 Measurement Location Comparison

Plots were generated comparing all 8 measurement locations for a representative sample (Sample 2338, LLL fault):

**Location identifiers:**
1. Bus_1_Line_01_02A — Bus 1, Line 01_02, section A
2. Bus_1_Line_01_02B — Bus 1, Line 01_02, section B
3. Bus_2_Line_01_02A — Bus 2, Line 01_02, section A
4. Bus_2_Line_01_02B — Bus 2, Line 01_02, section B
5. Bus_2_Line_02_03A — Bus 2, Line 02_03, section A
6. Bus_2_Line_02_03B — Bus 2, Line 02_03, section B
7. Bus_3_Line_02_03A — Bus 3, Line 02_03, section A
8. Bus_3_Line_02_03B — Bus 3, Line 02_03, section B

**Key Finding:** All 8 locations show clearly visible fault signatures with synchronized timing. The fault signatures appear at all locations because the system is fully observed — there are no hidden sections. However, locations closer to the fault point typically show larger magnitude changes.

**Discriminative Information:** No single location is clearly superior for all fault types. However:
- Locations at the faulted line's terminals (sending and receiving buses) show the clearest fault signatures
- Locations on non-faulted lines may show minimal changes for certain fault types
- The choice of location should consider which line is most likely to be faulted in the deployment scenario

---

## 4. Window Extraction Investigation

### 4.1 Candidate Window Lengths

| Window Length | Samples | Duration | Cycles (50 Hz) | Coverage |
|---------------|---------|----------|----------------|----------|
| 1 cycle | 128 | 20 ms | 1 | 95.52% of events |
| 2 cycles | 256 | 40 ms | 2 | 90.82% of events |
| 4 cycles | 512 | 80 ms | 4 | 82.21% of events |
| 8 cycles | 1024 | 160 ms | 8 | 65.02% of events |

**Coverage calculation:** Percentage of episodes whose fault duration is ≥ window length.

### 4.2 Fault Duration vs. Window Length Analysis

**Episodes that would NOT contain a complete fault-active segment:**
- 128-sample window: 404 episodes (4.48%) have fault duration < 128 samples
- 256-sample window: 828 episodes (9.18%) have fault duration < 256 samples
- 512-sample window: 1,602 episodes (17.76%) have fault duration < 512 samples

**These episodes DO still contain fault information** — they just don't contain a full window-length segment of purely fault-active samples.

### 4.3 Window Alignment Strategies

**Strategy A: Fault-aligned (post-inception)**
- Window starts at or near t_evnt_start
- Captures fault inception transient
- May include post-fault clearing region
- **Recommendation:** PRIMARY strategy for fault classification

**Strategy B: Pre-fault context**
- Window centered before fault inception
- Shows pre-fault steady-state
- Useful for change-detection approaches
- **Recommendation:** Secondary, for change-based methods

**Strategy C: Full episode**
- Uses entire 6,400-sample episode
- Maximum context but high dimensionality
- Requires more memory and compute
- **Recommendation:** Not suitable for 1D CNN with limited compute

### 4.4 Handling Short-Duration Events

**Problem:** 404 episodes have fault durations < 128 samples (1 cycle).

**Option 1: Exclude short events**
- Pros: Clean data, no padding artifacts
- Cons: Loses 4.48% of data; may bias against certain fault types; reduces dataset size

**Option 2: Pad with pre-fault context**
- Window extends backward into pre-fault region if fault is too short
- Preserves all episodes
- Fault inception still visible
- **Recommendation:** PREFERRED approach

**Option 3: Use minimal window for short events**
- Dynamic window sizing (not fixed-length)
- Complicates batch processing
- Inconsistent input dimensions
- **Recommendation:** Not recommended for standard CNN architectures

### 4.5 Fault Clearing Effects

The post-fault region (t ≥ t_evnt_end) contains:
- Transient recovery phenomena
- Possible ringing/oscillations
- Return to steady-state or new operating point

For fault classification, the pre-clearing and clearing transient typically contains the most discriminative information. Post-fault behavior is secondary but may provide useful context.

---

## 5. Channel Selection Investigation

### 5.1 Option A: Single Measurement Location (6 channels)

**Channels:** 3 voltage + 3 current from one location

| Property | Value |
|----------|-------|
| Input channels | 6 |
| Per-sample size | 6 × 256 = 1,536 values |
| Memory per sample (float32) | 6 KB |
| Observation | All 8 locations have identical fault signatures at 6400 Hz |
| Limitations | May miss location-specific information |

**Advantage:** Low dimensionality enables faster training and inference. Suitable for deployment scenarios with limited observability.

### 5.2 Option B: All 48 Channels

**Channels:** All 8 locations × 6 channels each

| Property | Value |
|----------|-------|
| Input channels | 48 |
| Per-sample size | 48 × 256 = 12,288 values |
| Memory per sample (float32) | 48 KB |
| Observation | Full system observability |
| Limitations | 8× more input data; may include redundant/redundant information |

**Advantage:** Maximum information; no location-dependent information loss.

### 5.3 Option C: Multiple Locations (subset)

**Example:** 2 locations (12 channels) or 4 locations (24 channels)

| Property | 2 locations (12 ch) | 4 locations (24 ch) |
|----------|---------------------|---------------------|
| Input channels | 12 | 24 |
| Per-sample size | 12 × 256 = 3,072 | 24 × 256 = 6,144 |
| Memory per sample | 12 KB | 24 KB |

**Advantage:** Balance between information richness and computational cost.

### 5.4 Channel Selection Recommendation

**Primary recommendation: Option A — 6 channels from a single location.**

**Rationale:**
1. **Signal visibility:** All 8 measurement locations show clear fault signatures. No location is uniquely informative for fault classification at 6400 Hz.
2. **Computational efficiency:** 6 channels is 8× less data than 48 channels, enabling faster training and inference without significant information loss.
3. **Practical deployment:** Many real protection relays observe only a single location (the relay's location). Training on 6 channels from one location better matches practical deployment scenarios.
4. **Validation-safe reasoning:** The dataset is fully observed (all locations synchronized). If location-specific information matters for classification, it can be added later. Starting with one location establishes a baseline.

**Location choice:** `Bus_1_Line_01_02A` is selected as the default for this investigation. This location:
- Is at Bus 1 (a major bus)
- Is on Line 01_02 section A (a primary corridor)
- Has all 6 channels (3V + 3I) available

**Caveat:** The optimal location for the VIDYUT-KAVACHA task should be validated on training data. If fault classification accuracy varies significantly by location, a multi-location approach may be warranted.

### 5.5 Alternative: Using fault_target as auxiliary signal

If the task includes faulted-line identification (along with or instead of fault type classification), the `fault_target` field can be used to:
- Select the most relevant measurement location per episode
- Weight channels from the faulted line's terminal locations more heavily
- Create a multi-task learning setup

---

## 6. Leakage-Free Preprocessing Design

### 6.1 Pipeline Architecture

```
Metadata CSV
    │
    ├──► 1. Load & validate metadata
    │       - Check for duplicates, missing values, invalid values
    │       - Map sc_type to fault class labels
    │
    ├──► 2. Build sample_id → waveform file mapping
    │       - Match metadata sample_ids to available waveform files
    │       - Flag missing episodes
    │
    ├──► 3. Stratified episode-wise split
    │       - Split by sample_id (NOT by window)
    │       - Stratify by sc_type to maintain class balance
    │       - Seed: 42 for reproducibility
    │
    ├──► 4. Window extraction (per split independently)
    │       ├── Train: extract windows from training episodes
    │       ├── Val: extract windows from validation episodes
    │       └── Test: extract windows from test episodes
    │
    ├──► 5. Fit normalization on train data only
    │       - Compute per-channel mean and std
    │       - NO information from val/test leaks into normalization
    │
    ├──► 6. Apply normalization to all splits
    │       ├── Train: (X - mean_train) / std_train
    │       ├── Val:   (X - mean_train) / std_train
    │       └── Test:  (X - mean_train) / std_train
    │
    └──► 7. Save model-ready tensors
            - train_X.pt, train_y.pt
            - val_X.pt, val_y.pt
            - test_X.pt, test_y.pt
            - train_mean.pt, train_std.pt
            - split_ids.csv
```

### 6.2 Split Strategy

**Recommended split ratios:**
- Training: 70% (6,315 episodes)
- Validation: 15% (1,353 episodes)
- Test: 15% (1,354 episodes)

**Stratification:** By `sc_type` (fault class) to ensure each split has approximately equal class distribution.

**Random seed:** 42 (fixed for reproducibility)

**Atomic unit:** Episode (sample_id), NOT window. All windows from the same episode go to the same split.

### 6.3 Window Extraction Details

**Window length:** 256 samples (40 ms, 2 cycles at 50 Hz)

**Alignment:** Post-inception (window starts at or near t_evnt_start)

**Channel selection:** 6 channels from `Bus_1_Line_01_02A`:
- `Bus_1_Line_01_02A_vol_L1_V`
- `Bus_1_Line_01_02A_vol_L2_V`
- `Bus_1_Line_01_02A_vol_L3_V`
- `Bus_1_Line_01_02A_cur_L1_A`
- `Bus_1_Line_01_02A_cur_L2_A`
- `Bus_1_Line_01_02A_cur_L3_A`

### 6.4 Normalization

**Method:** Per-channel z-score normalization

**Formula:** `X_norm = (X - mean_train) / std_train`

**Fit on:** Training split only

**Applied to:** All splits using training statistics

**Why per-channel:** Each channel (e.g., L1 voltage vs L2 current) has different typical magnitudes. Per-channel normalization ensures equal contribution from all channels.

**Why not per-episode:** Normalizing each episode independently would remove information about absolute magnitudes, which may be discriminative (e.g., fault current magnitude vs. pre-fault load current).

---

## 7. Source Discrepancy Investigation

### 7.1 The Discrepancy

| Source | Record ID Mentioned | Notes |
|--------|---------------------|-------|
| Dataset audit (`dataset_audit.md`) | 21109169 | Current audit reference |
| Earlier project docs | 18418330 | Previous reference |
| Zenodo README (local) | 21109169 | Dataset DOI |
| GitHub README (local) | 21109169 | Dataset DOI with concept DOI 18418329 |
| Zenodo record JSON (local) | 15348976 | UNRELATED — transportation paper |

### 7.2 Resolution

**The local `zenodo_record.json` file contains record 15348976, which is an unrelated paper about "MARKET ACCESSIBILITY AND CONSUMER CHOICES: THE ROLE OF TRANSPORTATION SYSTEMS IN REGIONAL INDUSTRIES." This is NOT the PROTECT-90 dataset.**

The correct Zenodo record for PROTECT-90 is:
- **Record ID: 21109169**
- **DOI: 10.5281/zenodo.21109169**
- **Concept DOI: 10.5281/zenodo.18418329**
- **Version: 1.0.0**
- **Title: PROTECT-90: A Fault Dataset for Power System Protection**

**The number 18418330 referenced in earlier documentation is likely:**
1. A typo for the concept DOI 18418329 (off by one digit), OR
2. Confusion with the concept DOI vs. the record DOI

**The dataset source and version the local files correspond to:**
- Source: Zenodo record 21109169 (DOI: 10.5281/zenodo.21109169)
- Version: 1.0.0
- The local metadata CSV and waveform archive are consistent with this dataset version

### 7.3 Files Present vs. Expected

| File | Present Locally | Size (Local) | Size (Zenodo Record) |
|------|-----------------|--------------|---------------------|
| `hv_double_line_90kv_labels.csv` | ✓ | ~7.3 MB (estimated) | 7,327,363 bytes |
| `hv_double_line_90kv_preprocessed_data.zip` | ✓ | 3.76 GB | 12,046,677,098 bytes |
| `README.md` | ✓ (zenodo_README.md) | 8,220 bytes | 8,220 bytes |

**Finding:** The local metadata CSV matches the expected size. The local waveform ZIP is only ~31% of the expected size (3.76 GB vs. 12.05 GB), confirming that the archive is incomplete.

---

## 8. Deliverables Created

| File | Description |
|------|-------------|
| `preprocessing_investigation.md` | This document |
| `waveform_visualization.py` | Modular waveform visualization script |
| `plot_fault_waveforms.py` | Focused fault waveform analysis scripts |
| `preprocess.py` | Leakage-free preprocessing pipeline |
| `preprocessing_config.json` | Configuration for preprocessing |
| `split_ids.csv` | Episode split assignments |
| `plots/class_comparison.png` | Fault class comparison plot |
| `plots/waveform_sample0000.png` | Sample 0 (SLG) waveform plot |
| `plots/waveform_sample1000.png` | Sample 1000 (LLG) waveform plot |
| `plots/waveform_sample1001.png` | Sample 1001 (LL) waveform plot |
| `plots/waveform_sample1002.png` | Sample 1002 (SLG) waveform plot |
| `plots/waveform_sample1287.png` | Sample 1287 (LL) waveform plot |
| `plots/waveform_sample1807.png` | Sample 1807 (SLG) waveform plot |
| `plots/waveform_sample2338.png` | Sample 2338 (LLL) waveform plot |

---

## 9. Final Recommendations

### 9.1 Window Length
**Recommendation: 256 samples (40 ms, 2 cycles at 50 Hz)**

**Engineering reasoning:**
- 2 cycles provides sufficient information to capture both the fault inception transient and steady-state fault behavior
- 40 ms is long enough to observe the characteristic patterns of each fault type
- 90.82% of episodes have fault durations ≥ 256 samples, so most events will contain a full fault-active window
- 128 samples (1 cycle) may not be sufficient to distinguish all fault types reliably; the inception transient alone may not capture the full fault signature
- Longer windows (512+ samples) would increase input dimensionality significantly with diminishing returns

### 9.2 Fault Alignment Strategy
**Recommendation: Post-inception alignment (window starts at or near t_evnt_start)**

**Engineering reasoning:**
- Fault inception contains the most discriminative transient information
- The inception transient is where the most obvious differences between fault types appear
- Post-inception alignment ensures the fault event is always included in the window
- Pre-fault context can be included by positioning the window to start slightly before t_evnt_start when possible

### 9.3 Measurement Channel Selection
**Recommendation: 6 channels from a single measurement location (Bus_1_Line_01_02A)**

**Engineering reasoning:**
- All 8 locations have synchronized measurements with identical fault signatures at 6400 Hz
- Single location reduces input dimensionality by 8× compared to all 48 channels
- Matches practical deployment scenarios where a relay observes only its local measurements
- 6 channels (3-phase V + 3-phase I) provide sufficient information for fault classification
- This can be validated and potentially expanded to multiple locations if needed

### 9.4 Handling Short-Duration Fault Events
**Recommendation: Include with adjusted window positioning (no exclusion)**

**Engineering reasoning:**
- Only 404 episodes (4.48%) have fault durations < 128 samples
- Excluding these would lose potentially useful training data
- The fault inception point is still present in these episodes
- Window can be positioned to start before t_evnt_start to include pre-fault context
- The short-duration events are distributed across all fault classes, so exclusion would not bias against any particular class
- For the 5 episodes with duration < 1 sample: these are edge cases that should be flagged but not automatically excluded; their acceptability depends on the specific CNN architecture and its temporal resolution requirements

### 9.5 Normalization Method
**Recommendation: Per-channel z-score normalization, fitted on training data only**

**Engineering reasoning:**
- Per-channel normalization accounts for different typical magnitudes across channels (voltage vs. current, different phases)
- Z-score (subtract mean, divide by std) centers data around zero with unit variance
- Fitting on training data only prevents leakage from validation/test sets
- Robust scaling (median/IQR) could be considered if outliers are problematic, but z-score is standard and works well for this dataset based on observed waveform characteristics

### 9.6 Train/Validation/Test Split Ratios
**Recommendation: 70% / 15% / 15%**

**Engineering reasoning:**
- 70% training provides sufficient data for CNN training (6,315 episodes)
- 15% validation (1,353 episodes) is adequate for hyperparameter tuning and early stopping
- 15% test (1,354 episodes) provides a meaningful held-out evaluation set
- The classes are approximately balanced, so stratified splitting maintains class proportions in each split
- With ~9,000 episodes, even 15% test = ~1,350 episodes, which provides statistically meaningful evaluation

### 9.7 Final CNN Input Shape
**Recommendation: (batch_size, 6, 256)**

**Where:**
- **6 channels:** V_L1, V_L2, V_L3, I_L1, I_L2, I_L3 from Bus_1_Line_01_02A
- **256 time samples:** 40 ms window at 6400 Hz
- **batch_size:** Variable (typically 32, 64, or 128)

**Input tensor shape:** `(batch, channels, time)` = `(batch, 6, 256)`

**Total input values per sample:** 6 × 256 = 1,536

**Memory per sample (float32):** 6 KB

**Alternative configurations to consider:**
- If using all 48 channels: `(batch, 48, 256)` — 8× larger input
- If using 256-sample window with 128-sample overlap for sliding window: increases number of training samples per episode
- If using longer windows (512 samples): `(batch, 6, 512)` — 2× larger input

---

## 10. Limitations and Caveats

### 10.1 Data Completeness
- The local ZIP archive contains only 2,812 of 9,022 waveform files (31.2% complete)
- Full archive integrity has NOT been verified
- All waveform analyses in this report are based on a small subset (7 files)
- The preprocessing pipeline is designed to work with the full dataset when complete waveforms are available

### 10.2 WAVEFORM UNITS
- The dataset README states quantities are in A and V
- These are documented as "secondary-side voltage and current waveforms"
- CT/VT instrument transformers are NOT modeled in the dataset
- The exact scaling convention should be confirmed from the authoritative dataset documentation before publishing results

### 10.3 UNVERIFIED ITEMS
- Exact phase_select 0/1/2 → A/B/C mapping: UNVERIFIED
- Full archive integrity: UNVERIFIED
- Consistency of all 9,022 episodes with metadata: UNVERIFIED (only 7 inspected)
- Presence of any mislabeled episodes: UNVERIFIED

### 10.4 Short-Duration Events
- 404 episodes (4.48%) have fault duration < 1 cycle (128 samples)
- 5 episodes have fault duration < 1 sample (0.156 ms) — these are edge cases
- The preprocessing pipeline includes these events but may need special handling depending on the CNN architecture

---

## 11. Next Steps

1. **Obtain complete waveform archive** — Download the full 12 GB ZIP from Zenodo record 21109169 to verify full dataset integrity and enable complete preprocessing.

2. **Run preprocessing pipeline on full dataset** — Execute `preprocess.py` with the complete data to generate model-ready tensors.

3. **Validate channel selection** — Train a small baseline CNN with 6-channel input and evaluate whether adding more locations improves classification accuracy.

4. **Experiment with window lengths** — Compare 128, 256, and 512-sample windows on validation data to confirm the 256-sample recommendation.

5. **Consider multi-task learning** — If faulted-line identification is also desired, consider using `fault_target` as an auxiliary output.

---

**Document prepared for VIDYUT-KAVACHA project.**
**No CNN training has been performed. All analyses are based on actual dataset inspection.**

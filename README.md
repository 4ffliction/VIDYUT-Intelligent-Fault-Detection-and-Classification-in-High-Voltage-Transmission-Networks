# VIDYUT-Intelligent-Fault-Detection-and-Classification-in-High-Voltage-Transmission-Networks
**VIDYUT** is an AI-driven transmission fault detection and classification framework designed to analyse electrical waveform data and identify different short-circuit fault conditions in high-voltage power systems. It combines signal processing, machine learning, and deep learning to develop a data-driven approach to power system fault analysis.

The project uses the **PROTECT-90 dataset**, comprising simulated three-phase voltage and current measurements from a 90 kV transmission system. It investigates four short-circuit fault categories: Single Line-to-Ground (SLG), Line-to-Line (LL), Double Line-to-Ground (LLG), and Three-Phase (LLL) faults.

VIDYUT explores a hybrid learning architecture that integrates:

- **Random Forest (RF):** Uses engineered electrical signal features, including RMS, standard deviation, peak-to-peak amplitude, and maximum absolute amplitude, to capture fault-related characteristics.
- **1D Convolutional Neural Network (CNN):** Learns temporal patterns directly from multichannel voltage and current waveforms.
- **Hybrid CNN–RF Framework:** Combines learned temporal representations with engineered electrical features to investigate complementary feature learning and classification performance.

### Key Technical Components

- Multichannel three-phase voltage and current waveform analysis.
- Fault-aligned signal windowing and preprocessing.
- Episode-level training, validation, and testing separation.
- Signal normalisation and engineered feature extraction.
- Comparative evaluation of machine learning and deep learning approaches.
- Per-class performance analysis, confusion matrices, and classification metrics.

### Objective

The primary objective of VIDYUT is to investigate reliable, data-driven fault identification in high-voltage transmission networks through the integration of electrical signal characteristics and machine learning techniques.

**VIDYUT — Bringing intelligence to power system protection.**

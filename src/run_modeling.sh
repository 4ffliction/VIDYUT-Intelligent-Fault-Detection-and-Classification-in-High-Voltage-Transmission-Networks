#!/usr/bin/env bash
# VIDYUT-KAVACHA: CNN + Random Forest hybrid pipeline (PROTECT-90).
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Step 1: build window store (fault-interval windows, train-only z-score) ==="
python preprocess.py

echo "=== Step 2: verify splits / shapes / normalisation ==="
python verify_pipeline.py

echo "=== Step 3: experiments A, B, C + validation selection + single test eval ==="
python train_hybrid.py

echo "=== DONE: see hybrid_results/ ==="

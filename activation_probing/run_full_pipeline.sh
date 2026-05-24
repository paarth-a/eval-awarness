#!/usr/bin/env bash
# End-to-end activation-probing pipeline.
# Runs sequentially: extract -> train -> apply -> analyze.
#
# Override the model via env var:
#   ACTPROBE_MODEL=Qwen/Qwen3-32B bash run_full_pipeline.sh
#
# Override sample limit per (benchmark, condition):
#   ACTPROBE_N_PER=80 bash run_full_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/.."   # cd to repo root so `python -m activation_probing.*` works

echo "=========================================="
echo "Activation Probing Pipeline"
echo "Model: ${ACTPROBE_MODEL:-Qwen/Qwen3-8B}"
echo "Layers: ${ACTPROBE_LAYERS:-auto}"
echo "Max tokens: ${ACTPROBE_MAX_TOKENS:-4096}"
echo "N per cell: ${ACTPROBE_N_PER:-all}"
echo "=========================================="

echo ""
echo "Step 0: Coverage check"
python -m activation_probing.data_loader

echo ""
echo "Step 1: Extract activations (all benchmarks × all conditions)"
python -m activation_probing.extract_activations --all

echo ""
echo "Step 2: Train probes (all benchmarks)"
python -m activation_probing.train_probes --all

echo ""
echo "Step 3: Apply probes (all benchmarks × all conditions)"
python -m activation_probing.apply_probes --all

echo ""
echo "Step 4: Aggregate + render figure"
python -m activation_probing.analyze_probes

echo ""
echo "Done. Results in activation_probing/results/"

#!/bin/bash
# Convenience wrapper: runs retrieval + VTAB + ImageNet sweeps in sequence and
# aggregates results into a CSV. All eval scripts read MODEL_SPECS from
# themselves and accept the env vars below.
#
# Required env vars:
#   CHECKPOINT_ROOT  dir with <model_dir>/checkpoints/epoch_<N>/  +  <model_dir>/params.txt
#   DATASET_ROOT     clip_benchmark dataset_root (retrieval + VTAB)
#   IMAGENET_ROOT    imagenet val dir
# Optional:
#   RESULTS_ROOT     where to write JSON outputs (default: <CLIP_benchmark>/results)
#   CUDA_VISIBLE_DEVICES, CONDA_ENV
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

: "${CHECKPOINT_ROOT:?must set CHECKPOINT_ROOT}"
: "${DATASET_ROOT:?must set DATASET_ROOT}"
: "${IMAGENET_ROOT:?must set IMAGENET_ROOT}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/results}"
export CHECKPOINT_ROOT DATASET_ROOT IMAGENET_ROOT RESULTS_ROOT

if [ -n "${CONDA_ENV:-}" ]; then
    source ~/.bashrc
    conda activate "${CONDA_ENV}"
fi

echo "===================="
echo "Phase 1: zeroshot retrieval"
echo "===================="
bash "${SCRIPT_DIR}/eval_retrieval_over_epochs.sh"

echo "===================="
echo "Phase 2: zeroshot VTAB classification"
echo "===================="
bash "${SCRIPT_DIR}/eval_vtab_over_epochs.sh"

echo "===================="
echo "Phase 3: ImageNet-1k zeroshot"
echo "===================="
bash "${SCRIPT_DIR}/eval_imagenet_over_epochs.sh"

echo "===================="
echo "Phase 4: aggregate"
echo "===================="
python "${SCRIPT_DIR}/aggregate_results.py" "${RESULTS_ROOT}" --output "${RESULTS_ROOT}/summary.csv"

echo "Done. CSV: ${RESULTS_ROOT}/summary.csv"

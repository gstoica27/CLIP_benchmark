#!/bin/bash
# Submit one beaker job per checkpoint for VTAB zeroshot classification.
# Each job runs cli_rope.py once with --dataset vtab; cli_rope expands to the
# 19-dataset VTAB collection internally (skipping pets/oxford_iiit_pet/resisc45/sun397
# = 16 actual evals). Outputs go to ${RESULTS_ROOT}/<model>/epoch_<N>/.
#
# Required env vars: CHECKPOINT_ROOT, DATASET_ROOT
# Optional: RESULTS_ROOT, OPEN_CLIP_REPO, BEAKER_HOME, CONDA_ENV,
#           WORKSPACE, BUDGET, PRIORITY, BEAKER_IMAGE  (see eval_retrieval_per_ckpt.sh)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIP_BENCH_DIR="$(dirname "${SCRIPT_DIR}")"
DEFAULT_REPO="$(dirname "${CLIP_BENCH_DIR}")"

: "${CHECKPOINT_ROOT:?must set CHECKPOINT_ROOT}"
: "${DATASET_ROOT:?must set DATASET_ROOT}"
RESULTS_ROOT="${RESULTS_ROOT:-${CLIP_BENCH_DIR}/results}"
OPEN_CLIP_REPO="${OPEN_CLIP_REPO:-${DEFAULT_REPO}}"
BEAKER_HOME="${BEAKER_HOME:-${HOME}}"
CONDA_ENV="${CONDA_ENV:-trainer}"
WORKSPACE="${WORKSPACE:-ai2/oe-encoder}"
BUDGET="${BUDGET:-ai2/oe-mm}"
PRIORITY="${PRIORITY:-urgent}"
BEAKER_IMAGE="${BEAKER_IMAGE:-sanghol/molmo2-torch2.7.1-cuda12.8}"

declare -a MODEL_SPECS=(
    # "tag|model_dir|model_arch|epoch"
)

if [ "${#MODEL_SPECS[@]}" -eq 0 ]; then
    echo "ERROR: MODEL_SPECS is empty — edit $0 and add your checkpoints" >&2
    exit 1
fi

for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r tag model_name model_arch epoch <<< "$spec"
    pretrained_path="${CHECKPOINT_ROOT}/${model_name}/checkpoints/epoch_${epoch}"
    output_pattern="${RESULTS_ROOT}/${model_name}/epoch_${epoch}/{dataset}_{model}_{language}_{task}.json"
    job_name="clipbench_vtab_${tag}"

    gantry run --allow-dirty \
        --task-name "$job_name" \
        --name "$job_name" \
        --workspace "$WORKSPACE" \
        --cluster ai2/saturn \
        --cluster ai2/jupiter \
        --cluster ai2/ceres \
        --gpus 1 \
        --priority "$PRIORITY" \
        --shared-memory 512GiB \
        --weka prior-default:/weka/prior-default \
        --weka oe-training-default:/weka/oe-training-default \
        --weka oe-mm-default:/weka/oe-mm \
        --env "HOME=${BEAKER_HOME}" \
        --budget "$BUDGET" \
        --beaker-image "$BEAKER_IMAGE" \
        -- /usr/bin/bash -c "source ~/.bashrc \
            && cd ${OPEN_CLIP_REPO}/CLIP_benchmark \
            && conda activate ${CONDA_ENV} \
            && CUDA_VISIBLE_DEVICES=0 python -m clip_benchmark.cli_rope eval \
                --model_type open_clip \
                --model ${model_arch} \
                --pretrained ${pretrained_path} \
                --task zeroshot_classification \
                --dataset_root ${DATASET_ROOT} \
                --dataset vtab \
                --output ${output_pattern} \
                --is-fsdp \
                --skip_existing"
done

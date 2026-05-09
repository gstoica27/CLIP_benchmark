#!/bin/bash
# Submit one beaker job per checkpoint for zeroshot retrieval (mscoco_captions
# + flickr30k). Each job runs cli_rope.py once with --dataset mscoco_captions
# flickr30k. Outputs go to ${RESULTS_ROOT}/<model>/epoch_<N>/.
#
# Required env vars:
#   CHECKPOINT_ROOT  dir with <model_dir>/checkpoints/epoch_<N>/  +  <model_dir>/params.txt
#   DATASET_ROOT     clip_benchmark dataset_root
# Optional:
#   RESULTS_ROOT     default: <repo>/CLIP_benchmark/results
#   OPEN_CLIP_REPO   absolute path to the cloned open_clip repo on the weka mount
#                    that the beaker container should `cd` into (default: derived
#                    from this script's location)
#   BEAKER_HOME      $HOME inside the container (default: $HOME on the submit host).
#                    The .bashrc here must contain conda init and a `${CONDA_ENV}`
#                    env with torch / open_clip / task_adaptation deps installed.
#   CONDA_ENV        conda env name (default: trainer)
#   WORKSPACE        gantry workspace (default: ai2/oe-encoder)
#   BUDGET           gantry budget (default: ai2/oe-mm)
#   PRIORITY         gantry priority (default: urgent)
#   BEAKER_IMAGE     gantry beaker image (default: sanghol/molmo2-torch2.7.1-cuda12.8)
#
# Edit MODEL_SPECS below.
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
    # Example:
    # "400mv2|400Mv2_lr0p001-...-rope_dp|ViT-B-16-SigLIP|49"
)

if [ "${#MODEL_SPECS[@]}" -eq 0 ]; then
    echo "ERROR: MODEL_SPECS is empty — edit $0 and add your checkpoints" >&2
    exit 1
fi

for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r tag model_name model_arch epoch <<< "$spec"
    pretrained_path="${CHECKPOINT_ROOT}/${model_name}/checkpoints/epoch_${epoch}"
    output_pattern="${RESULTS_ROOT}/${model_name}/epoch_${epoch}/{dataset}_{model}_{language}_{task}.json"
    job_name="clipbench_retrieval_${tag}"

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
                --task zeroshot_retrieval \
                --dataset_root ${DATASET_ROOT} \
                --dataset mscoco_captions flickr30k \
                --output ${output_pattern} \
                --batch_size 64 \
                --language en \
                --recall_k 1 5 \
                --is-fsdp \
                --skip_existing"
done

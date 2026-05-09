#!/bin/bash
# Loop zeroshot retrieval (mscoco_captions, flickr30k) over every (model, epoch)
# in the MODEL_SPECS list below.
#
# Required env vars (no defaults — set before running):
#   CHECKPOINT_ROOT  dir with <model_dir>/checkpoints/epoch_<N>/  +  <model_dir>/params.txt
#   DATASET_ROOT     clip_benchmark dataset_root (must contain flickr30k/, val2014/, etc.)
# Optional:
#   RESULTS_ROOT     where to write JSON outputs (default: <CLIP_benchmark>/results)
#   CUDA_VISIBLE_DEVICES  single-GPU index (default: 0)
#
# Edit MODEL_SPECS below to match your checkpoints. Each entry is:
#   "tag|model_dir|model_arch|epoch_list"
# `epoch_list` is space-separated.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

: "${CHECKPOINT_ROOT:?must set CHECKPOINT_ROOT (dir of <model>/checkpoints/epoch_*/)}"
: "${DATASET_ROOT:?must set DATASET_ROOT (clip_benchmark dataset_root)}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/results}"

declare -a MODEL_SPECS=(
    # Edit these to match your checkpoints. Examples:
    # "400mv2|400Mv2_lr0p001-...-rope_dp|ViT-B-16-SigLIP|49"
    # "coca-do0|coca_mixed_contrastive_dropout0_...-rope_dp|coca_ViT-B-32-siglip|1 2 3 4"
)

if [ "${#MODEL_SPECS[@]}" -eq 0 ]; then
    echo "ERROR: MODEL_SPECS is empty — edit $0 and add your checkpoints" >&2
    exit 1
fi

for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r tag model_name model_arch epochs <<< "$spec"
    pretrained_dir="${CHECKPOINT_ROOT}/${model_name}/checkpoints"
    for epoch in $epochs; do
        pretrained_path="${pretrained_dir}/epoch_${epoch}"
        for dataset in mscoco_captions flickr30k; do
            output_pattern="${RESULTS_ROOT}/${model_name}/epoch_${epoch}/{dataset}_{model}_{language}_{task}.json"
            echo "=== ${tag} epoch_${epoch} ${dataset} (retrieval) ==="
            CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m clip_benchmark.cli_rope eval \
                --model_type open_clip \
                --model "$model_arch" \
                --pretrained "$pretrained_path" \
                --task zeroshot_retrieval \
                --dataset_root "$DATASET_ROOT" \
                --dataset "$dataset" \
                --output "$output_pattern" \
                --batch_size 64 \
                --language en \
                --recall_k 1 5 \
                --is-fsdp \
                --skip_existing
        done
    done
done

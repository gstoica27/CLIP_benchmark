#!/bin/bash
# Loop zeroshot classification on ImageNet-1k val (50K images, 1000 classes)
# over every (model, epoch) in MODEL_SPECS.
#
# Required env vars:
#   CHECKPOINT_ROOT  dir with <model_dir>/checkpoints/epoch_<N>/  +  <model_dir>/params.txt
#   IMAGENET_ROOT    dir with val/<wnid>/*.JPEG  +  ILSVRC2012_devkit_t12.tar.gz
# Optional:
#   RESULTS_ROOT     where to write JSON outputs (default: <CLIP_benchmark>/results)
#   CUDA_VISIBLE_DEVICES  single-GPU index (default: 0)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

: "${CHECKPOINT_ROOT:?must set CHECKPOINT_ROOT (dir of <model>/checkpoints/epoch_*/)}"
: "${IMAGENET_ROOT:?must set IMAGENET_ROOT (dir with val/ + ILSVRC2012_devkit_t12.tar.gz)}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/results}"

declare -a MODEL_SPECS=(
    # "tag|model_dir|model_arch|epoch_list"
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
        output_pattern="${RESULTS_ROOT}/${model_name}/epoch_${epoch}/{dataset}_{model}_{language}_{task}.json"
        echo "=== ${tag} epoch_${epoch} imagenet1k ==="
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m clip_benchmark.cli_rope eval \
            --model_type open_clip \
            --model "$model_arch" \
            --pretrained "$pretrained_path" \
            --task zeroshot_classification \
            --dataset_root "$IMAGENET_ROOT" \
            --dataset imagenet1k \
            --output "$output_pattern" \
            --batch_size 128 \
            --language en \
            --is-fsdp \
            --skip_existing
    done
done

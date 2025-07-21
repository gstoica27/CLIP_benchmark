#!/bin/bash
# This script runs a set of checkpoints from the training of a model on the CLIP benchmark.
# experiment_name="vitl_baseline_8kBs"
# experiment_name="ogdense_multipos3_4kBs"
model_source="open_clip"
# model_architecture="ViT-SO400M-14-SigLIP"
model_architecture="ViT-B-16-SigLIP"
# pretrained_dir="/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp15M/logs/model_ViT-B-16-SigLIP-lr_0.001-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_128-j_1-p_amp-targ_dense_multipos-NumSent_3/checkpoints"
# pretrained_dir="/weka/oe-training-default/georges/checkpoints/ViT_SO400M_14/DataComp15M/logs/original_captions/model_ViT-SO400M-14-SigLIP-lr_0.001-b_128-j_1-p_amp-target_dense_multipos-NumSent_2/checkpoints" # our model
# pretrained_dir="/weka/oe-training-default/georges/checkpoints/ViT_SO400M_14/DataComp15M/logs/original_captions/model_ViT-SO400M-14-SigLIP-lr_0.001-b_128-j_1-p_amp/checkpoints" # original model
# pretrained_dir="/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp15M/logs/model_ViT-B-16-SigLIP-lr_0.001-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-ns_3-ocp_0.75v2/checkpoints"
task="zeroshot_classification"
benchmark_datasets_root="/weka/oe-data-default/georges/datasets/clip_benchmark"
dataset_type="vtab"
# output_pattern="/weka/prior-default/georges/research/open_clip/CLIP_benchmark/clip_benchmark/vitb16_datacomp15m/${experiment_name}/{dataset}_{pretrained}_{model}_{language}_{task}.json"
# output_pattern="/weka/prior-default/georges/research/open_clip/CLIP_benchmark/clip_benchmark/vitl14_datacomp15m/${experiment_name}/{dataset}_{pretrained}_{model}_{language}_{task}.json"

MAX_EPOCHS=27
MIN_EPOCHS=20
# MAX_EPOCHS=55
MODELS_TO_RUN=(
    "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp15M/logs/model_ViT-B-16-SigLIP-lr_0.001-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-ns_5-ocp_0.75v2/checkpoints"
    "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp15M/logs/model_ViT-B-16-SigLIP-lr_0.001-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-ns_3-ocp_0.75v2/checkpoints"
    "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp15M/logs/model_ViT-B-16-SigLIP-lr_0.001-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-ns_1-ocp_0.75v2/checkpoints"
    "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp15M/logs/model_ViT-B-16-SigLIP-lr_0.001-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-ns_5-ocp_0.5v2/checkpoints/"
    "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp15M/logs/model_ViT-B-16-SigLIP-lr_0.001-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-ns_3-ocp_0.5v2/checkpoints/"
    "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp15M/logs/model_ViT-B-16-SigLIP-lr_0.001-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-ns_1-ocp_0.5v2/checkpoints/"
    "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp15M/logs/model_ViT-B-16-SigLIP-lr_0.001-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-ns_5-ocp_0.25v2/checkpoints/"
    "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp15M/logs/model_ViT-B-16-SigLIP-lr_0.001-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-ns_3-ocp_0.25v2/checkpoints/"
    "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp15M/logs/model_ViT-B-16-SigLIP-lr_0.001-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-ns_1-ocp_0.25v2/checkpoints/"
)
EXPERIMENT_NAMES=(
    "vitb16_datacomp15m_ns5_ocp0p75_4kBs"
    "vitb16_datacomp15m_ns3_ocp0p75_4kBs"
    "vitb16_datacomp15m_ns1_ocp0p75_4kBs"
    "vitb16_datacomp15m_ns5_ocp0p5_4kBs"
    "vitb16_datacomp15m_ns3_ocp0p5_4kBs"
    "vitb16_datacomp15m_ns1_ocp0p5_4kBs"
    "vitb16_datacomp15m_ns5_ocp0p25_4kBs"
    "vitb16_datacomp15m_ns3_ocp0p25_4kBs"
    "vitb16_datacomp15m_ns1_ocp0p25_4kBs"
)

if (( ${#MODELS_TO_RUN[@]} != ${#EXPERIMENT_NAMES[@]} )); then
  echo "⚠️  Arrays are different lengths!" >&2
  exit 1
fi

for i in "${!MODELS_TO_RUN[@]}"; do
    pretrained_dir="${MODELS_TO_RUN[i]}"
    experiment_name="${EXPERIMENT_NAMES[i]}"
    output_pattern="/weka/prior-default/georges/research/open_clip/CLIP_benchmark/clip_benchmark/vitb16_datacomp15m/${experiment_name}/{dataset}_{pretrained}_{model}_{language}_{task}.json"
    for epoch in $(seq $MAX_EPOCHS -1 $MIN_EPOCHS); do
        pretrained_path="${pretrained_dir}/epoch_${epoch}.pt"
        CUDA_VISIBLE_DEVICES=0 python -m clip_benchmark.cli eval \
            --model_type $model_source \
            --model $model_architecture \
            --pretrained $pretrained_path \
            --task $task \
            --dataset_root $benchmark_datasets_root \
            --dataset $dataset_type \
            --output $output_pattern
    done
done
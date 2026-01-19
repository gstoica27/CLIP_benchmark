#!/bin/bash
# This script runs a set of checkpoints from the training of a model on the CLIP benchmark.
# experiment_name="baseline_4kBs"
# experiment_name="vitl_multipos2_8kBs"
model_source="open_clip"
model_architecture="ViT-B-16-SigLIP"
# model_architecture="ViT-B-16-quickgelu"
# model_architecture="ViT-SO400M-14-SigLIP"
# pretrained_dir="/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp15M/logs/model_ViT-B-16-SigLIP-lr_0.001-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_128-j_1-p_amp-targ_dense_multipos-NumSent_3/checkpoints" # our model
# pretrained_dir="/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp15M/logs/model_ViT-B-16-SigLIP-lr_0.001-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_128-j_1-p_amp-targ_original/checkpoints" # original model
# pretrained_dir="/weka/oe-training-default/georges/checkpoints/ViT_SO400M_14/DataComp15M/logs/original_captions/model_ViT-SO400M-14-SigLIP-lr_0.001-b_128-j_1-p_amp-target_dense_multipos-NumSent_2/checkpoints" # our model
# pretrained_dir="/weka/oe-training-default/georges/checkpoints/ViT_SO400M_14/DataComp15M/logs/original_captions/model_ViT-SO400M-14-SigLIP-lr_0.001-b_128-j_1-p_amp/checkpoints" # original model
benchmark_datasets_root="/weka/oe-training-default/georges/datasets/clip_benchmark"
task="zeroshot_retrieval"
# dataset=mscoco_captions
# dataset=flickr30k
# output_pattern="/weka/prior-default/georges/research/open_clip/CLIP_benchmark/clip_benchmark/vitl14_datacomp15m/${experiment_name}/{dataset}_{pretrained}_{model}_{language}_{task}.json"
# output_pattern="/weka/prior-default/georges/research/open_clip/CLIP_benchmark/clip_benchmark/vitb16_datacomp15m/${experiment_name}/{dataset}_{pretrained}_{model}_{language}_{task}.json"

MAX_EPOCHS=11
MIN_EPOCHS=11
# MAX_EPOCHS=5
# MIN_EPOCHS=5
# MAX_EPOCHS=55
CHECKPOINT_ROOT="/weka/oe-training-default/georges/checkpoints/ViT_B_16/ablations/DataComp50M_MetaclipBased/logs/"
MODELS_TO_RUN=(
    # "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp100M_MetaclipBased/logs/100M-16K2model_ViT-B-16-SigLIP-lr_0.001-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-dense_og_disj-ns_5/checkpoints"
    # "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp100M_MetaclipBased/logs/100MFD-16Kmodel_ViT-B-16-SigLIP-lr_0.005-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-dense_og_disj-ns_5/checkpoints"
    # "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp100M_MetaclipBased/logs/100M-16Kmodel_ViT-B-16-SigLIP-lr_0.004-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-dense_og_disj-ns_5/checkpoints"
    # "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp100M_MetaclipBased/logs/100M-16Kmodel_ViT-B-16-SigLIP-lr_0.004-wd_0.01-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-dense_og_disj-ns_5/checkpoints"
    # "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp100M_MetaclipBased/logs/100MFD-16Kmodel_ViT-B-16-SigLIP-lr_0.005-wd_0.01-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-dense_og_disj-ns_5/checkpoints"
    # Training for 30.5K steps | Varying Warmup
    # "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp100M_MetaclipBased/logs/100M-16Kmodel_ViT-B-16-SigLIP-lr_0.005-wd_0.01-opt_adamw-w_10000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-dense_og_disj-ns_5/checkpoints"
    # "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp100M_MetaclipBased/logs/100M-16Kmodel_ViT-B-16-SigLIP-lr_0.005-wd_0.01-opt_adamw-w_5000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-dense_og_disj-ns_5/checkpoints"
    # Training for 61K steps
    # "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp100M_MetaclipBased/logs/100M-16K3model_ViT-B-16-SigLIP-lr_0.004-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-dense_og_disj-ns_5/checkpoints"
    # "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp100M_MetaclipBased/logs/100M-16K3model_ViT-B-16-SigLIP-lr_0.005-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-dense_og_disj-ns_5/checkpoints"
    # "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp100M_MetaclipBased/logs/100M-16K3model_ViT-B-16-SigLIP-lr_0.005-wd_0.01-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-dense_og_disj-ns_5/checkpoints"
    # "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp100M_MetaclipBased/logs/100M-16K3model_ViT-B-16-SigLIP-lr_0.001-wd_0.0001-opt_adamw-w_20000-b1_0.9-b2_0.95-gc_1.0-bs_256-j_1-p_amp-dense_og_disj-ns_5/checkpoints"
    # "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp100M_MetaclipBased/logs/100M-16K3model_ViT-B-16-SigLIP-lr_0.005-wd_0.0001-opt_adamw-w_20K-b1_0.9-b2_0.95-gc_1.0-j_1-p_amp-pm_br_0.0625_tr_0.4-ms_31K/checkpoints"
    # "/weka/oe-training-default/georges/checkpoints/ViT_B_16/DataComp100M_MetaclipBased/logs/100M-16K3model_ViT-B-16-SigLIP-lr_0.005-wd_0.01-w_20K-b1_0.9-b2_0.95-p_amp-ms_61K-rope_delpos/checkpoints"
    # Training for 781K steps
    # "/weka/oe-training-default/georges/checkpoints/ViT_B_16/v0DataMixture_MetaclipBased/logs/v0-exp3model_ViT-B-16-SigLIP-lr_0.005-wd_0.01-w_20K-b1_0.9-b2_0.95-p_amp_bf16-l_1.0-ms_781K-rope_delpos-ijepa_head_l_1.0/checkpoints"
    # "metaclip_400m"
    # "SigLIP_B16"
    # Ablations on 50M
    # "50M_Abl_SL_lr_0.001-wd_0.01-w_20K-b1_0.9-b2_0.95-p_amp_bf16-l_1.0-ms_123K-rope_dp"
    # "50M_Abl_2_lr_0.001-wd_0.01-w_20K-b1_0.9-b2_0.95-p_amp_bf16-l_1.0-ms_123K-rope_dp"
    # "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-pamp_bf16-l1p0-ms123K-rope_dp-dino_c2_l0p5"
    # "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-ms123K-rope_dp-dino_c2_l1p0"
    "50M_Abl_2_lr_0.001-wd_0.01-w_20K-b1_0.9-b2_0.95-p_amp_bf16-l_1.0-ms_123K-rope_dp"
    "50M_Abl_2_lr_0.001-wd_0.01-w_20K-b1_0.9-b2_0.95-p_amp_bf16-l_1.0-ms_123K-rope_dp-ijepa_all_l_0.5"
    "50M_Abl_2_lr_0.001-wd_0.01-w_20K-b1_0.9-b2_0.95-p_amp_bf16-l_1.0-ms_123K-rope_dp-ijepa_head_l_0.5"
    "50M_Abl_2_ViT-B-16-SigLIP-Qwen-0.6B-DataComp50M-Batch_16K-Steps_2K-LR_1e-3-WD_1e-2"
    "50M_Abl_lr_0.001-wd_0.01-w_20K-b1_0.9-b2_0.95-p_amp_bf16-l_1.0-ms_123K-rope_dp-dino_c_5_l_0.5"
    "50M_Abl_lr_0.001-wd_0.01-w_20K-b1_0.9-b2_0.95-p_amp_bf16-l_1.0-ms_123K-rope_dp-ijepa_head_l_0.5-dino_c_5_l_0.5"
    "50M_Abl_lr_0.001-wd_0.01-w_20K-b1_0.9-b2_0.95-p_amp_bf16-l_1.0-pmpr_br_0.0625_tr_0.4_l_0.1-ms_123K-rope_dp-dino_c_5_l_0.5"
    "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-cna_ct0p5af1p0g1p0-ms123K-rope_dp"
    "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-cna_ct0p8af10p0g1p0-ms123K-rope_dp"
    "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-cna_ct0p8af1p0g1p0-ms123K-rope_dp"
    "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-ms123K-rope_dp-dino_c2_l1p0"
    "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-ms123K-rope_dp-dino_c3_l0p5"
    "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-ms123K-rope_dp-dino_c5_l0p5"
    "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-pmpr_br0p0625_tr0p4_l0p1-ms123K-rope_dp-dino_c2_l0p5"
    "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-pmpr_br0p0625_tr0p4_l0p1-ms123K-rope_dp-dino_c5_l0p5"
    "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-pmpr_br0p0625_tr0p4_l0p1-ms123K-rope_dp-ijepaHead_l0p5"
    "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-pmpr_br0p0625_tr0p4_l0p5-ms123K-rope_dp-dino_c5_l0p5"
    "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-pamp_bf16-l1p0-fa0p5-ms123K-rope_dp"
    "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-pamp_bf16-l1p0-fa0p8-ms123K-rope_dp"
    "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-pamp_bf16-l1p0-fg1p5-ms123K-rope_dp"
    "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-pamp_bf16-l1p0-fg2p0-fa0p8-ms123K-rope_dp"
    "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-pamp_bf16-l1p0-fg2p0-ms123K-rope_dp"
    "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-pamp_bf16-l1p0-ms123K-rope_dp-dino_c2_l0p5"
    "50M_Abl_SL_lr_0.001-wd_0.01-w_20K-b1_0.9-b2_0.95-p_amp_bf16-l_1.0-ms_123K-rope_dp"
    "50M_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-cna_ct0p8af1p0g1p0CnstE1-ms123K-rope_dp"
    "50M_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-cna_ct0p8af1p0g1p0CnstE5-ms123K-rope_dp"
    "50M_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-cna_ct0p8af1p0g1p0Exp-ms123K-rope_dp"
    "50M_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-cna_ct0p8af1p0g1p0ExpS-ms123K-rope_dp"
    "50M_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-cna_ct0p8af1p0g1p0Lin-ms123K-rope_dp"
    "50M_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-dmix-ms123K-rope_dp"
    "50M_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-pmpr_br0p0625_tr0p4_l0p5-ms123K-rope_dp-ijepaHead_l0p5-dino_c5_l0p5"
    "50Mv2_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-ms123K-rope"
    "ViTB16_SigLIP-DataComp50M-Batch_16K-Steps_2K-LR_1e-3-WD_1e-2"
)
EXPERIMENT_NAMES=(
     # Training for 30.5K steps | Constant Warmup
    # "vitb16_datacomp100m_lr0p001_wd0p0001_warmup20K_ms30p5K"
    # "vitb16_datacomp100m_lr0p005_wd0p0001_warmup20K_ms30p5K"
    # "vitb16_datacomp100m_lr0p004_wd0p0001_warmup20K_ms30p5K"
    # "vitb16_datacomp100m_lr0p004_wd0p01_warmup20K_ms30p5K"
    # "vitb16_datacomp100m_lr0p005_wd0p01_warmup20K_ms30p5K"
    # Training for 30.5K steps | Varying Warmup
    # "vitb16_datacomp100m_lr0p005_wd0p01_warmup10K_ms30p5K"
    # "vitb16_datacomp100m_lr0p005_wd0p01_warmup5K_ms30p5K"
    # Training for 61K steps
    # "vitb16_datacomp100m_lr0p004_wd0p0001_warmup20K_ms61K"
    # "vitb16_datacomp100m_lr0p005_wd0p0001_warmup20K_ms61K"
    # "vitb16_datacomp100m_lr0p005_wd0p01_warmup20K_ms61K"
    # "vitb16_datacomp100m_lr0p001_wd0p0001_warmup20K_ms61K"
    # "vitb16_datacomp100m_lr0p005_wd0p0001_warmup20K_ms30p5K_maskreg"
    # "vitb16_datacomp100m_lr0p005_wd0p01_warmup20K_ms61K_RoPE_DelPosEmb"
    # Training for 781K steps
    # "vitb16_datacomp100m_lr0p005_wd0p01_warmup20K_ms781K_RoPE_DelPosEmb"
    # "vit_b16_metaclip400m"
    # "hf-hub:timm/ViT-B-16-SigLIP"
     # Ablations on 50M
    # "50M_Abl_SL_lr_0.001-wd_0.01-w_20K-b1_0.9-b2_0.95-p_amp_bf16-l_1.0-ms_123K-rope_dp"
    # "50M_Abl_2_lr_0.001-wd_0.01-w_20K-b1_0.9-b2_0.95-p_amp_bf16-l_1.0-ms_123K-rope_dp"
    # "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-pamp_bf16-l1p0-ms123K-rope_dp-dino_c2_l0p5"
    # "50M_Abl_lr0p001-wd0p01-w20K-b10p9-b20p95-bs_4p1K-pamp_bf16-l1p0-ms123K-rope_dp-dino_c2_l1p0"
)

# if (( ${#MODELS_TO_RUN[@]} != ${#EXPERIMENT_NAMES[@]} )); then
#   echo "⚠️  Arrays are different lengths!" >&2
#   exit 1
# fi

for i in "${!MODELS_TO_RUN[@]}"; do
    pretrained_dir="${CHECKPOINT_ROOT}/${MODELS_TO_RUN[i]}/checkpoints"
    # experiment_name="${EXPERIMENT_NAMES[i]}"
    experiment_name="${MODELS_TO_RUN[i]}"
    echo "Evaluating dir ${pretrained_dir} with experiment name ${experiment_name}"
    for dataset in "mscoco_captions flickr30k"; do
        # output_pattern="/weka/prior-default/georges/research/open_clip/CLIP_benchmark/clip_benchmark/vitb16_datacomp100m/${experiment_name}/{dataset}_{pretrained}_{model}_{language}_{task}.json"
        # output_pattern="/weka/prior-default/georges/research/open_clip/CLIP_benchmark/clip_benchmark/vitb16_datacomp100m_v0/${experiment_name}/{dataset}_{pretrained}_{model}_{language}_{task}.json"
        # output_pattern="/weka/prior-default/georges/research/open_clip/CLIP_benchmark/clip_benchmark/siglip/${experiment_name}/{dataset}_{pretrained}_{model}_{language}_{task}.json"
        for epoch in $(seq $MAX_EPOCHS -1 $MIN_EPOCHS); do
            output_pattern="/weka/prior-default/georges/research/open_clip/CLIP_benchmark/clip_benchmark/vitb16_datacomp50m/epoch_${epoch}/${experiment_name}/{dataset}_{model}_{language}_{task}.json"
            # pretrained_path="${pretrained_dir}/epoch_${epoch}.pt"
            pretrained_path="${pretrained_dir}/epoch_${epoch}"
            # pretrained_path="${pretrained_dir}"
            CUDA_VISIBLE_DEVICES=1 python -m clip_benchmark.cli_rope eval \
                --model_type $model_source \
                --model $model_architecture \
                --pretrained $pretrained_path \
                --task $task \
                --dataset_root $benchmark_datasets_root \
                --dataset $dataset \
                --output $output_pattern \
                --batch_size 64 \
                --language "en" \
                --recall_k 1 5 \
                --is-fsdp \
                --skip_existing
        done
    done
done
#!/bin/bash
# Auto-discover every model dir under $CHECKPOINT_ROOT, find its latest epoch,
# and submit (at most) two Beaker jobs per model:
#   1. zeroshot_retrieval (mscoco_captions + flickr30k) — one job
#   2. zeroshot_classification on imagenet1k          — one job
#
# Each job is only submitted if its expected output JSON(s) are missing under
# $RESULTS_ROOT. Re-fire the script anytime — it is idempotent.
#
# Expected layout of $CHECKPOINT_ROOT:
#   $CHECKPOINT_ROOT/
#   ├── <model_dir>/
#   │   ├── params.txt                    (must have a `model: <arch>` line)
#   │   └── checkpoints/
#   │       ├── epoch_1/, epoch_2/, ...   (FSDP2 distcp shards under each)
#   │       └── epoch_<N>/                (the highest N → "latest")
#   └── ...
#
# Required env vars:
#   CHECKPOINT_ROOT  parent dir holding all model dirs
#   DATASET_ROOT     clip_benchmark dataset_root (must contain flickr30k/, val2014/, …)
#   IMAGENET_ROOT    imagenet val dir (val/ + ILSVRC2012_devkit_t12.tar.gz)
# Optional:
#   RESULTS_ROOT     where to write JSON outputs   (default: <CLIP_benchmark>/results)
#   OPEN_CLIP_REPO   absolute path to the cloned open_clip repo for beaker `cd`
#                                                  (default: derived from this script)
#   BEAKER_HOME      $HOME inside the container    (default: $HOME on this machine)
#   CONDA_ENV        conda env name                (default: trainer)
#   WORKSPACE        gantry workspace              (default: ai2/oe-encoder)
#   BUDGET           gantry budget                 (default: ai2/oe-omai)
#   PRIORITY         gantry priority               (default: high)
#   BEAKER_IMAGE     gantry beaker image           (default: sanghol/molmo2-torch2.7.1-cuda12.8)
#
# Flags (mutually exclusive in spirit; --force also valid with default submit mode):
#   (no flag)        submit beaker jobs for every (model, task) whose JSON is missing
#   --status         print a results table (latest epoch per model + the metric
#                    numbers extracted from existing JSONs); no submissions
#   --dry-run        like default but only print what would be submitted, fire nothing
#   --force          submit every job regardless of existing JSONs
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIP_BENCH_DIR="$(dirname "${SCRIPT_DIR}")"
DEFAULT_REPO="$(dirname "${CLIP_BENCH_DIR}")"

: "${CHECKPOINT_ROOT:?must set CHECKPOINT_ROOT}"
: "${DATASET_ROOT:?must set DATASET_ROOT (clip_benchmark dataset_root for retrieval)}"
: "${IMAGENET_ROOT:?must set IMAGENET_ROOT (imagenet val + devkit)}"
RESULTS_ROOT="${RESULTS_ROOT:-${CLIP_BENCH_DIR}/results}"
OPEN_CLIP_REPO="${OPEN_CLIP_REPO:-${DEFAULT_REPO}}"
BEAKER_HOME="${BEAKER_HOME:-${HOME}}"
CONDA_ENV="${CONDA_ENV:-trainer}"
WORKSPACE="${WORKSPACE:-ai2/oe-encoder-data}"
BUDGET="${BUDGET:-ai2/oe-omai}"
PRIORITY="${PRIORITY:-high}"
BEAKER_IMAGE="${BEAKER_IMAGE:-sanghol/molmo2-torch2.7.1-cuda12.8}"

# Flags
STATUS_MODE=false
DRY_RUN=false
FORCE=false
for arg in "$@"; do
    case "$arg" in
        --status)  STATUS_MODE=true ;;
        --dry-run) DRY_RUN=true ;;
        --force)   FORCE=true ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# Return the highest N from $1/checkpoints/epoch_N/ (echoes empty string if none).
find_latest_epoch() {
    local ckpt_dir="$1/checkpoints"
    [ -d "$ckpt_dir" ] || return 0
    ls -1 "$ckpt_dir" 2>/dev/null \
        | grep -E '^epoch_[0-9]+$' \
        | sort -t_ -k2 -n \
        | tail -1 \
        | sed 's/^epoch_//'
}

# Echo the `model: <arch>` value from $1/params.txt (empty if missing).
read_model_arch() {
    local pt="$1/params.txt"
    [ -f "$pt" ] || return 0
    grep -E '^model:[[:space:]]+' "$pt" | head -1 | sed -E 's/^model:[[:space:]]+//; s/[[:space:]]+$//'
}

# Short tag: lowercased model_dir, alnum + dash only, truncated.
short_tag() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/-+$//; s/^-+//' | cut -c1-60
}

submit_retrieval() {
    local model_name="$1" model_arch="$2" epoch="$3"
    local pretrained_path="${CHECKPOINT_ROOT}/${model_name}/checkpoints/epoch_${epoch}"
    local out_dir="${RESULTS_ROOT}/${model_name}/epoch_${epoch}"
    local flickr_json="${out_dir}/flickr30k_${model_arch}_en_zeroshot_retrieval.json"
    local mscoco_json="${out_dir}/mscoco_captions_${model_arch}_en_zeroshot_retrieval.json"

    if [ "$FORCE" != "true" ] && [ -f "$flickr_json" ] && [ -f "$mscoco_json" ]; then
        echo "  [skip retrieval] both JSONs present at ${out_dir}"
        return 0
    fi
    local tag; tag=$(short_tag "$model_name")
    local job_name="clipbench_retr_${tag}"
    local output_pattern="${out_dir}/{dataset}_{model}_{language}_{task}.json"

    if [ "$DRY_RUN" == "true" ]; then
        echo "  [dry-run] would submit retrieval for ${model_name} epoch_${epoch} (arch=${model_arch}) → ${job_name}"
        return 0
    fi
    echo "  [submit retr]    ${model_name} epoch_${epoch} → ${job_name}"
    gantry run --allow-dirty \
        --task-name "$job_name" --name "$job_name" \
        --workspace "$WORKSPACE" \
        --cluster ai2/saturn --cluster ai2/jupiter --cluster ai2/ceres \
        --gpus 1 --priority "$PRIORITY" --shared-memory 512GiB \
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
}

submit_imagenet() {
    local model_name="$1" model_arch="$2" epoch="$3"
    local pretrained_path="${CHECKPOINT_ROOT}/${model_name}/checkpoints/epoch_${epoch}"
    local out_dir="${RESULTS_ROOT}/${model_name}/epoch_${epoch}"
    local imagenet_json="${out_dir}/imagenet1k_${model_arch}_en_zeroshot_classification.json"

    if [ "$FORCE" != "true" ] && [ -f "$imagenet_json" ]; then
        echo "  [skip imagenet]  JSON present at ${imagenet_json}"
        return 0
    fi
    local tag; tag=$(short_tag "$model_name")
    local job_name="clipbench_in1k_${tag}"
    local output_pattern="${out_dir}/{dataset}_{model}_{language}_{task}.json"

    if [ "$DRY_RUN" == "true" ]; then
        echo "  [dry-run] would submit imagenet for ${model_name} epoch_${epoch} (arch=${model_arch}) → ${job_name}"
        return 0
    fi
    echo "  [submit in1k]    ${model_name} epoch_${epoch} → ${job_name}"
    gantry run --allow-dirty \
        --task-name "$job_name" --name "$job_name" \
        --workspace "$WORKSPACE" \
        --cluster ai2/saturn --cluster ai2/jupiter --cluster ai2/ceres \
        --gpus 1 --priority "$PRIORITY" --shared-memory 512GiB \
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
                --dataset_root ${IMAGENET_ROOT} \
                --dataset imagenet1k \
                --output ${output_pattern} \
                --batch_size 128 \
                --language en \
                --is-fsdp \
                --skip_existing"
}

# ---- --status: print results table and exit ----
if [ "$STATUS_MODE" == "true" ]; then
    CHECKPOINT_ROOT="$CHECKPOINT_ROOT" RESULTS_ROOT="$RESULTS_ROOT" \
    python3 - <<'PYEOF'
import json, os, re

CKPT = os.environ["CHECKPOINT_ROOT"]
RES  = os.environ["RESULTS_ROOT"]
EP_RE = re.compile(r'^epoch_(\d+)$')

def latest_epoch(model_dir):
    cps = os.path.join(model_dir, "checkpoints")
    if not os.path.isdir(cps): return None
    epochs = [int(m.group(1)) for f in os.listdir(cps) for m in [EP_RE.match(f)] if m]
    return max(epochs) if epochs else None

def read_arch(model_dir):
    pt = os.path.join(model_dir, "params.txt")
    if not os.path.isfile(pt): return None
    with open(pt) as f:
        for line in f:
            if line.startswith("model:"):
                return line.split(":", 1)[1].strip()
    return None

def load(path):
    if not os.path.isfile(path): return None
    try:
        with open(path) as f: return json.load(f)
    except Exception: return None

rows = []
for name in sorted(os.listdir(CKPT)):
    md = os.path.join(CKPT, name)
    if not os.path.isdir(md): continue
    ep = latest_epoch(md)
    arch = read_arch(md)
    if ep is None or arch is None:
        rows.append((name, ep, arch, None, None, None, None, None))
        continue
    base = os.path.join(RES, name, f"epoch_{ep}")
    fl  = load(os.path.join(base, f"flickr30k_{arch}_en_zeroshot_retrieval.json"))
    ms  = load(os.path.join(base, f"mscoco_captions_{arch}_en_zeroshot_retrieval.json"))
    im  = load(os.path.join(base, f"imagenet1k_{arch}_en_zeroshot_classification.json"))
    fl_i = fl["metrics"]["image_retrieval_recall@1"] if fl else None
    fl_t = fl["metrics"]["text_retrieval_recall@1"]  if fl else None
    ms_i = ms["metrics"]["image_retrieval_recall@1"] if ms else None
    ms_t = ms["metrics"]["text_retrieval_recall@1"]  if ms else None
    in1k = (im["metrics"].get("acc1") or im["metrics"].get("acc")) if im else None
    rows.append((name, ep, arch, fl_i, fl_t, ms_i, ms_t, in1k))

# Table
def fmt(v): return f"{v:.3f}" if isinstance(v, float) else "—"
name_w = max((len(r[0]) for r in rows), default=10)
name_w = min(name_w, 60)
hdr = f"{'model':<{name_w}}  {'epoch':>5}  {'fl I@1':>6}  {'fl T@1':>6}  {'ms I@1':>6}  {'ms T@1':>6}  {'in1k':>6}"
print(hdr)
print("-" * len(hdr))
for name, ep, arch, fl_i, fl_t, ms_i, ms_t, in1k in rows:
    short = name if len(name) <= name_w else name[:name_w-1] + "…"
    ep_s  = str(ep) if ep is not None else "—"
    print(f"{short:<{name_w}}  {ep_s:>5}  {fmt(fl_i):>6}  {fmt(fl_t):>6}  {fmt(ms_i):>6}  {fmt(ms_t):>6}  {fmt(in1k):>6}")

# Mark missing
n_total = len(rows)
n_full  = sum(1 for r in rows if all(v is not None for v in r[3:]))
print()
print(f"summary: {n_full}/{n_total} model(s) have all three eval results")
PYEOF
    exit 0
fi

# ---- main loop ----
echo "scanning $CHECKPOINT_ROOT for models..."
n_models=0; n_skip_retr=0; n_skip_in1k=0; n_no_epoch=0; n_no_arch=0
for model_dir in "$CHECKPOINT_ROOT"/*/; do
    model_name="$(basename "${model_dir%/}")"
    epoch=$(find_latest_epoch "${model_dir%/}")
    if [ -z "$epoch" ]; then
        echo "[skip] $model_name : no epoch_N/ under checkpoints/" >&2
        n_no_epoch=$((n_no_epoch+1))
        continue
    fi
    model_arch=$(read_model_arch "${model_dir%/}")
    if [ -z "$model_arch" ]; then
        echo "[skip] $model_name : params.txt missing or no 'model:' line" >&2
        n_no_arch=$((n_no_arch+1))
        continue
    fi
    n_models=$((n_models+1))
    echo "[model] $model_name : epoch_${epoch} (arch=${model_arch})"

    # Count what would be skipped, then submit if needed
    flickr_json="${RESULTS_ROOT}/${model_name}/epoch_${epoch}/flickr30k_${model_arch}_en_zeroshot_retrieval.json"
    mscoco_json="${RESULTS_ROOT}/${model_name}/epoch_${epoch}/mscoco_captions_${model_arch}_en_zeroshot_retrieval.json"
    imagenet_json="${RESULTS_ROOT}/${model_name}/epoch_${epoch}/imagenet1k_${model_arch}_en_zeroshot_classification.json"
    [ -f "$flickr_json" ] && [ -f "$mscoco_json" ] && n_skip_retr=$((n_skip_retr+1))
    [ -f "$imagenet_json" ] && n_skip_in1k=$((n_skip_in1k+1))

    submit_retrieval "$model_name" "$model_arch" "$epoch"
    submit_imagenet  "$model_name" "$model_arch" "$epoch"
done

echo
echo "summary: ${n_models} model(s) scanned; ${n_skip_retr} retrieval already-done, ${n_skip_in1k} imagenet already-done"
[ "$n_no_epoch" -gt 0 ] && echo "  ${n_no_epoch} model(s) skipped: no checkpoints/epoch_N/"
[ "$n_no_arch"  -gt 0 ] && echo "  ${n_no_arch} model(s) skipped: no model: in params.txt"
[ "$DRY_RUN" == "true" ] && echo "(--dry-run — nothing was submitted)"




# CHECKPOINT_ROOT=/weka/prior-default/chenhaoz/home/open_clip/important_ckpt DATASET_ROOT=/weka/oe-training-default/georges/datasets/clip_benchmark IMAGENET_ROOT=/weka/prior-default/georges/datasets/imagenet1k/imagenet bash ../open_clip/CLIP_benchmark/submit_jobs/auto_submit_retrieval_imagenet.sh
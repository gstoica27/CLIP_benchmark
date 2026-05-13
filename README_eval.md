# CLIP / CoCa eval — retrieval & zero-shot classification

How to run `clip_benchmark` against `open_clip_train` FSDP2-sharded
checkpoints. Covers retrieval (MSCOCO, Flickr30k), zero-shot classification
(VTAB-1k, ImageNet-1k), and one-Beaker-job-per-(checkpoint, task) fan-out.

No paths are hardcoded — everything is parameterized by the env vars
documented below, plus a `MODEL_SPECS` array you fill in at the top of each
sweep script.

---

## 1. Expected file layout

The eval expects two layout invariants:

### A. Repo layout (after `git clone`)

```
open_clip/                                    ← the repo (this open_clip checkout)
├── src/open_clip/
│   ├── drop_in_replacements.py               ← used by cli_rope at eval time
│   └── ...
└── CLIP_benchmark/                           ← this directory
    ├── clip_benchmark/
    │   ├── cli_rope.py                       ← main eval entrypoint
    │   ├── models/open_clip.py               ← FSDP2 loader, forces local open_clip on sys.path
    │   ├── metrics/, datasets/
    ├── eval_retrieval_over_epochs.sh         ← single-GPU sweep scripts
    ├── eval_vtab_over_epochs.sh
    ├── eval_imagenet_over_epochs.sh
    ├── run_all_evals.sh                      ← convenience: retrieval + VTAB + ImageNet + aggregate
    ├── submit_jobs/                          ← one-Beaker-job-per-checkpoint scripts
    │   ├── eval_retrieval_per_ckpt.sh
    │   ├── eval_vtab_per_ckpt.sh
    │   └── eval_imagenet_per_ckpt.sh
    ├── aggregate_results.py                  ← results/*.json → CSV
    └── README_eval.md                        ← this file
```

The Python entry derives the local `open_clip/src` path from
`__file__`, so cloning the repo to **anywhere** works as long as
`src/open_clip/` and `CLIP_benchmark/clip_benchmark/` live in the same
parent directory. Override with `OPEN_CLIP_SRC=...` if your layout differs.

### B. Checkpoint layout (`$CHECKPOINT_ROOT`)

Each model's checkpoint must look like:

```
$CHECKPOINT_ROOT/
└── <model_dir>/                              ← name appears in MODEL_SPECS
    ├── params.txt                            ← read by cli_rope to recover arch, precision, siglip flag, etc.
    └── checkpoints/
        ├── epoch_1/
        │   ├── model/__*.distcp              ← FSDP2 sharded weights
        │   └── metadata.pt
        ├── epoch_2/
        ├── ...
        └── epoch_<N>/
```

This is the standard layout `open_clip_train` writes.

### C. Dataset layout (`$DATASET_ROOT`, `$IMAGENET_ROOT`)

For retrieval + VTAB-1k classification (`$DATASET_ROOT`):

```
$DATASET_ROOT/
├── flickr30k/
│   ├── flickr30k_test_karpathy.txt
│   └── *.jpg
├── val2014/                                  ← MSCOCO val images
├── coco_test_karpathy.json
├── caltech101/, cifar100/, dtd/, eurosat/, ...      ← VTAB tfds-prepared dirs
└── ...
```

You can reuse `/weka/oe-training-default/georges/datasets/clip_benchmark` for
this — it has the right layout for everything except `pets`, `resisc45`, and
`sun397`, which `cli_rope.py` skips automatically.

For ImageNet-1k (`$IMAGENET_ROOT`):

```
$IMAGENET_ROOT/
├── ILSVRC2012_devkit_t12.tar.gz              ← class metadata
└── val/
    └── n01440764/, n01443537/, ...           ← wnid dirs of *.JPEG files
```

`/weka/prior-default/georges/datasets/imagenet1k/imagenet` works as-is.

### D. Output layout (`$RESULTS_ROOT`, default `<CLIP_benchmark>/results`)

Each eval writes one JSON per (model, epoch, dataset, task):

```
$RESULTS_ROOT/
└── <model_dir>/
    └── epoch_<N>/
        └── {dataset}_{model_arch}_{lang}_{task}.json
```

`aggregate_results.py` walks this tree and folds it into one CSV.

---

## 2. One-time conda env setup

The eval reuses your training conda env with five extra pinned deps:

```bash
source ~/.bashrc && conda activate trainer  # or whatever env has torch + open_clip

pip install --no-deps task_adaptation==0.1 pycocotools pycocoevalcap
pip install tensorflow==2.15.1 tensorflow-datasets
pip install --no-deps mock tensorflow-addons==0.23.0 tensorflow-hub
pip install --no-deps keras==2.15.0 tf-keras==2.15.0
pip install protobuf==4.25.9
```

Why each pin:

| pin | reason |
|---|---|
| `tensorflow==2.15.1` | matches what `tensorflow_addons==0.23.0` was built against; bumping TF triggers `keras.src.engine` ModuleNotFoundError inside `tfa.utils.types` |
| `keras==2.15.0`, `tf-keras==2.15.0` | `tensorflow_addons.utils.types` imports `keras.src.engine.keras_tensor`, which only exists in keras 2.x |
| `protobuf==4.25.9` | `tfds.read_only_builder` accesses `FieldDescriptor.label`, removed in protobuf 5+ |

`pip check` will complain about beaker-py / google-api-core / wandb wanting
newer protobuf — they keep working anyway. If you don't want to pollute your
training env, clone it first
(`conda create --clone <train_env> -p ~/.conda/envs/clipbench`) and install
into the clone.

### Smoke test (one checkpoint, one dataset)

```bash
cd CLIP_benchmark
CUDA_VISIBLE_DEVICES=0 python -m clip_benchmark.cli_rope eval \
    --model_type open_clip --model ViT-B-16-SigLIP \
    --pretrained $CHECKPOINT_ROOT/<model_dir>/checkpoints/epoch_<N> \
    --task zeroshot_retrieval \
    --dataset_root $DATASET_ROOT \
    --dataset flickr30k \
    --output /tmp/smoke.json \
    --batch_size 64 --language en --recall_k 1 5 \
    --is-fsdp
```

Look for a sane recall@1 in `/tmp/smoke.json`.

---

## 3. Single-GPU sweep over many checkpoints

Each sweep script reads the same env vars and reads its own `MODEL_SPECS`
array. Edit `MODEL_SPECS` at the top of each script you want to use:

```bash
declare -a MODEL_SPECS=(
    # "tag|model_dir|model_arch|epoch_list"
    "400mv2|400Mv2_lr0p001-...-rope_dp|ViT-B-16-SigLIP|49"
    "coca-do0|coca_mixed_contrastive_dropout0_...|coca_ViT-B-32-siglip|1 2 3 4"
)
```

- `tag`: a short label used for log lines (whatever you want)
- `model_dir`: name of the checkpoint dir under `$CHECKPOINT_ROOT`
- `model_arch`: open_clip architecture name (must match `params.txt`'s `model:` field)
- `epoch_list`: space-separated epochs

Then run:

```bash
export CHECKPOINT_ROOT=/path/to/your/checkpoints
export DATASET_ROOT=/weka/oe-training-default/georges/datasets/clip_benchmark
export IMAGENET_ROOT=/weka/prior-default/georges/datasets/imagenet1k/imagenet
# optional:
export RESULTS_ROOT=/path/to/output    # default: <repo>/CLIP_benchmark/results
export CUDA_VISIBLE_DEVICES=0           # default: 0

bash CLIP_benchmark/eval_retrieval_over_epochs.sh
bash CLIP_benchmark/eval_vtab_over_epochs.sh
bash CLIP_benchmark/eval_imagenet_over_epochs.sh
# or all at once + aggregate:
bash CLIP_benchmark/run_all_evals.sh
```

All scripts pass `--skip_existing`, so re-running is cheap and resumable.

ETA on a single A100-80GB:

| eval | per-checkpoint | 6 checkpoints |
|---|---|---|
| flickr30k retrieval | ~30 s | ~3 min |
| mscoco_captions retrieval | ~1.5 min | ~10 min |
| ImageNet-1k val | ~2 min | ~12 min |
| VTAB-1k (16 datasets) | ~10–15 min | ~1–1.5 h |

---

## 4. Beaker fan-out — one job per (checkpoint, task)

For sweeps where you'd rather wait 1 hour wall-clock than serial-run for 10
hours, use the `submit_jobs/` scripts. They fire one `gantry run` per
checkpoint.

```bash
# launch from anywhere (script picks the empty/clean repo state up via --allow-dirty)
export CHECKPOINT_ROOT=/path/to/your/checkpoints
export DATASET_ROOT=/weka/oe-training-default/georges/datasets/clip_benchmark
export IMAGENET_ROOT=/weka/prior-default/georges/datasets/imagenet1k/imagenet

bash CLIP_benchmark/submit_jobs/eval_retrieval_per_ckpt.sh
bash CLIP_benchmark/submit_jobs/eval_vtab_per_ckpt.sh
bash CLIP_benchmark/submit_jobs/eval_imagenet_per_ckpt.sh
```

Each script accepts these optional gantry overrides via env var (defaults in
parentheses):

| env var | default |
|---|---|
| `WORKSPACE` | `ai2/oe-encoder` |
| `BUDGET` | `ai2/oe-mm` |
| `PRIORITY` | `urgent` |
| `BEAKER_IMAGE` | `sanghol/molmo2-torch2.7.1-cuda12.8` |
| `CONDA_ENV` | `trainer` |
| `BEAKER_HOME` | `$HOME` (your local home; the container's `~/.bashrc` and `~/.conda/envs/$CONDA_ENV` must live here) |
| `OPEN_CLIP_REPO` | derived from script location (the cloned repo root that contains `src/` and `CLIP_benchmark/`) |

Important: the Beaker container needs to find your conda env. The submit
script sets `--env HOME=$BEAKER_HOME` so the container's `~/.bashrc` and
`~/.conda/envs/$CONDA_ENV` resolve to your weka home. If you put your repo
under a different weka path than your home, override `OPEN_CLIP_REPO`.

---

### Auto-scanning every model in `$CHECKPOINT_ROOT`

If you keep all your checkpoints under one root and just want "eval every model
at its latest epoch, skip whatever is already done", use the auto-scanner:

```bash
CHECKPOINT_ROOT=/path/to/your/checkpoints \
DATASET_ROOT=/weka/oe-training-default/georges/datasets/clip_benchmark \
IMAGENET_ROOT=/weka/prior-default/georges/datasets/imagenet1k/imagenet \
bash CLIP_benchmark/submit_jobs/auto_submit_retrieval_imagenet.sh
```

For each `<model_dir>` under `$CHECKPOINT_ROOT` it:

1. Picks the highest-numbered `checkpoints/epoch_N/`
2. Reads the architecture from `<model_dir>/params.txt` (`model:` line)
3. Submits 1 Beaker job for retrieval (mscoco_captions + flickr30k) and 1 for
   imagenet1k — **only if** the expected output JSON(s) are missing under
   `$RESULTS_ROOT/<model_dir>/epoch_N/`

Flags:
- `--status` → dry-run, prints what would be submitted, fires nothing
- `--force`  → ignore existing JSONs; submit every job

Re-fire any time; it's idempotent.

## 5. Aggregating results

After the JSON files land under `$RESULTS_ROOT`, fold them into a single CSV:

```bash
python CLIP_benchmark/aggregate_results.py "$RESULTS_ROOT" \
    --output "$RESULTS_ROOT/summary.csv"
```

The CSV has one row per `(model_dir, epoch, dataset, task)` and one column
per metric. `run_all_evals.sh` does this automatically as its last step.

---

## 6. Datasets currently supported

### Retrieval (`--task zeroshot_retrieval`)

| dataset | metrics | language(s) |
|---|---|---|
| `flickr30k` | `image_retrieval_recall@k`, `text_retrieval_recall@k` | `en`, `zh` |
| `mscoco_captions` | same | `en` |

### Classification (`--task zeroshot_classification`)

| dataset | metric key | notes |
|---|---|---|
| `cifar100`, `cifar10` | `acc1` | torchvision direct |
| `vtab` (collection) | `acc1` | expands to 16 datasets per ckpt; pets/resisc45/sun397 skipped (data not on disk) |
| `imagenet1k` | `acc1` | uses torchvision `ImageNet(root, split="val")` |

To add a new VTAB dataset that's missing locally: copy its tfds-prepared
data into `$DATASET_ROOT` and remove the dataset name from the skip list at
`CLIP_benchmark/clip_benchmark/cli_rope.py:189`.

---

## 7. Known gotchas

- **CoCa vs CLIP `encode_image` return**: after `drop_in_replacements`, CLIP
  models return `{"x": ...}` and CoCa models return `{"image_latent": ...}`.
  Both `metrics/zeroshot_classification.py` and `metrics/zeroshot_retrieval.py`
  fall back across these keys, so as long as you don't change those files
  you don't need to think about it.
- **Stale `open_clip` import**: `clip_benchmark/models/open_clip.py` does
  `sys.path.insert(0, OPEN_CLIP_SRC)` and then drops any pre-imported
  `open_clip` from `sys.modules` so the local source (which has
  `drop_in_replacements`) wins over a pip-installed `open_clip_torch`. If you
  see `ImportError: cannot import name 'drop_in_replacements'`, that block
  has been removed or `OPEN_CLIP_SRC` is wrong.
- **VTAB skip list**: see §6. The "missing" 3 datasets (`pets`/`oxford_iiit_pet`,
  `resisc45`, `sun397`) are an artifact of the public dataset_root, not a
  cli_rope bug.
- **Single-GPU FSDP load**: `cli_rope.py` loads FSDP2 distcp checkpoints into
  a non-sharded model on 1 GPU via
  `torch.distributed.checkpoint.state_dict_loader.load(...)`. This works for
  any world_size the checkpoint was saved at; you don't need to match training
  GPU count.

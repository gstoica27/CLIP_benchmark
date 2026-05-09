import os
import pdb
import torch
import sys

# Force the local user open_clip checkout (which has drop_in_replacements / RoPE / FSDP-aware
# extras) ahead of any pip-installed open_clip_torch in site-packages. Must run BEFORE the
# `from open_clip import drop_in_replacements` line below.
# Force the local open_clip checkout (which has drop_in_replacements / RoPE / FSDP-aware
# extras) ahead of any pip-installed open_clip_torch in site-packages. The path is
# computed relative to this file:
#   <repo>/CLIP_benchmark/clip_benchmark/models/open_clip.py  →  <repo>/src
# so it works for any clone location. Override with $OPEN_CLIP_SRC if you keep the
# open_clip source somewhere else.
_DEFAULT_SRC = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src")
)
_USER_OPEN_CLIP_SRC = os.environ.get("OPEN_CLIP_SRC", _DEFAULT_SRC)
while _USER_OPEN_CLIP_SRC in sys.path:
    sys.path.remove(_USER_OPEN_CLIP_SRC)
sys.path.insert(0, _USER_OPEN_CLIP_SRC)
# If a stale `open_clip` was already imported (e.g. from site-packages), drop it so the next
# import resolves against the user's source.
for _mod in list(sys.modules):
    if _mod == "open_clip" or _mod.startswith("open_clip."):
        del sys.modules[_mod]
import open_clip
import torch.distributed.checkpoint as dist_cp
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from open_clip import drop_in_replacements
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner


def load_open_clip(
        model_name: str = "ViT-B-32-quickgelu", 
        pretrained: str = "laion400m_e32", 
        cache_dir: str = None, 
        device="cpu", 
        is_fsdp: bool = False,
        use_rope: bool = False,
        del_pos_emb: bool = False,
        image_mean: list = None,
        image_std: list = None,
        image_interpolation: str = None,
    ):
    pdb.set_trace()

    if is_fsdp:
        model, _, transform = open_clip.load_model_and_preprocess(
            model_name, pretrained=pretrained, cache_dir=cache_dir,
            image_mean=image_mean, image_std=image_std,
            image_interpolation=image_interpolation,
            output_dict=True
        )
        sd_options = dist_cp_sd.StateDictOptions(full_state_dict=False, cpu_offload=False)
        model_path = os.path.join(pretrained, "model")
        model_state = {'model': dist_cp_sd.get_model_state_dict(model, options=sd_options)}
        dist_cp.state_dict_loader.load(
            model_state,
            storage_reader=dist_cp.FileSystemReader(model_path),
            planner=DefaultLoadPlanner(),
        )
        pdb.set_trace()

        sd = dist_cp_sd.get_model_state_dict(model_path, options=sd_options)
        model.load_state_dict(sd)

    elif pretrained.endswith('.pth') or pretrained.endswith('.pt'):
        model, _, transform = open_clip.load_model_and_preprocess(
            model_name, pretrained=pretrained, cache_dir=cache_dir,
            image_mean=image_mean, image_std=image_std,
            image_interpolation=image_interpolation,
            output_dict=True
        )
        sd = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(sd)
    else:
        model, _, transform = open_clip.load_model_and_preprocess(
            model_name, pretrained=pretrained, cache_dir=cache_dir,
            image_mean=image_mean, image_std=image_std,
            image_interpolation=image_interpolation,
            output_dict=True
        )
    # model, transform = open_clip.create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP')
    drop_in_replacements.replace_forward_functions(
        model, instructions={
            'apply_rope': use_rope,
            'del_pos_emb': del_pos_emb
        }
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer

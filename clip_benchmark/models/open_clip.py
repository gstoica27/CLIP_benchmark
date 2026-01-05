import os
import pdb
import torch
import sys
sys.path.append(
    '/weka/prior-default/georges/research/open_clip/src'
)
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

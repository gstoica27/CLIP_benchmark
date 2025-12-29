import os
import pdb
import torch
import open_clip
import torch.distributed.checkpoint as dist_cp
import torch.distributed.checkpoint.state_dict as dist_cp_sd


def load_open_clip(model_name: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu", is_fsdp: bool = False):
    pdb.set_trace()
    if is_fsdp:
        model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained='', cache_dir=cache_dir)
        sd_options = dist_cp_sd.StateDictOptions(full_state_dict=False, cpu_offload=False)
        model_path = os.path.join(pretrained, "model")
        sd = dist_cp_sd.get_model_state_dict(model_path, options=sd_options)
        model.load_state_dict(sd)
    else:
        model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, cache_dir=cache_dir)
    # model, transform = open_clip.create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP')
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, transform, tokenizer

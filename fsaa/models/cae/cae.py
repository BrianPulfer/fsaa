import os
from functools import partial

import gdown
import torch
import torch.nn as nn

from fsaa.attack import TransformAndModelWrapper
from fsaa.models.cae.modeling_finetune import VisionTransformer
from fsaa.transforms.normalize import IMAGENET_MEAN, IMAGENET_STD, Normalize

NAME_TO_URL = {
    "cae_large_1600ep": "https://drive.google.com/file/d/1jSsB-eKCIWEla7p0c2osoifwV0Cq0U8i/view?usp=sharing",
    "cae_base_1600ep": "https://drive.google.com/file/d/1CJAjBN0F7-Eijmv2ZmtwpZ6PS_SRcb7I/view?usp=sharing",
}

SUPPORTED_CAE_MODELS = ["cae_base_1600ep", "cae_large_1600ep"]


def name_to_model(name: str):
    if "base" in name:
        model = VisionTransformer(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=0.1,
        )
    if "large" in name:
        model = VisionTransformer(
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=0.1,
        )
    return model


class CAEModel(nn.Module):
    def __init__(self, model_name, cache_dir="./.cae_cache", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Checking if the model is supported
        if model_name not in SUPPORTED_CAE_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not supported. \
                Pick one of {SUPPORTED_CAE_MODELS}"
            )

        # Loading the state dict
        os.makedirs(cache_dir, exist_ok=True)
        ckpt_path = os.path.join(cache_dir, f"{model_name}.pth")

        if not os.path.isfile(ckpt_path):
            print(
                f"Checkpoint for {model_name} not found in \
                {os.path.abspath(cache_dir)}. Downloading..."
            )
            url = NAME_TO_URL[model_name]
            gdown.download(url=url, output=ckpt_path, quiet=False, fuzzy=True)

        state_dict = torch.load(ckpt_path)["model"]
        state_dict = {
            k.replace("encoder.", "").replace("norm.", "fc_norm."): v
            for k, v in state_dict.items()
            if k != "mask_token"
        }

        # Initializing CAE model
        cae = name_to_model(model_name)
        cae.head = nn.Identity()

        cae.load_state_dict(state_dict, strict=True)

        # Wrapping the model with normalization
        self.model = TransformAndModelWrapper(
            cae, transform=Normalize(IMAGENET_MEAN, IMAGENET_STD)
        )

    def forward(self, x):
        return self.model(x)

import os

import torch
import torch.nn as nn
import wget

from fsaa.attack import TransformAndModelWrapper
from fsaa.models.ijepa.vision_transformer import vit_giant, vit_huge
from fsaa.transforms.normalize import IMAGENET_MEAN, IMAGENET_STD, Normalize

SUPPORTED_IJEPA_MODELS = [
    "ijepa_vith_14_pt22k",
    "ijepa_vitg_16_pt22k",
]

NAME_TO_URL = {
    "ijepa_vith_14_pt22k": "https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar",
    "ijepa_vitg_16_pt22k": "https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar",
}

NAMES_TO_MODELS = {
    "ijepa_vith_14_pt22k": lambda: vit_huge(patch_size=14),
    "ijepa_vitg_16_pt22k": lambda: vit_giant()
}


class IJEPA(nn.Module):
    def __init__(self, model_name, cache_dir="./.ijepa_cache", *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Checking if the model is supported
        if model_name not in SUPPORTED_IJEPA_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not supported. \
                Pick one of {SUPPORTED_IJEPA_MODELS}"
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
            wget.download(url, ckpt_path)

        state_dict = torch.load(ckpt_path, map_location="cpu")['encoder']
        state_dict = {k.replace("module.", ""): v for k,
                      v in state_dict.items()}

        # Initializing iBot model
        ijepa = NAMES_TO_MODELS[model_name]()
        ijepa.load_state_dict(state_dict, strict=True)

        # Wrapping the model with normalization
        self.model = TransformAndModelWrapper(
            ijepa, transform=Normalize(IMAGENET_MEAN, IMAGENET_STD)
        )

    def forward(self, x):
        return self.model(x)

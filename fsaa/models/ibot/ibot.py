"""Models from https://github.com/bytedance/ibot"""
import os

import torch
import torch.nn as nn
import wget

from fsaa.models.core import TransformAndModelWrapper
from fsaa.models.ibot.vits import vit_base, vit_large
from fsaa.transforms.normalize import IMAGENET_MEAN, IMAGENET_STD, Normalize

SUPPORTED_IBOT_MODELS = [
    "ibot_vitb_16_pt22k",
    "ibot_vitl_16_pt22k"
]

NAMES_TO_MODELS = {
    "ibot_vitb_16_pt22k": lambda: vit_base(masked_im_modeling=True),
    "ibot_vitl_16_pt22k": lambda: vit_large(masked_im_modeling=True)
}


def model_name_to_url(model_name: str):
    name = model_name.split("ibot_")[1]
    prefix = "https://lf3-nlp-opensource.bytetos.com"
    prefix += "/obj/nlp-opensource/archive/2022/ibot"
    postfix = ("checkpoint_teacher.pth"
               if "pt22k" not in model_name
               else "checkpoint_student.pth")
    return f"{prefix}/{name}/{postfix}"


class iBOTModel(nn.Module):
    f"""
    Official iBOT model from https://arxiv.org/abs/2111.07832.
    The model is wrapped with its pre-processing normalization.

    Args:
        model_name (str): Name of the model to be used. Pick one of {SUPPORTED_IBOT_MODELS}.
        cache_dir (str): Path to the directory to cache the model checkpoints.
    """

    def __init__(
        self,
        model_name,
        cache_dir="~/.cache/fsaa/models/ibot",
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        # Checking if the model is supported
        if model_name not in SUPPORTED_IBOT_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not supported. \
                Pick one of {SUPPORTED_IBOT_MODELS}"
            )

        # Dealing with home directory
        if cache_dir.startswith("~/"):
            cache_dir = os.path.join(os.path.expanduser("~"), cache_dir[2:])

        # Loading the state dict
        os.makedirs(cache_dir, exist_ok=True)
        ckpt_path = os.path.join(cache_dir, f"{model_name}.pth")

        if not os.path.isfile(ckpt_path):
            print(f"Checkpoint for {model_name} not found in \
                {os.path.abspath(cache_dir)}. Downloading...")
            url = model_name_to_url(model_name)
            wget.download(url, ckpt_path)

        state_dict = torch.load(ckpt_path)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']

        # Dropping head parameters from the state dict
        state_dict = {k: v
                      for k, v in state_dict.items()
                      if not k.startswith("head")}

        # Initializing iBot model
        ibot = NAMES_TO_MODELS[model_name]()
        ibot.load_state_dict(state_dict, strict=True)

        # Wrapping the model with normalization
        self.model = TransformAndModelWrapper(
            ibot,
            transform=Normalize(IMAGENET_MEAN, IMAGENET_STD)
        )

    def forward(self, x):
        return self.model(x)

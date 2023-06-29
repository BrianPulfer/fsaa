import torch
from torch.nn import Module

from fsaa.attack import TransformAndModelWrapper
from fsaa.transforms.normalize import IMAGENET_MEAN, IMAGENET_STD, Normalize

SUPPORTED_DINOV2_MODELS = [
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
]


class DinoV2Model(Module):
    """Base class for all feature extractors."""

    def __init__(self, model_name):
        super(DinoV2Model, self).__init__()
        if model_name not in SUPPORTED_DINOV2_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not supported. \
                Pick one of {SUPPORTED_DINOV2_MODELS}"
            )

        dino = torch.hub.load("facebookresearch/dinov2",
                              model_name, verbose=False)

        self.model = TransformAndModelWrapper(
            dino,
            transform=Normalize(IMAGENET_MEAN, IMAGENET_STD)
        )

    def forward(self, x):
        return self.model(x)

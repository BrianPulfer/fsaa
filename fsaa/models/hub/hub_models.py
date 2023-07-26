import torch
from torch.nn import Module

from fsaa.attack import TransformAndModelWrapper
from fsaa.transforms.normalize import IMAGENET_MEAN, IMAGENET_STD, Normalize

SUPPORTED_BARLOWTWINS_MODELS = [
    "barlowtwins_resnet50",
]

SUPPORTED_DINOV2_MODELS = [
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
]

SUPPORTED_SWAV_MODELS = [
    "swav_resnet50",
    "swav_resnet50w2",
    "swav_resnet50w4",
    "swav_resnet50w5",
]

SUPPORTED_VICREG_MODELS = [
    "vicreg_resnet50",
    "vicreg_resnet50x2",
    "vicreg_resnet200x2"
]

SUPPORTED_HUB_MODELS = (
    SUPPORTED_BARLOWTWINS_MODELS +
    SUPPORTED_DINOV2_MODELS +
    SUPPORTED_SWAV_MODELS +
    SUPPORTED_VICREG_MODELS
)


class HubModel(Module):
    """Base class for all feature extractors."""

    def __init__(self, model_name):
        super(HubModel, self).__init__()
        if model_name not in SUPPORTED_HUB_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not supported. \
                Pick one of {SUPPORTED_HUB_MODELS}"
            )

        if model_name in SUPPORTED_BARLOWTWINS_MODELS:
            hub_model = torch.hub.load("facebookresearch/barlowtwins:main",
                                       model_name.split("_")[1],
                                       verbose=False)

        if model_name in SUPPORTED_DINOV2_MODELS:
            hub_model = torch.hub.load("facebookresearch/dinov2",
                                       model_name,
                                       verbose=False)

        if model_name in SUPPORTED_SWAV_MODELS:
            hub_model = torch.hub.load("facebookresearch/swav:main",
                                       model_name.split("_")[1],
                                       verbose=False)

        if model_name in SUPPORTED_VICREG_MODELS:
            hub_model = torch.hub.load('facebookresearch/vicreg:main',
                                       model_name.split("_")[1],
                                       verbose=False)

        # Removing classification head if any
        if hasattr(hub_model, "fc"):
            hub_model.fc = torch.nn.Identity()

        self.model = TransformAndModelWrapper(
            hub_model,
            transform=Normalize(IMAGENET_MEAN, IMAGENET_STD)
        )

    def forward(self, x):
        return self.model(x)

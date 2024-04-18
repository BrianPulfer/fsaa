import torch
from torch.nn import Module

SUPPORTED_BARLOWTWINS_MODELS = [
    "barlowtwins_resnet50",
]

SUPPORTED_SWAV_MODELS = [
    "swav_resnet50",
    "swav_resnet50w2",
    "swav_resnet50w4",
    "swav_resnet50w5",
]

SUPPORTED_VICREG_MODELS = ["vicreg_resnet50",
                           "vicreg_resnet50x2", "vicreg_resnet200x2"]

SUPPORTED_HUB_MODELS = (
    SUPPORTED_BARLOWTWINS_MODELS + SUPPORTED_SWAV_MODELS + SUPPORTED_VICREG_MODELS
)


def load_hub_model(model_name: str):
    if model_name not in SUPPORTED_HUB_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not supported. \
                Pick one of {SUPPORTED_HUB_MODELS}"
        )

    if model_name in SUPPORTED_BARLOWTWINS_MODELS:
        hub_model = torch.hub.load(
            "facebookresearch/barlowtwins:main", model_name.split("_")[1], verbose=False
        )

    if model_name in SUPPORTED_SWAV_MODELS:
        hub_model = torch.hub.load(
            "facebookresearch/swav:main", model_name.split("_")[1], verbose=False
        )

    if model_name in SUPPORTED_VICREG_MODELS:
        hub_model = torch.hub.load(
            "facebookresearch/vicreg:main", model_name.split("_")[1], verbose=False
        )

    # Removing classification head if any
    if hasattr(hub_model, "fc"):
        hub_model.fc = torch.nn.Identity()


class HubModel(Module):
    """
    Base class for all TorchHub models.
    Models are wrapped with corresponding pre-processing normalization.

    Args:
        model_name (str): Name of the model to be used.
    """

    def __init__(self, model_name, *args, **kwargs):
        super(HubModel, self).__init__()
        self.model_name = model_name

        if model_name not in SUPPORTED_HUB_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not supported. \
                Pick one of {SUPPORTED_HUB_MODELS}"
            )

        if model_name in SUPPORTED_BARLOWTWINS_MODELS:
            hub_model = torch.hub.load(
                "facebookresearch/barlowtwins:main",
                model_name.split("_")[1],
                verbose=False,
            )

        if model_name in SUPPORTED_SWAV_MODELS:
            hub_model = torch.hub.load(
                "facebookresearch/swav:main", model_name.split("_")[1], verbose=False
            )

        if model_name in SUPPORTED_VICREG_MODELS:
            hub_model = torch.hub.load(
                "facebookresearch/vicreg:main", model_name.split("_")[1], verbose=False
            )

        # Removing classification head if any
        if hasattr(hub_model, "fc"):
            hub_model.fc = torch.nn.Identity()

        self.model = hub_model

    def forward(self, x):
        return self.model(x)

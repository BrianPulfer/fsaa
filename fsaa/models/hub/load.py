import torch

from .hub_models import (SUPPORTED_BARLOWTWINS_MODELS, SUPPORTED_HUB_MODELS,
                         SUPPORTED_SWAV_MODELS, SUPPORTED_VICREG_MODELS)


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

    return hub_model

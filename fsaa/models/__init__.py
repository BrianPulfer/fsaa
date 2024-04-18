from torch.nn import Module
from torchvision.transforms import Normalize
from transformers import AutoImageProcessor

from .cae.load import SUPPORTED_CAE_MODELS, load_cae_model
from .hf.hf_models import (SUPPORTED_BEIT_MODELS, SUPPORTED_HF_MODELS,
                           load_hf_model)
from .hub.load import SUPPORTED_HUB_MODELS, load_hub_model
from .ibot.load import SUPPORTED_IBOT_MODELS, load_ibot_model
from .ijepa.load import SUPPORTED_IJEPA_MODELS, load_ijepa_model

IMAGENET_INCEPTION_MEAN = [0.5, 0.5, 0.5]
IMAGENET_INCEPTION_STD = [0.5, 0.5, 0.5]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

OPENAI_NORMALIZATION_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_NORMALIZATION_STD = [0.26862954, 0.26130258, 0.27577711]

SUPPORTED_MODELS = (
    SUPPORTED_CAE_MODELS
    + SUPPORTED_HUB_MODELS
    + SUPPORTED_HF_MODELS
    + SUPPORTED_IBOT_MODELS
    + SUPPORTED_IJEPA_MODELS
)

SUPPORTED_MODELS_ACTIVATIONS = (
    SUPPORTED_CAE_MODELS
    + SUPPORTED_HF_MODELS
    + SUPPORTED_IBOT_MODELS
    + SUPPORTED_IJEPA_MODELS
)


def get_model(model_name: str, *args, **kwargs) -> Module:
    """
    Returns the pre-trained model with the given name.
    The model is wrapped with its corresponding pre-processing differentiable transform.

    Args:
        model_name (str): Name of the model.

    Returns:
        Module: Pre-trained model with the given name.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not supported. \
            Pick one of {SUPPORTED_MODELS}"
        )

    if model_name in SUPPORTED_CAE_MODELS:
        return load_cae_model(model_name, *args, **kwargs)

    if model_name in SUPPORTED_HUB_MODELS:
        return load_hub_model(model_name, *args, **kwargs)

    if model_name in SUPPORTED_HF_MODELS:
        return load_hf_model(model_name, *args, **kwargs)

    if model_name in SUPPORTED_IBOT_MODELS:
        return load_ibot_model(model_name, *args, **kwargs)

    if model_name in SUPPORTED_IJEPA_MODELS:
        return load_ijepa_model(model_name, *args, **kwargs)


def get_default_transform(model_name):
    if model_name in SUPPORTED_BEIT_MODELS:
        return Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)

    if model_name in SUPPORTED_HF_MODELS:
        processor = AutoImageProcessor.from_pretrained(model_name)
        return lambda tensor: processor(tensor, return_tensors="pt", do_rescale=False)[
            "pixel_values"
        ]

    return Normalize(IMAGENET_MEAN, IMAGENET_STD)

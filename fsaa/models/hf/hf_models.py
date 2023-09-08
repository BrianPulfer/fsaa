import torch
from torch.nn import Module
from transformers import AutoModel, logging

from fsaa.models.core import TransformAndModelWrapper
from fsaa.transforms.normalize import (IMAGENET_INCEPTION_MEAN,
                                       IMAGENET_INCEPTION_STD, IMAGENET_MEAN,
                                       IMAGENET_STD, Normalize)

SUPPORTED_BEIT_MODELS = [
    "microsoft/beit-base-patch16-224-pt22k",
    "microsoft/beit-large-patch16-224-pt22k"
]

SUPPORTED_DINO_MODELS = [
    "facebook/dino-vits16",
    "facebook/dino-vitb16",
]

SUPPORTED_MAE_MODELS = [
    "facebook/vit-mae-base",
    "facebook/vit-mae-large",
    "facebook/vit-mae-huge",
]

SUPPORTED_MSN_MODELS = [
    "facebook/vit-msn-small",
    "facebook/vit-msn-base",
    "facebook/vit-msn-large"
]

SUPPORTED_HF_MODELS = (
    SUPPORTED_BEIT_MODELS
    + SUPPORTED_DINO_MODELS
    + SUPPORTED_MAE_MODELS
    + SUPPORTED_MSN_MODELS
)


def name_to_model(model_name: str):
    """Returns the model from the given name."""
    logging.set_verbosity_error()
    model = AutoModel.from_pretrained(model_name)

    if model_name in SUPPORTED_MAE_MODELS:
        model.embeddings.config.mask_ratio = 0

    logging.set_verbosity_warning()
    return model


class HFModel(Module):
    """
    Base class for all models from HuggingFace.
    Models are wrapped with corresponding pre-processing normalization.

    Args:
        model_name (str): Name of the model to be used.
    """

    def __init__(self, model_name, *args, **kwargs):
        super(HFModel, self).__init__()
        if model_name not in SUPPORTED_HF_MODELS:
            raise ValueError(
                f"Model '{model_name}' is not supported. \
                Pick one of {SUPPORTED_BEIT_MODELS}"
            )

        self.model_name = model_name
        hf_model = name_to_model(model_name)

        if model_name in SUPPORTED_BEIT_MODELS:
            mean, std = IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        else:
            mean, std = IMAGENET_MEAN, IMAGENET_STD

        self.model = TransformAndModelWrapper(
            hf_model, transform=Normalize(mean, std)
        )

    def forward(self, x):
        """Runs the given batch through the model to extract features."""
        out = self.model(x)
        hidden_state = out["last_hidden_state"]

        # Sorting tokens in hidden state for MAE models
        if self.model_name in SUPPORTED_MAE_MODELS:
            d = hidden_state.shape[-1]
            ids = out['ids_restore']
            ids = ids.unsqueeze(-1).expand(-1, -1, d)

            hidden_state = torch.cat(
                [
                    hidden_state[:, 0].unsqueeze(1),
                    torch.gather(
                        hidden_state[:, 1:],
                        dim=1,
                        index=ids
                    )
                ],
                dim=1
            )

        return hidden_state

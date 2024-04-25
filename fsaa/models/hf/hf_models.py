import torch
from torch.nn import Module
from transformers import AutoModel, logging

SUPPORTED_BEIT_MODELS = [
    "microsoft/beit-base-patch16-224-pt22k",
    "microsoft/beit-large-patch16-224-pt22k",
]

SUPPORTED_DINO_MODELS = [
    "facebook/dino-vits16",
    "facebook/dino-vitb16",
]

SUPPORTED_DINOV2_MODELS = [
    "facebook/dinov2-small",
    "facebook/dinov2-base",
    "facebook/dinov2-large",
    "facebook/dinov2-giant",
]

SUPPORTED_MAE_MODELS = [
    "facebook/vit-mae-base",
    "facebook/vit-mae-large",
    "facebook/vit-mae-huge",
]

SUPPORTED_MSN_MODELS = [
    "facebook/vit-msn-small",
    "facebook/vit-msn-base",
    "facebook/vit-msn-large",
]

SUPPORTED_HF_MODELS = (
    SUPPORTED_BEIT_MODELS
    + SUPPORTED_DINO_MODELS
    + SUPPORTED_DINOV2_MODELS
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


def load_hf_model(model_name, *args, **kwargs):
    model = name_to_model(model_name)
    return HFModelWrapper(model, model_name=model_name, *args, **kwargs)


class HFModelWrapper(Module):
    """
    Base class for all models from HuggingFace. Hidden states are extracted by default.

    Args:
        model_name (str): Name of the model to be used.
    """

    def __init__(
        self, hf_model, model_name=None, output_key="last_hidden_state", *args, **kwargs
    ):
        super(HFModelWrapper, self).__init__()
        self.hf_model = hf_model
        self.model_name = model_name
        self.output_key = output_key

    def forward(self, x):
        """Runs the given batch through the model to extract features."""
        out = self.hf_model(x)
        output = out[self.output_key]

        # Sorting tokens in hidden state for MAE models
        if (
            self.model_name in SUPPORTED_MAE_MODELS
            and self.output_key == "last_hidden_state"
        ):
            d = output.shape[-1]
            ids = out["ids_restore"]
            ids = ids.unsqueeze(-1).expand(-1, -1, d)

            output = torch.cat(
                [
                    output[:, 0].unsqueeze(1),
                    torch.gather(output[:, 1:], dim=1, index=ids),
                ],
                dim=1,
            )

        return output

    def forward_activations(self, x, layer_idxs=None):
        """Runs the given batch through the model to extract features."""
        if layer_idxs is None:
            layer_idxs = list(range(self.hf_model.config.num_hidden_layers))

        out = self.hf_model(x, output_hidden_states=True)
        output = out[self.output_key]
        acts = torch.stack(out["hidden_states"], dim=1)

        # Sorting tokens in hidden state for MAE models
        if (
            self.model_name in SUPPORTED_MAE_MODELS
            and self.output_key == "last_hidden_state"
        ):
            d = output.shape[-1]
            ids = out["ids_restore"]
            ids = ids.unsqueeze(-1).expand(-1, -1, d)

            output = torch.cat(
                [
                    output[:, 0].unsqueeze(1),
                    torch.gather(output[:, 1:], dim=1, index=ids),
                ],
                dim=1,
            )

        return output, acts

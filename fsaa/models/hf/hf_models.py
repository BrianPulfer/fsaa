from typing import List, Tuple

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

    def _sort_mae_output(self, output, ids_restore):
        ids = ids_restore.unsqueeze(-1).expand(-1, -1, output.shape[-1])
        return torch.cat(
            [
                output[:, 0].unsqueeze(1),
                torch.gather(output[:, 1:], dim=1, index=ids),
            ],
            dim=1,
        )

    def forward(self, x):
        """Runs the given batch through the model to output"""
        out = self.hf_model(x)
        output = out[self.output_key]

        # Sorting tokens in hidden state for MAE models
        if (
            self.model_name in SUPPORTED_MAE_MODELS
            and self.output_key == "last_hidden_state"
        ):
            output = self._sort_mae_output(output, out["ids_restore"])

        return output

    # TODO: layer_idxs is not used
    def forward_activations(self, x: torch.Tensor, layer_idxs: List[int] = None):
        """Runs the given batch through the model to extract features.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, h, w)
            layer_idxs (List[int], optional): List of layers used to extract activations. If None, uses all layers. Defaults to None.

        Returns:
            _type_: Tuple[torch.Tensor, torch.Tensor]: Tuple containing the output tensor and the activations tensor. The output tensor has shape (B, T, D), where B is the batch size, T is the number of tokens and D is the hidden dimensionality. The activations tensor has shape (B, L, T, D), where L is the number of layers.
        """
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
            output = self._sort_mae_output(output, out["ids_restore"])

        return output, acts

    # TODO: layer_idxs is not used
    def forward_attn(
        self, x: torch.Tensor, layer_idxs: List[int] = None
    ) -> Tuple[torch.Tensor]:
        """Runs the given batch through the model to extract attention maps.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, h, w)
            layer_idxs (List[int], optional): List of layers used to extract attention maps. If None, uses all layers. Defaults to None.

        Returns:
            Tuple[torch.Tensor]: Model output and Attention maps. The attention maps have shape (B, L, H, P, P), where B is the batch size, L is the number of layers, H the number of attention heads and P is the number of patches.
        """

        if layer_idxs is None:
            layer_idxs = list(range(self.hf_model.config.num_hidden_layers))

        out = self.hf_model(x, output_attentions=True)
        output = out[self.output_key]
        attns = torch.stack(out["attentions"], dim=1)

        if (
            self.model_name in SUPPORTED_MAE_MODELS
            and self.output_key == "last_hidden_state"
        ):
            output = self._sort_mae_output(output, out["ids_restore"])

        return output, attns

    # TODO: layer_idxs is not used
    def forward_all(
        self, x: torch.Tensor, layer_idxs: List[int] = None
    ) -> Tuple[torch.Tensor]:
        """Runs the given batch through the model to extract output, activations, attention maps and gradients. The gradients are taken with respect to the provided loss function.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, h, w)
            transform (Callable): Transform function to be applied to the input tensor.
            layer_idxs (List[int], optional): List of layers used to extract activations and attention maps. If None, uses all layers. Defaults to None.

        Returns:
            Tuple[torch.Tensor]: Model output, activations, attention maps and gradients. The activations tensor has shape (B, L, T, D), where B is the batch size, L is the number of layers, T is the number of tokens and D is the hidden dimensionality. The attention maps have shape (B, L, H, P, P), where B is the batch size, L is the number of layers, H the number of attention heads and P is the number of patches. The gradients tensor has shape (B, C, H, W).
        """

        if layer_idxs is None:
            layer_idxs = list(range(self.hf_model.config.num_hidden_layers))

        x.requires_grad_(True)

        out = self.hf_model(x, output_attentions=True,
                            output_hidden_states=True)
        output = out[self.output_key]
        acts = torch.stack(out["hidden_states"], dim=1)
        attns = torch.stack(out["attentions"], dim=1)

        if (
            self.model_name in SUPPORTED_MAE_MODELS
            and self.output_key == "last_hidden_state"
        ):
            output = self._sort_mae_output(output, out["ids_restore"])

        return output, acts, attns

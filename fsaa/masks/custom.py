import warnings

import torch
from torch import Tensor

from fsaa.core import PerceptualMask


class CustomMask(PerceptualMask):
    """
    CustomMask allows to pass a custom tensor as a mask.

    Args:
        **kwargs: Keyword arguments to be passed to the parent class. Argument 'mask' is used to pass the custom mask.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_mask = kwargs.get("mask", None)

        if self.grad_mask is None:
            warnings.warn(
                "No mask provided for CustomMask. No masking will be used.")

    def mask(self, x: Tensor) -> Tensor:
        """
        Returns the custom mask if any, otherwise returns a tensor of ones.

        Args:
            x (Tensor): Input tensor to be masked.

        Returns:
            Tensor: Masked tensor.
        """
        if self.grad_mask is not None:
            return self.grad_mask
        return torch.ones_like(x)

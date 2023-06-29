import warnings

import torch

from fsaa.core import PerceptualMask


class CustomMask(PerceptualMask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_mask = kwargs.get("mask", None)

        if self.grad_mask is None:
            warnings.warn(
                "No mask provided for CustomMask. No masking will be used.")

    def mask(self, x):
        if self.grad_mask is not None:
            return self.grad_mask
        return torch.ones_like(x)

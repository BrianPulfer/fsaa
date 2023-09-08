import torch
from torch import Tensor

from fsaa.core import PerceptualMask


class NoMask(PerceptualMask):
    """Dummy mask that does not mask any part of the input tensor."""""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mask(self, x: Tensor) -> Tensor:
        """Returns a tensor of ones of the same shape as x.

        Args:
            x (Tensor): The input tensor

        Returns:
            Tensor: A tensor of ones of the same shape as x
        """
        return torch.ones_like(x)

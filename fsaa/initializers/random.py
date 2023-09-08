import torch
from torch import Tensor

from fsaa.core import PerturbationInitializer


class RandomInitializer(PerturbationInitializer):
    """
    RandomInitializer is a PerturbationInitializer that adds random noise to the input.
    The noise is sampled from a normal distribution.
    """

    def __init__(self, lr: float = 2 / 255, *args, **kwargs):
        super(RandomInitializer, self).__init__(lr, *args, **kwargs)

    def initialize(self, x: Tensor) -> Tensor:
        """Returns the input with random noise added to it.

        Args:
            x (Tensor): the input tensor

        Returns:
            Tensor: the input tensor with random noise added to it
        """
        return x + torch.randn_like(x) * self.lr

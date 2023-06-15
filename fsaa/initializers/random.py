import torch

from fsaa.core import PerturbationInitializer


class RandomInitializer(PerturbationInitializer):
    def __init__(self, alpha: float = 2 / 255, *args, **kwargs):
        super(RandomInitializer, self).__init__(alpha, *args, **kwargs)

    def initialize(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.rand_like(x) * self.alpha

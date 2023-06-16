import torch

from fsaa.core import PerturbationInitializer


class RandomInitializer(PerturbationInitializer):
    def __init__(self, lr: float = 2 / 255, *args, **kwargs):
        super(RandomInitializer, self).__init__(lr, *args, **kwargs)

    def initialize(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.lr

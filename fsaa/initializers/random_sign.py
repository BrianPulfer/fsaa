import torch

from fsaa.core import PerturbationInitializer


class RandomSignInitializer(PerturbationInitializer):
    def __init__(self, lr: float = 2 / 255, *args, **kwargs):
        super(RandomSignInitializer, self).__init__(lr, *args, **kwargs)

    def initialize(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x).sign() * self.lr

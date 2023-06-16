import torch

from fsaa.core import PerturbationUpdater


class RandomUpdater(PerturbationUpdater):
    """Random perturbation update.
    It servers as a baseline for other attacks."""

    def __init__(self, lr: float = 2 / 255, *args, **kwargs):
        super(RandomUpdater, self).__init__(lr, *args, **kwargs)

    def update(
        self,
        x: torch.Tensor,
        grad: torch.Tensor,
        step: int,
        steps: int,
        loss: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return x - self.lr * torch.randn_like(x)

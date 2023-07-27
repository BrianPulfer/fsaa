import torch

from fsaa.core import PerturbationUpdater, Scheduler


class RandomUpdater(PerturbationUpdater):
    """Random perturbation update.
    It servers as a baseline for other attacks."""

    def __init__(self,
                 lr: float = 2 / 255,
                 scheduler: Scheduler = None,
                 *args,
                 **kwargs):
        super(RandomUpdater, self).__init__(lr, scheduler, *args, **kwargs)

    def update(
        self,
        x: torch.Tensor,
        grad: torch.Tensor,
        lr: float,
        step: int,
        steps: int,
        loss: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return x - lr * torch.randn_like(x)

import torch

from fsaa.core import PerturbationUpdater, Scheduler


class FGSMUpdater(PerturbationUpdater):
    def __init__(self,
                 lr: float = 2 / 255,
                 scheduler: Scheduler = None,
                 *args,
                 **kwargs):
        super(FGSMUpdater, self).__init__(lr, scheduler, *args, **kwargs)

    def update(
        self,
        x: torch.Tensor,
        grad: torch.Tensor,
        lr: float,
        step: int,
        steps: int,
        loss: torch.Tensor,
    ) -> torch.Tensor:

        return x - lr * grad.sign()

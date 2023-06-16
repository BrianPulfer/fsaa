import torch

from fsaa.core import PerturbationUpdater


class FGSMUpdater(PerturbationUpdater):
    def __init__(self, lr: float = 2 / 255, *args, **kwargs):
        super(FGSMUpdater, self).__init__(lr, *args, **kwargs)

    def update(
        self,
        x: torch.Tensor,
        grad: torch.Tensor,
        step: int,
        steps: int,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        return x - self.lr * grad.sign()

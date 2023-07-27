import torch

from fsaa.core import PerturbationUpdater, Scheduler


class PGDUpdater(PerturbationUpdater):
    def __init__(self,
                 lr: float = 2 / 255,
                 scheduler: Scheduler = None,
                 *args,
                 **kwargs):
        super(PGDUpdater, self).__init__(lr, scheduler, *args, **kwargs)
        self.epsilon = kwargs.get("epsilon", None)

    def update(
        self,
        x: torch.Tensor,
        grad: torch.Tensor,
        lr: float,
        step: int,
        steps: int,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        x_adv = x - lr * grad.sign()

        if self.epsilon is None:
            return x_adv

        delta = torch.clamp(x_adv - x, min=-self.epsilon, max=self.epsilon)
        return x + delta

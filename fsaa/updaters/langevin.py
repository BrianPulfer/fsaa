import torch

from fsaa.core import PerturbationUpdater


class LangevinUpdater(PerturbationUpdater):
    """Langevin perturbation update."""

    def __init__(self,
                 lr: float = 2 / 255,
                 eta: float = None,
                 *args,
                 **kwargs):
        super(LangevinUpdater, self).__init__(lr, *args, **kwargs)
        self.eta = eta if eta is not None else (2 * self.lr) ** 0.5

    def update(
        self,
        x: torch.Tensor,
        grad: torch.Tensor,
        step: int,
        steps: int,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        grad_log_loss = grad / loss
        return x - self.lr * grad_log_loss + self.eta * torch.randn_like(x)

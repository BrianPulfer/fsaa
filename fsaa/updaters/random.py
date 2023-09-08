import torch
from torch import Tensor

from fsaa.core import PerturbationUpdater, Scheduler


class RandomUpdater(PerturbationUpdater):
    """Perturbation update that adds random normally-distributed noise to the input.
    It servers as a baseline for other attacks.

    Args:
        lr (float): Learning rate.
        scheduler (Scheduler): Scheduler to be used.
    """

    def __init__(self,
                 lr: float = 2 / 255,
                 scheduler: Scheduler = None,
                 *args,
                 **kwargs):
        super(RandomUpdater, self).__init__(lr, scheduler, *args, **kwargs)

    def update(
        self,
        x: Tensor,
        grad: Tensor,
        lr: float,
        step: int,
        steps: int,
        loss: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        r"""Updates the adversarial samples with random noise and ignoring the gradient.
        The update is done as follows: :math:`x_{t+1} = x_t - lr * \epsilon_t`, where :math:`\epsilon_t \sim \mathcal{N}(0, 1)`.

        Args:
            x (Tensor): Input tensor.
            grad (Tensor): Gradient tensor.
            lr (float): Learning rate.
            step (int): Current step.
            steps (int): Total number of steps.
            loss (Tensor): Loss tensor.

        Returns:
            Tensor: Updated adversarial samples."""
        return x - lr * torch.randn_like(x)

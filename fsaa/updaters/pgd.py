import torch
from torch import Tensor

from fsaa.core import PerturbationUpdater, Scheduler


class PGDUpdater(PerturbationUpdater):
    """Updates the adversarial samples by moving in the direction of the sign of the gradient."""

    def __init__(self,
                 lr: float = 2 / 255,
                 scheduler: Scheduler = None,
                 *args,
                 **kwargs):
        super(PGDUpdater, self).__init__(lr, scheduler, *args, **kwargs)
        self.epsilon = kwargs.get("epsilon", None)

    def update(
        self,
        x: Tensor,
        grad: Tensor,
        lr: float,
        step: int,
        steps: int,
        loss: Tensor,
    ) -> Tensor:
        r"""Updates the adversarial samples by using the sign of the gradient.
        The update is truncated in case of :math:`\epsilon`-bounded attacks.
        The update is the following: :math:`x_{t+1} = x_t - lr * \text{sign}(\nabla_x L(x_t, y))`.

        Args:
            x (Tensor): Input tensor.
            grad (Tensor): Gradient tensor.
            lr (float): Learning rate.
            step (int): Current step.
            steps (int): Total number of steps.
            loss (Tensor): Loss tensor.

        Returns:
            Tensor: Updated adversarial samples."""
        x_adv = x - lr * grad.sign()

        if self.epsilon is None:
            return x_adv

        delta = torch.clamp(x_adv - x, min=-self.epsilon, max=self.epsilon)
        return x + delta

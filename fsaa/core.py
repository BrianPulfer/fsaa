from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class PerturbationInitializer(ABC):
    """
    Abstract class for perturbation initializers.
    The perturbation initializer is used to initialize the perturbation in the first step of the attack.

    Args:
        lr: Learning rate (or magnitude) for the initial perturbation.
    """

    def __init__(self, lr: float, *args, **kwargs):
        super(PerturbationInitializer, self).__init__(*args, **kwargs)
        self.lr = lr

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Returns the initial perturbation for the given batch.

        Args:
            x (Tensor): Batch to perturb.
        """
        return self.initialize(x, *args, **kwargs)

    @abstractmethod
    def initialize(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Returns the initial perturbation for the given batch.

        Args:
            x (Tensor): Batch to perturb.
        """
        raise NotImplementedError


class Scheduler(ABC):
    """
    Abstract class for schedulers.
    The scheduler adjusts the learning rate (or magnitude) of the update in each step of the attack.

    Args:
        base_lr: Base learning rate (or magnitude) for the update.
    """

    def __init__(self, base_lr: float = 2 / 255, *args, **kwargs):
        super(Scheduler, self).__init__(*args, **kwargs)
        self.base_lr = base_lr

    def __call__(self, step: int, steps: int, *args, **kwargs) -> float:
        """
        Updates the learning rate (or magnitude) of the update for the given step and returns it.

        Args:
            step (int): Current step of the attack.
            steps (int): Total number of steps of the attack.

        Returns:
            float: Learning rate (or magnitude) for the update in the given step.
        """
        return self.get_step_lr(step, steps, *args, **kwargs)

    @abstractmethod
    def get_step_lr(self, step: int, steps: int, *args, **kwargs) -> float:
        """
        Updates the learning rate (or magnitude) of the update for the given step and returns it.

        Args:
            step (int): Current step of the attack.
            steps (int): Total number of steps of the attack.

        Returns:
            float: Learning rate (or magnitude) for the update in the given step.
        """
        raise NotImplementedError


class PerturbationUpdater(ABC):
    """
    Abstract class for perturbation updaters.
    The perturbation updater decides, given the current perturbation and the gradient of the loss with respect to the perturbation, how to update the perturbation.

    Args:
        lr (float): Learning rate (or magnitude) for the update.
        scheduler (Scheduler): Scheduler for the learning rate (or magnitude).
    """

    def __init__(self, lr: float, scheduler: Scheduler = None, *args, **kwargs):
        super(PerturbationUpdater, self).__init__()
        self.lr = lr
        self.scheduler = scheduler

    def __call__(
        self,
        x: Tensor,
        grad: Tensor,
        step: int,
        steps: int,
        loss: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        """
        Updates the perturbation for the current step of the attack.

        Args:
            x (Tensor): Input to perturb.
            grad (Tensor): Gradient of the loss with respect to the perturbation.
            step (int): Current step of the attack.
            steps (int): Total number of steps of the attack.
            loss (Tensor): Loss of the current perturbation.

        Returns:
            Tensor: Updated perturbation.
        """
        lr = self.lr

        if self.scheduler is not None:
            lr = self.scheduler(step, steps)

        return self.update(x, grad, lr, step, steps, loss, *args, **kwargs)

    @abstractmethod
    def update(
        self, x: Tensor, grad: Tensor, lr: float, step: int, steps: int, loss: Tensor
    ) -> Tensor:
        """
        Updates the perturbation for the current step of the attack.

        Args:
            x (Tensor): Input to perturb.
            grad (Tensor): Gradient of the loss with respect to the perturbation.
            lr (float): Learning rate (or magnitude) for this update. Either the base learning rate or the learning rate returned by the scheduler.
            step (int): Current step of the attack.
            steps (int): Total number of steps of the attack.
            loss (Tensor): Loss of the current perturbation.

        Returns:
            Tensor: Updated perturbation.
        """
        raise NotImplementedError


class PerceptualMask(ABC):
    """
    Abstract class for perceptual masks.
    The mask is used to weight the perturbation update after the PerturbationUpdater has been applied.
    """

    def __init__(self, *args, **kwargs):
        super(PerceptualMask, self).__init__()

    def __call__(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Returns the mask for the given batch.

        Args:
            x (Tensor): Batch of original data.
        """
        return self.mask(x)

    @abstractmethod
    def mask(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Returns the mask for the given batch.

        Args:
            x (Tensor): Batch of original data.
        """
        raise NotImplementedError


class DifferentiableTransform(ABC, nn.Module):
    """
    Abstract class for differentiable transforms.
    A differentiable transform is a pre-processing step that is applied to the input before passing it to the model.
    """

    def __init__(self, *args, **kwargs):
        super(DifferentiableTransform, self).__init__()

    def __call__(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Pre-processes the given batch.

        Args:
            x (Tensor): Batch to pre-process.
        """
        return self.process(x)

    @abstractmethod
    def process(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Pre-processes the given batch.

        Args:
            x (Tensor): Batch to pre-process.
        """
        raise NotImplementedError

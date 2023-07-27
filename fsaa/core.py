from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class PerturbationInitializer(ABC):
    def __init__(self, lr, *args, **kwargs):
        super(PerturbationInitializer, self).__init__(*args, **kwargs)
        self.lr = lr

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.initialize(x, *args, **kwargs)

    @abstractmethod
    def initialize(self, x: Tensor, *args, **kwargs) -> Tensor:
        raise NotImplementedError


class Scheduler(ABC):
    def __init__(self, base_lr: float = 2 / 255, *args, **kwargs):
        super(Scheduler, self).__init__(*args, **kwargs)
        self.base_lr = base_lr

    def __call__(self, step: int, steps: int, *args, **kwargs) -> float:
        return self.get_step_lr(step, steps, *args, **kwargs)

    @abstractmethod
    def get_step_lr(self, step: int, steps: int, *args, **kwargs) -> float:
        raise NotImplementedError


class PerturbationUpdater(ABC):
    def __init__(self, lr, scheduler: Scheduler = None, *args, **kwargs):
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
        lr = self.lr

        if self.scheduler is not None:
            lr = self.scheduler(step, steps)

        return self.update(x, grad, lr, step, steps, loss, *args, **kwargs)

    @abstractmethod
    def update(
        self, x: Tensor, grad: Tensor, lr: float, step: int, steps: int, loss: Tensor
    ) -> Tensor:
        raise NotImplementedError


class PerceptualMask(ABC):
    def __init__(self, *args, **kwargs):
        super(PerceptualMask, self).__init__()

    def __call__(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.mask(x)

    @abstractmethod
    def mask(
        self,
        x: Tensor,
    ) -> Tensor:
        raise NotImplementedError


class DifferentiableTransform(ABC, nn.Module):
    def __init__(self, *args, **kwargs):
        super(DifferentiableTransform, self).__init__()

    def __call__(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.process(x)

    @abstractmethod
    def process(
        self,
        x: Tensor,
    ) -> Tensor:
        raise NotImplementedError

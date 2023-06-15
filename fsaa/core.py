from abc import ABC, abstractmethod

from torch import Tensor


class PerturbationInitializer(ABC):
    def __init__(self, alpha, *args, **kwargs):
        super(PerturbationInitializer, self).__init__(*args, **kwargs)
        self.alpha = alpha

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.initialize(x, *args, **kwargs)

    @abstractmethod
    def initialize(self, x: Tensor, *args, **kwargs) -> Tensor:
        raise NotImplementedError


class PerturbationUpdater(ABC):
    def __init__(self, alpha, *args, **kwargs):
        super(PerturbationUpdater, self).__init__()
        self.alpha = alpha

    def __call__(
        self,
        x: Tensor,
        grad: Tensor,
        step: int,
        steps: int,
        loss: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        return self.update(x, grad, step, steps, loss, *args, **kwargs)

    @abstractmethod
    def update(
        self, x: Tensor, grad: Tensor, step: int, steps: int, loss: Tensor
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

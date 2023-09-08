from torch import Tensor
from torch.nn import Module

from fsaa.core import DifferentiableTransform


class TransformAndModelWrapper(Module):
    """
    Creates a wrapper model that uses the given transform before passing the input to the given model.

    Args:
        model (Module): Model to wrap.
        transform (DifferentiableTransform): Transform to apply.
    """

    def __init__(
        self, model: Module, transform: DifferentiableTransform, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.transform = transform

    def forward(self, x: Tensor) -> Tensor:
        return self.model(self.transform(x))

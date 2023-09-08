from typing import List

from torch import Tensor

from fsaa.core import DifferentiableTransform


class Compose(DifferentiableTransform):
    """Composes multiple transforms together sequentially.

    Args:
        transforms (List[DifferentiableTransform]): List of transforms to be composed together.
    """

    def __init__(self,
                 transforms: List[DifferentiableTransform],
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = transforms

    def process(self, x: Tensor) -> Tensor:
        """Returns the transformed input.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Transformed input."""
        for transform in self.transforms:
            x = transform(x)
        return x

from typing import List

from torch import Tensor

from fsaa.core import DifferentiableTransform


class Compose(DifferentiableTransform):
    def __init__(self,
                 transforms: List[DifferentiableTransform],
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = transforms

    def process(self, x: Tensor) -> Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x

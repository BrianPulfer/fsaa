from warnings import warn

import torch
from torch import Tensor

from torchvision.transforms import Normalize as TorchNormalize

from fsaa.core import DifferentiableTransform

HALF_NORMALIZATION_MEAN = [0.5, 0.5, 0.5]
HALF_NORMALIZATION_STD = [0.5, 0.5, 0.5]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

OPENAI_NORMALIZATION_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_NORMALIZATION_STD = [0.26862954, 0.26130258, 0.27577711]


class Normalize(DifferentiableTransform):
    def __init__(self,
                 mean: Tensor = None,
                 std: Tensor = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if mean is None:
            mean = IMAGENET_MEAN
            warn("No mean provided for Normalization: using ImageNet mean.")

        if std is None:
            std = IMAGENET_STD
            warn("No std provided for Normalization: using ImageNet std.")

        self.register_buffer("norm_mean", mean)
        self.register_buffer("norm_std", std)
        self.transform = TorchNormalize(mean, std)

    def process(self, x: Tensor) -> Tensor:
        self.transform = self.transform.to(x.device)
        return self.transform(x)

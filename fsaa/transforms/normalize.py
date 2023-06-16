from warnings import warn

import torch
from torch import Tensor

from fsaa.core import DifferentiableTransform

HALF_NORMALIZATION_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
HALF_NORMALIZATION_STD = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

OPENAI_NORMALIZATION_MEAN = torch.tensor([
    0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)
OPENAI_NORMALIZATION_STD = torch.tensor([
    0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)


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

    def process(self, x: Tensor) -> Tensor:
        mean = self.norm_mean.to(x.device)
        std = self.norm_std.to(x.device)
        return (x - mean) / std

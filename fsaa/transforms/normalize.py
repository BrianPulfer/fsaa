from warnings import warn

from torch import Tensor
from torchvision.transforms import Normalize as TorchNormalize

from fsaa.core import DifferentiableTransform

IMAGENET_INCEPTION_MEAN = [0.5, 0.5, 0.5]
IMAGENET_INCEPTION_STD = [0.5, 0.5, 0.5]

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

        self.transform = TorchNormalize(mean=mean, std=std)

    def process(self, x: Tensor) -> Tensor:
        return self.transform(x)

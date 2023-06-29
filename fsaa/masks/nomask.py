import torch

from fsaa.core import PerceptualMask


class NoMask(PerceptualMask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mask(self, x):
        return torch.ones_like(x)

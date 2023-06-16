"""
Implementation of the Just Noticeable Difference (JND) mask.
Mostly a copy-paste from:
https://github.com/facebookresearch/active_indexing/blob/main/activeindex/attenuations.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fsaa.core import PerceptualMask


class JNDMask(PerceptualMask):
    """https://ieeexplore.ieee.org/document/7885108"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        kernel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        kernel_y = [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]
        kernel_lum = [
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 0, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ]

        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        kernel_lum = torch.FloatTensor(kernel_lum).unsqueeze(0).unsqueeze(0)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
        self.weight_lum = nn.Parameter(data=kernel_lum, requires_grad=False)

        self.clc = kwargs.get("clc", 0.3)

    def mask(self, x):
        assert x.ndim == 4

        jnd_mask = self._jnd(x)
        jnd_mask = jnd_mask / jnd_mask.max()
        jnd_mask = jnd_mask.clamp(0, 1)
        jnd_mask = jnd_mask.repeat(1, 3, 1, 1)
        return jnd_mask

    def _jnd(self, x):
        x = 255 * x
        x = (
            0.299 * x[..., 0:1, :, :]
            + 0.587 * x[..., 1:2, :, :]
            + 0.114 * x[..., 2:3, :, :]
        )
        la = self._jnd_la(x)
        cm = self._jnd_cm(x)
        out = la + cm - self.clc * torch.minimum(la, cm)
        return torch.clamp_min(out, 5) / 255

    def _jnd_la(self, x, alpha=1.0, eps=1e-3):
        """Luminance masking: x must be in [0,255]"""
        dev = x.device
        la = F.conv2d(x, self.weight_lum.to(dev), padding=2) / 32
        mask_lum = la <= 127
        la[mask_lum] = 17 * (1 - torch.sqrt(la[mask_lum] / 127 + eps)) + 3
        la[~mask_lum] = 3 / 128 * (la[~mask_lum] - 127) + 3
        return alpha * la

    def _jnd_cm(self, x, beta=0.117):
        """Contrast masking: x must be in [0,255]"""
        dev = x.device
        grad_x = F.conv2d(x, self.weight_x.to(dev), padding=1)
        grad_y = F.conv2d(x, self.weight_y.to(dev), padding=1)
        cm = torch.sqrt(grad_x**2 + grad_y**2)
        cm = 16 * cm**2.4 / (cm**2 + 26**2)
        return beta * cm


if __name__ == "__main__":
    import requests
    from PIL import Image
    from torchvision.transforms import ToPILImage, ToTensor

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image = ToTensor()(image).unsqueeze(0)
    mask = JNDMask()(image)

    tensor = torch.cat([image, mask.repeat(1, 3, 1, 1)], dim=2).squeeze(0)
    ToPILImage()(tensor).save("debug.png")

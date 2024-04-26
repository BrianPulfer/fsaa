from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests as r
import torch
from PIL import Image
from torchvision.transforms import ToTensor


def plot_activations_3d(
    activations: torch.Tensor,
    cmap: str = "Greens",
    abs: bool = True,
    figsize=(8, 6),
    wspace=0.0,
    rstride=1,
    linewidth=10,
    label_size=15,
):
    feat = activations.squeeze()
    assert feat.ndim == 2, "Input tensor must be 2D"
    X, Y = np.meshgrid(np.arange(feat.shape[-2]), np.arange(feat.shape[-1]))

    fig = plt.figure(figsize=figsize)
    fig.tight_layout()
    plt.subplots_adjust(wspace=wspace)

    ax = fig.add_subplot(1, 1, 1, projection="3d")

    Z = feat
    if abs:
        Z = Z.abs()
    Z = Z.cpu().numpy().T

    ax.plot_surface(X, Y, Z, rstride=rstride, linewidth=linewidth, cmap=cmap)

    ax.tick_params(axis="x", labelsize=label_size)
    ax.tick_params(axis="y", labelsize=label_size)
    ax.tick_params(axis="z", labelsize=label_size)


def get_dummy_image(
    url: str = "http://images.cocodataset.org/val2017/000000039769.jpg",
    size: Tuple[int, int] = (224, 224),
):
    image = Image.open(r.get(url, stream=True).raw).convert("RGB").resize(size)
    return ToTensor()(image).unsqueeze(0)

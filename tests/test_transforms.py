import torch

from fsaa.transforms.normalize import Normalize

ALL_TRANSFORMS = [Normalize]


def test_transforms():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(16, 3, 224, 224).to(device)

    for transform in ALL_TRANSFORMS:
        t = transform()
        tx = t(x)

        assert tx.shape == x.shape
        assert tx.device == x.device

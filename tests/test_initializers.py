import pytest
import torch
from accelerate import Accelerator

from fsaa.utils import INITIALIZERS


@pytest.fixture
def data():
    # Device
    device = Accelerator().device

    # Data
    B, C, H, W = 1, 3, 224, 224
    x = torch.randn(B, C, H, W).clamp(0, 1).to(device)

    return x


def test_initializers(data):
    """Tests that the all initializers return a perturbation"""
    x = data

    for initializer in INITIALIZERS.values():
        adv = initializer()(x)
        assert adv.shape == x.shape
        assert adv.device == x.device

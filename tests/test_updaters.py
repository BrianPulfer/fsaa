import pytest
import torch
from accelerate import Accelerator

from fsaa.utils import UPDATERS


@pytest.fixture
def data():
    # Device
    device = Accelerator().device

    # Data
    B, C, H, W = 1, 3, 224, 224
    x = torch.randn(B, C, H, W).clamp(0, 1).to(device)

    return x


def test_updaters(data):
    """Tests that the all initializers return a perturbation"""
    x = data
    grad = torch.randn_like(x)

    for updater in UPDATERS.values():
        update = updater()(x, grad, step=0, steps=10, loss=1.0)
        assert update.shape == x.shape
        assert update.device == x.device

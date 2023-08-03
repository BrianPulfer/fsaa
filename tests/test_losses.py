import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fsaa.utils import LOSSES, get_loss


@pytest.fixture
def data():
    shape = (1, 3, 224, 224)
    return torch.randn(shape).clamp(0, 1), torch.randn(shape).clamp(0, 1)


def test_custom_losses():
    """Tests that the all custom losses return a loss"""
    for loss_name in LOSSES:
        loss_fn = get_loss(loss_name)
        assert callable(loss_fn)
        assert isinstance(loss_fn, nn.Module)


def test_nn_losses(data):
    x, y = data

    mse_loss = get_loss("MSELoss")
    assert callable(mse_loss)
    loss = mse_loss(x, y)
    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, nn.MSELoss()(x, y))
    assert torch.allclose(loss, (x - y).pow(2).mean())


def test_functional_losses(data):
    x, y = data

    mse_loss = get_loss("mse_loss")
    assert callable(mse_loss)
    loss = mse_loss(x, y)
    assert isinstance(loss, torch.Tensor)
    assert torch.allclose(loss, F.mse_loss(x, y))
    assert torch.allclose(loss, (x - y).pow(2).mean())

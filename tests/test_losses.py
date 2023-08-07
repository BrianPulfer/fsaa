import pytest
import torch
from torch import nn

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
    assert torch.allclose(loss.mean(), nn.MSELoss()(x, y))
    assert torch.allclose(loss.mean(), (x - y).pow(2).mean())

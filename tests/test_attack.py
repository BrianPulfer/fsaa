import pytest
import torch
from accelerate import Accelerator
from torchvision.models.resnet import resnet18

from fsaa.attack import attack
from fsaa.initializers.random import RandomInitializer
from fsaa.losses.mse_loss import MeanSquaredErrorLoss
from fsaa.updaters.fgsm import FGSMUpdater


@pytest.fixture
def model_and_data():
    # Device
    device = Accelerator().device

    # Data
    B, C, H, W = 1, 3, 224, 224
    x = torch.randn(B, C, H, W).clamp(0, 1).to(device)

    return resnet18().eval().to(device), x


def test_attacks(model_and_data):
    """Tests that the all initializers return a perturbation"""
    model, x = model_and_data
    labels = model(x)
    perturbation = attack(
        model,
        x,
        labels,
        1,
        RandomInitializer(),
        FGSMUpdater(),
        MeanSquaredErrorLoss(),
        MeanSquaredErrorLoss(),
        0.0,
        -1.0,
    )

    assert perturbation.shape == x.shape
    assert perturbation.device == x.device
    assert not torch.allclose(perturbation, x)

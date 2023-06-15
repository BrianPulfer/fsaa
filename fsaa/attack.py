import torch
from torch import Tensor
from torch.nn import Module

from fsaa.core import (PerceptualMask, PerturbationInitializer,
                       PerturbationUpdater)
from fsaa.initializers.random import RandomInitializer
from fsaa.losses.mse_loss import MeanSquaredErrorLoss
from fsaa.updaters.pgd import PGDUpdater


def attack(
    model: Module,
    x: Tensor,
    labels: Tensor = None,
    steps: int = 1,
    initializer: PerturbationInitializer = RandomInitializer(),
    updater: PerturbationUpdater = PGDUpdater(),
    image_loss: Module = MeanSquaredErrorLoss(),
    feature_loss: Module = MeanSquaredErrorLoss(),
    ilw: float = 1.0,
    flw: float = -1.0,
    perceptual_mask: PerceptualMask = None,
    device: torch.device = None,
) -> Tensor:
    r"""
    Performs adversarial attack on the given model.


    Args:
        model (Module): Model to attack.
        x (Tensor): Batch of images to attack.
        labels (Tensor): Labels of the batch of images.
        steps (int): Number of steps to perform.
        initializer (PerturbationInitializer): Initializer of the perturbation.
        updater (PerturbationUpdate): Updater of the perturbation.
        image_loss (Module): Image loss function.
        feature_loss (Module): Feature loss function.
        ilw (float): Weight of the image loss.
        flw (float): Weight of the feature loss.
        epsilon (float): Maximum perturbation.
        device (torch.device): Device to use.
    """
    # Set model to eval mode
    model = model.eval()

    # Initialize perturbation and feature labels
    device = x.device if device is None else device
    x = x.clone().detach().to(device)
    x_adv = initializer(x).clone().detach().clamp(0, 1)

    # Copy labels and detach from graph
    if labels is None:
        labels = model(x)
    labels = labels.clone().detach().to(device)

    # Moving losses to device
    image_loss = image_loss.to(device)
    feature_loss = feature_loss.to(device)

    # Getting the mask
    mask = None if perceptual_mask is None else perceptual_mask(x)

    # Performing attack
    best_loss = float("inf")
    best_adv = x_adv
    for step in range(steps):
        # Getting feature representation
        x_adv.requires_grad = True
        features = model(x_adv)

        # Computing the gradient w.r.t loss
        i_loss = 0 if ilw == 0 else ilw * image_loss(x_adv, x).mean()
        f_loss = 0 if flw == 0 else flw * feature_loss(features, labels).mean()
        loss = f_loss + i_loss
        grad = -torch.autograd.grad(loss, x_adv)[0]

        # Masking the gradient
        if mask is not None:
            grad = mask * grad

        # Storing best perturbation
        if best_loss > loss:
            best_loss = loss
            best_adv = x_adv.clone().detach()

        # Updating perturbation
        x_adv = updater(x_adv.detach(), grad, step, steps, loss)
        x_adv = torch.clamp(x, min=0, max=1).detach()

    return best_adv

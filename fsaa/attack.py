from warnings import warn

import torch
from torch import Tensor
from torch.nn import Module
from tqdm.auto import tqdm

from fsaa.core import (DifferentiableTransform, PerceptualMask,
                       PerturbationInitializer, PerturbationUpdater)
from fsaa.initializers.random import RandomInitializer
from fsaa.losses.mse_loss import MeanSquaredErrorLoss
from fsaa.updaters.pgd import PGDUpdater


class TransformAndModelWrapper(Module):
    def __init__(
        self,
        model: Module,
        transform: DifferentiableTransform,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.transform = transform

    def forward(self, x: Tensor) -> Tensor:
        return self.model(self.transform(x))


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
    pbar: bool = True,
    max_img_loss: float = float("inf"),
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
        perceptual_mask (PerceptualMask): Mask to apply to the gradient.
        pbar (bool): Whether to show a progress bar.
        max_img_loss (float): Maximum image loss allowed (without weighting).
        device (torch.device): Device to use.
    """
    assert 0 <= x.min() and x.max() <= 1, "x must be in [0, 1]. "

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
    mask = None if perceptual_mask is None else perceptual_mask(x).detach()

    # Performing attack
    best_loss = float("inf")
    best_adv = x_adv
    bar = range(steps) if not pbar else tqdm(
        range(steps), desc="Attack", leave=False)
    for step in bar:
        # Getting feature representation
        x_adv.requires_grad = True
        features = model(x_adv)

        # Computing the gradient w.r.t loss
        i_loss = 0 if ilw == 0 else ilw * image_loss(x_adv, x).mean()
        f_loss = 0 if flw == 0 else flw * feature_loss(features, labels).mean()
        loss = f_loss + i_loss
        grad = torch.autograd.grad(loss, x_adv)[0]

        if torch.all(grad == 0):
            warn("Gradient is zero. Stopping attack.")
            break

        # Masking the gradient
        if mask is not None:
            grad = mask * grad

        # Storing best perturbation
        if best_loss > loss and (ilw == 0 or i_loss / ilw <= max_img_loss):
            best_loss = loss
            best_adv = x_adv.clone().detach()

        # Updating perturbation
        x_adv = updater(x_adv.detach(), grad, step, steps, loss)
        x_adv = torch.clamp(x_adv, min=0, max=1).detach()

    return best_adv

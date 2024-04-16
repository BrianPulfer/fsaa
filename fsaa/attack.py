from typing import List
from warnings import warn

import torch
from torch import Tensor
from torch.nn import Module
from tqdm.auto import tqdm

from fsaa.core import (PerceptualMask, PerturbationInitializer,
                       PerturbationUpdater)
from fsaa.initializers.random import RandomInitializer
from fsaa.losses.mse_loss import MeanSquaredErrorLoss
from fsaa.transforms.normalize import DEFAULT_IMAGE_RANGE
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
    max_img_loss: float = float("inf"),
    image_range: List[tuple] = DEFAULT_IMAGE_RANGE,
    pbar: bool = True,
    device: torch.device = None,
    verbose: bool = False
) -> Tensor:
    """
    Performs adversarial attack on the given model.


    Args:
        model (Module): Model to attack.
        x (Tensor): Batch of images to attack.
        labels (Tensor): Labels of the batch of images in feature space.
        steps (int): Number of steps to perform.
        initializer (PerturbationInitializer): Initializer of the perturbation.
        updater (PerturbationUpdate): Updater of the perturbation.
        image_loss (Module): Image loss function.
        feature_loss (Module): Feature loss function.
        ilw (float): Weight of the image loss.
        flw (float): Weight of the feature loss.
        perceptual_mask (PerceptualMask): Mask to apply to the gradient.
        max_img_loss (float): Maximum image loss allowed (without weighting).
        image_range (List[tuple]): Image range to clamp the perturbation channel-wise.
        pbar (bool): Whether to show a progress bar.
        device (torch.device): Device to use.
    """
    # Set model to eval mode
    model = model.eval()

    # Initialize perturbation and feature labels
    device = x.device if device is None else device
    x = x.detach().clone().to(device)
    x_adv = initializer(x).detach().clone()

    def clamp_in_range(x, image_range):
        return x.permute(0, 2, 3, 1).clamp(image_range[0], image_range[1]).permute(0, 3, 1, 2)

    if image_range is not None:
        assert image_range.shape == (
            2, 3), "Image range should be channel-wise."
        image_range = image_range.to(device)
        x_adv = clamp_in_range(x_adv, image_range)

    # Copy labels and detach from graph
    if labels is None:
        with torch.no_grad():
            labels = model(x)
    labels = labels.detach().clone().to(device)

    # Moving losses to device
    image_loss = image_loss.to(device)
    feature_loss = feature_loss.to(device)

    # Getting the mask
    mask = None if perceptual_mask is None else perceptual_mask(x).detach()
    if mask is not None:
        assert 0 <= mask.min() and mask.max() <= 1, "Mask should be between 0 and 1."

    # Performing attack
    best_loss = torch.tensor([float("inf")] * len(x)).to(device)
    best_adv = x_adv.detach().clone()
    bar = range(steps) if not pbar else tqdm(
        range(steps), desc="Attack", leave=False)
    for step in bar:
        # Getting feature representation
        x_adv.requires_grad = True
        features = model(x_adv)

        # Computing the gradient w.r.t loss
        i_loss = 0 if ilw == 0 else ilw * image_loss(x_adv, x)
        f_loss = 0 if flw == 0 else flw * feature_loss(features, labels)

        if isinstance(i_loss, Tensor) and i_loss.ndim > 1:
            i_loss = i_loss.mean(dim=list(range(1, i_loss.ndim)))

        if isinstance(f_loss, Tensor) and f_loss.ndim > 1:
            f_loss = f_loss.mean(dim=list(range(1, f_loss.ndim)))

        loss = f_loss + i_loss

        if verbose:
            print(f"Step {step + 1}/{steps}: Loss {loss.mean().item():.4f} - Image Loss {i_loss.mean().item():.4f} - Feature Loss {f_loss.mean().item():.4f}")

        # Storing best perturbation
        update_best = torch.bitwise_and(
            loss < best_loss, ilw == 0 or i_loss / ilw <= max_img_loss
        )
        best_loss[update_best] = loss[update_best].detach()
        best_adv[update_best] = x_adv[update_best].detach().clone()

        grad = torch.autograd.grad(loss.mean(), x_adv)[0]

        if torch.all(grad == 0):
            warn("Gradient is zero. Stopping attack.")
            break

        # Updating perturbation
        x_adv_new = updater(x_adv.detach(), grad, step, steps, loss)

        # Masking the update to be less perceptible
        if mask is not None:
            update = (x_adv_new - x_adv).detach()
            x_adv_new = x_adv.detach() + update * mask

        # Clamping to image range
        if image_range is not None:
            x_adv_new = clamp_in_range(x_adv_new, image_range)

        x_adv = x_adv_new.detach()

    return best_adv

from typing import Callable
from warnings import warn

import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage, ToTensor
from tqdm.auto import tqdm

from fsaa.optimizers.pgd import PGDOptimizer


def to_valid_image(image: torch.Tensor) -> torch.Tensor:
    to_pil = ToPILImage()
    to_tensor = ToTensor()

    image = torch.clamp(image, 0, 1).clone().detach().cpu()
    return to_tensor(to_pil(image))


def reduce_loss(loss: torch.Tensor) -> torch.Tensor:
    if loss.ndim > 1:
        return loss.mean(dim=list(range(1, loss.dim())))
    return loss


def attack(
    model: nn.Module,
    images: torch.Tensor,
    transform: Callable,
    target: torch.Tensor = None,
    steps: int = 1,
    optimizer_fn: torch.optim.Optimizer = PGDOptimizer,
    optimizer_kwargs: dict = {"lr": 1e-4},
    scheduler_fn: torch.optim.lr_scheduler.LRScheduler = None,
    scheduler_kwargs: dict = {},
    image_loss: nn.Module = torch.nn.MSELoss(reduction="none"),
    feature_loss: nn.Module = torch.nn.CosineSimilarity(dim=-1),
    ilw: float = 0.0,
    flw: float = 1.0,
    initial_noise_scale: float = 2 / 255,
    max_img_mse: float = float("inf"),
    do_flatten_features: bool = False,
    device: torch.device = None,
    pbar: bool = False,
) -> torch.Tensor:
    """
    Performs adversarial attack on the given model.

    Args:

    """

    # Sanity checks
    assert isinstance(
        images, torch.Tensor
    ), "Input images must be a torch.Tensor. Use ToTensor transform to convert PIL image to torch.Tensor."
    assert (
        images.min() >= 0 and images.max() <= 1
    ), "Input image must be in range [0, 1]. Pass the transform as an argument to convert the image to the correct range."

    if model.training:
        warn("Attacking model which is in training mode.")

    # Moving model to device
    device = images.device if device is None else device
    model = model.to(device)

    # Initializing adversarial images
    images = images.to(device)
    images_adv = (
        (images + torch.randn_like(images) * initial_noise_scale)
        .clamp(0, 1)
        .to(device)
        .clone()
        .detach()
        .requires_grad_(True)
    )

    # Initializing optimizer and scheduler
    optim = optimizer_fn([images_adv], **optimizer_kwargs)
    scheduler = (
        None if scheduler_fn is None else scheduler_fn(
            optim, **scheduler_kwargs)
    )

    if target is None:
        with torch.no_grad():
            target = model(transform(images))
            if do_flatten_features:
                target = target.flatten(start_dim=1)

    pbar = tqdm(range(steps)) if pbar else range(steps)
    best_losses = torch.ones(images.size(0), device=device) * float("inf")
    best_images_adv = images_adv.clone().detach()
    for _ in pbar:
        pred = model(transform(images_adv))
        if do_flatten_features:
            pred = pred.flatten(start_dim=1)

        f_loss = reduce_loss(feature_loss(pred, target))
        losses = flw * f_loss

        if ilw is not None and ilw != 0:
            i_loss = reduce_loss(image_loss(images_adv, images))
            losses += ilw * i_loss

        loss = losses.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        if scheduler is not None:
            scheduler.step()

        # Keeping image in bounds
        images_adv.data = torch.clamp(images_adv.data, 0, 1)

        # Updating best adversarial images
        update_mask = losses < best_losses

        if max_img_mse < float("inf"):
            mse = (images - images_adv).pow(2).mean(dim=list(range(1, images.dim())))
            has_budget = mse < max_img_mse

            if torch.all(torch.logical_not(has_budget)):
                warn("Stopping attack as max_img_mse is reached for all images.")
                break

            update_mask = torch.logical_and(update_mask, has_budget)

        best_losses[update_mask] = losses[update_mask]
        best_images_adv[update_mask] = images_adv[update_mask].clone().detach()

    # Converting to valid images
    best_images_adv = torch.stack([to_valid_image(img) for img in best_images_adv]).to(
        device
    )
    return best_images_adv


if __name__ == "__main__":
    import requests as r
    from PIL import Image

    from fsaa.models import get_default_transform, get_model

    torch.random.manual_seed(1)

    # Getting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Getting model and transform
    name = "facebook/dinov2-large"
    model = get_model(name).to(device).eval()
    transform = get_default_transform(name)

    # Getting data
    urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "https://farm2.staticflickr.com/1206/1434916947_825c74b04a_z.jpg",
    ]

    images = [
        Image.open(r.get(url, stream=True).raw).convert(
            "RGB").resize((224, 224))
        for url in urls
    ]
    totensor = ToTensor()
    batch = torch.stack([totensor(img) for img in images]).to(device)
    adv_batch = attack(
        model,
        batch,
        transform,
        steps=350,
        max_img_mse=1e-4,
        pbar=True,
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_kwargs={"T_max": 350},
        do_flatten_features=True,
    )

    with torch.no_grad():
        y1 = model(transform(batch))
        y2 = model(transform(adv_batch))

        mses = reduce_loss(
            torch.nn.functional.mse_loss(batch, adv_batch, reduction="none")
        ).cpu()

        cossims = reduce_loss(
            torch.nn.functional.cosine_similarity(
                y1.flatten(1), y2.flatten(1), dim=-1)
        ).cpu()

        for idx, (m, c) in enumerate(zip(mses, cossims)):
            print(
                f"Image {idx+1}\tMSE: {m.item():.4f}, Cosine Similarity: {c.item():.4f}"
            )

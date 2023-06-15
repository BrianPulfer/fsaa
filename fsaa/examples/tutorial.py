import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm.auto import tqdm

from fsaa.attack import attack
from fsaa.utils import get_initializer, get_loss, get_updater


def main():
    # Setting reproducibility
    torch.manual_seed(0)

    # Getting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model to be attacked
    model = resnet18(pretrained=True).to(device).eval()

    # Getting data
    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CIFAR10(
        root="./data/cifar10",
        transform=transform,
        download=True,
        train=False
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Attack parameters: you are flexible to pick different combinations!
    lr = 0.01
    initializer = get_initializer("RandomInitializer", lr)
    updater = get_updater("PGDUpdater", lr)
    img_loss = get_loss("MeanSquaredErrorLoss").to(device)
    feat_loss = get_loss("MeanSquaredErrorLoss").to(device)

    # Attacking batches and storing stats
    images_mses = []
    features_mses = []
    for batch in tqdm(loader):
        imgs, _ = batch
        imgs = imgs.to(device)
        features = model(imgs)

        # Attacking the batch such that features are as
        # different as possible in terms of cosine similarity
        # but close in LPIPS distance to the original images
        adv_batch = attack(
            model,
            imgs,
            labels=features,
            steps=1,
            initializer=initializer,
            updater=updater,
            image_loss=img_loss,
            feature_loss=feat_loss,
            ilw=1,  # Minimize difference in images
            flw=-1,  # Maximize difference in features
            device=device,
        )

        # Storing stats
        features_mses.extend(
            ((features - model(adv_batch)) ** 2)
            .mean(dim=1).detach().cpu().numpy()
        )
        images_mses.extend(
            ((imgs - adv_batch) ** 2)
            .mean(dim=(1, 2, 3)).detach().cpu().numpy()
        )

    # Printing stats
    print(f"Mean MSE in feature space: {np.mean(features_mses):.4f}")
    print(f"Mean MSE in image space: {np.mean(images_mses):.4f}")


if __name__ == "__main__":
    main()

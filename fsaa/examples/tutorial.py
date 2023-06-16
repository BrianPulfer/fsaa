import requests as r
import torch
from PIL import Image
from torchvision.models.resnet import resnet18
from torchvision.transforms import ToTensor

from fsaa.attack import TransformAndModelWrapper, attack
from fsaa.masks.jnd import JNDMask
from fsaa.transforms.normalize import IMAGENET_MEAN, IMAGENET_STD, Normalize
from fsaa.utils import get_initializer, get_loss, get_updater

# Reproducibility
torch.manual_seed(0)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model to be attacked
# Note: we backprop all the way before pre-processing!
model = resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
model = TransformAndModelWrapper(
    model,
    Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
).to(device).eval()

# Batch of data
# No pre-processing is needed (just ToTensor)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(r.get(url, stream=True).raw)
batch = ToTensor()(image).unsqueeze(0).to(device)

# Label for the attack
features = model(batch).detach()

# Attacking the batch
adv_batch = attack(
    model,
    batch,
    labels=features,
    steps=100,
    initializer=get_initializer("Random", 1 / 255),
    updater=get_updater("PGD", lr=2 / 255),
    image_loss=get_loss("MSE"),
    feature_loss=get_loss("MSE"),
    ilw=1,  # Minimize MSE in image space
    flw=-1,  # Maximize MSE in feature space
    perceptual_mask=JNDMask(),
    max_img_loss=0.001,  # Maximum MSE in image space
    device=device,
)

# Comparing image and feature distortions
with torch.no_grad():
    adv_features = model(adv_batch)

mse_f = (features - adv_features).pow(2).mean().item()
mse_i = (batch - adv_batch).pow(2).mean().item()
print(f"MSE in feature space: {mse_f:.4f}")
print(f"MSE in image space: {mse_i:.4f}")

import requests as r
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from fsaa.attack import attack
from fsaa.masks.jnd import JNDMask
from fsaa.utils import get_initializer, get_loss, get_model, get_updater

# Reproducibility
torch.manual_seed(0)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model to be attacked
# The model already comes wrapped with its normalizer!
model = get_model('facebook/vit-mae-base').to(device).eval()

# Batch of data
# No normalization is needed
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(r.get(url, stream=True).raw)
transform = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor()
])
batch = transform(image).unsqueeze(0).to(device)

# Label for the attack
features = model(batch).detach()

# Attacking the batch
adv_batch = attack(
    model,
    batch,
    labels=features,
    steps=350,
    initializer=get_initializer("Random", 4e-4),
    updater=get_updater("PGD", 4e-4),
    image_loss=get_loss("MSE"),
    feature_loss=get_loss("CosSim"),
    ilw=1,  # Minimize MSE in image space
    flw=1,  # Minimize Cosine Similarity in feature space
    perceptual_mask=JNDMask(),
    max_img_loss=0.001,  # Maximum MSE in image space
    device=device,
)

# Comparing image and feature distortions
with torch.no_grad():
    adv_features = model(adv_batch)

cossim = nn.CosineSimilarity(dim=-1)(features, adv_features).mean().item()
mse = (batch - adv_batch).pow(2).mean().item()
print(f"Cosine Similarity in feature space: {cossim:.4f}")
print(f"MSE in image space: {mse:.4f}")

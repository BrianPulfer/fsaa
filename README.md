<h1 align="center">
  <img width="auto" height="150px" src="assets/logo.png" />
</h1>


# FSAA: Feature Space Adversarial Attacks
FSAA allows to create adversarial examples that corrupt the features of the victim models. Various attacks are possible, by altering initialization and update strategies, losses and perceptual masks.
FSAA is written in Python and is based on PyTorch.
___


## Installation
If you would like to use `fsaa` in you project, simply run:
```bash
pip install fsaa
```
___

## Usage example
Here's a generic (and quite complete) [example](fsaa/examples/tutorial.py) on how to use FSAA:

```python
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
```
Which results in an **image MSE of 0.001**, a **feature MSE of 91.7**, and the following corruption:

<center>

| Original | Corrupted | JND Mask |
| :------: | :-------: | :------: |
| <img src="imgs/orig.png" width="300px" /> | <img src="imgs/adv.png" width="300px" /> | <img src="imgs/mask.png" width="300px" />|

</center>

The library also comes with support for pre-trained SSL models from huggingface and other repositories:
```python
from fsaa.utils import get_model, SUPPORTED_MODELS

model = get_model("microsoft/beit-base-patch16-224").eval().to(device)
```
___

## Contributing
Contributions are highly welcome! Please refer to the [contributing guidelines](./CONTRIBUTING.md).
___
## License
The code is distributed according to the **Attribution-NonCommercial 4.0 International** [LICENSE](./LICENSE).
___

## Citation
If you used this library as part of your work, please cite the repository as follows:

```
@software{Pulfer_FSAA_2023,
author = {Pulfer, Brian},
month = jul,
title = {{FSAA}},
url = {https://github.com/BrianPulfer/fsaa},
version = {0.0.8},
year = {2023}
}
```
___

## Acknowledgements
Part of the code was taken and adapted from the following repositories:
  - [facebookresearch/active_indexing](https://github.com/facebookresearch/active_indexing)
    - JND masking

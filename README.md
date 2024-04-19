<h1 align="center">
  <img width="auto" height="150px" src="assets/logo.png" />
</h1>
<a href="https://pypi.org/project/fsaa/" rel="nofollow"><img src="https://img.shields.io/pypi/v/fsaa" alt="PyPi"style="max-width: 100%;"></a>
<a href="https://pypi.org/project/fsaa" rel="nofollow"><img src="https://img.shields.io/pypi/dm/fsaa" alt="Downloads" style="max-width: 100%;"></a>
<a href="https://github.com/BrianPulfer/fsaa" rel="nofollow"><img src="https://img.shields.io/github/stars/BrianPulfer/fsaa?style=social" alt="Downloads" style="max-width: 100%;"></a>


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
```python
import requests as r
from PIL import Image
import torch
from torchvision.transforms import ToTensor

from fsaa import attack
from fsaa.models import get_default_transform, get_model

# Getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Getting model and transform
name = "facebook/dinov2-large"
model = get_model(name).to(device).eval()
transform = get_default_transform(name)

# Getting data
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(r.get(url, stream=True).raw).convert("RGB").resize((224, 224))
batch = ToTensor()(image).unsqueeze(0).to(device)

# Computing attack
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

cossim = torch.nn.functional.cosine_similarity(y1.flatten(1), y2.flatten(1), dim=-1)

print(f"Cosine Similarity: {cossim.item():.4f}")
```

Please refer to the [tutorial notebook](./notebooks/tutorial.ipynb) for a more detailed explanation.
___

## Contributing
Contributions are highly welcome! Please refer to the [contributing guidelines](./CONTRIBUTING.md).
___
## License
The code is distributed according to the **Attribution-NonCommercial 4.0 International** [LICENSE](./LICENSE).
___

## Citation
If you used this library as part of your work, please cite the repository as follows:

```bibtex
@software{Pulfer_FSAA_2024,
author = {Pulfer, Brian},
month = April
title = {{FSAA}},
url = {https://github.com/BrianPulfer/fsaa},
version = {0.0.2},
year = {2024}
}
```
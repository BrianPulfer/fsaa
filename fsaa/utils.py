from typing import Tuple

import requests as r
from PIL import Image
from torchvision.transforms import ToTensor


def get_dummy_image(
    url: str = "http://images.cocodataset.org/val2017/000000039769.jpg",
    size: Tuple[int, int] = (224, 224),
):
    image = Image.open(r.get(url, stream=True).raw).convert("RGB").resize(size)
    return ToTensor()(image).unsqueeze(0)

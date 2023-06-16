import pytest
import requests
from PIL import Image
from torchvision.transforms import ToTensor

from fsaa.utils import MASKS, get_mask


@pytest.fixture
def image():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return ToTensor()(image).unsqueeze(0)


def test_masks(image):
    """Tests that the all initializers return a perturbation"""
    for mask_name in MASKS:
        perceptual_mask = get_mask(mask_name)
        mask = perceptual_mask(image)

        assert mask.shape == image.shape
        assert 0 <= mask.min() <= mask.max() <= 1

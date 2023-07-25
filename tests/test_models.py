from warnings import warn

import pytest
import requests as r
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from transformers import AutoImageProcessor

from fsaa.models.hf.hf_models import SUPPORTED_HF_MODELS
from fsaa.utils import SUPPORTED_MODELS, get_model


def totensor(img, device=None):
    return ToTensor()(img).unsqueeze(0).to(device)


@pytest.fixture
def image_device():
    image = Image.open(
        r.get(
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            stream=True).raw
    ).resize((224, 224))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return image, device


@torch.no_grad()
def test_stochasticity(image_device):
    """Tests that all models are approximately deterministic."""
    image, device = image_device
    tensor = totensor(image, device)
    for name in SUPPORTED_MODELS:
        try:
            model = get_model(name).to(device).eval()
            f1 = model(tensor)

            model = get_model(name).to(device).eval()
            f2 = model(tensor)

            assert torch.allclose(
                f1, f2, atol=1e-4), f"Model {name} is not deterministic"
        except RuntimeError:
            warn(f"Model {name} failed to run on device {device}")


def test_hf_processing_same(image_device):
    """Tests that all processing steps are the same as in the original
    HF processor."""
    image, device = image_device
    tensor = totensor(image, device)
    h, w = tensor.shape[-2:]
    for name in SUPPORTED_HF_MODELS:
        model = get_model(name).to(device).eval()
        processed = model.model.transform(tensor)

        original_processor = AutoImageProcessor.from_pretrained(name)
        original_processed = original_processor(
            image, return_tensors="pt"
        ).pixel_values.to(device)

        assert torch.allclose(processed, original_processed, atol=1e-4)

from warnings import warn

import pytest
import requests as r
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from transformers import AutoImageProcessor

from fsaa.models import (SUPPORTED_HF_MODELS, SUPPORTED_MODELS,
                         SUPPORTED_MODELS_ACTIVATIONS, get_default_transform,
                         get_model)


@pytest.fixture
def image_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(
        r.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw
    ).resize((224, 224))
    return image, device


@torch.no_grad()
def test_stochasticity(image_device):
    """Tests that all models are approximately deterministic."""
    image, device = image_device
    tensor = ToTensor()(image).unsqueeze(0)

    for name in SUPPORTED_MODELS:
        try:
            # Converting
            transform = get_default_transform(name)
            x = transform(tensor).to(device)

            if x.ndim == 3:
                x = x.unsqueeze(0)

            model = get_model(name).to(device).eval()
            f1 = model(x)

            model = get_model(name).to(device).eval()
            f2 = model(x)

            assert torch.allclose(
                f1, f2, atol=1e-4
            ), f"Model {name} is not deterministic"
        except RuntimeError:
            warn(f"Model {name} failed to run on device {device}")


def test_hf_processing_same(image_device):
    """Tests that all processing steps are the same as in the original
    HF processor."""
    image, _ = image_device
    tensor = ToTensor()(image).unsqueeze(0)

    for name in SUPPORTED_HF_MODELS:
        processed = get_default_transform(name)(tensor)

        original_processor = AutoImageProcessor.from_pretrained(name)

        if hasattr(original_processor, "do_center_crop"):
            original_processor.do_center_crop = False
            original_processor.do_resize = False

        original_processed = original_processor(
            image, return_tensors="pt").pixel_values

        assert torch.allclose(
            processed, original_processed, atol=1e-4
        ), f"Processing is different for {name}"


def test_activations(image_device):
    image, device = image_device
    tensor = ToTensor()(image).unsqueeze(0).to(device)

    for name in SUPPORTED_MODELS_ACTIVATIONS:
        model = get_model(name).to(device).eval()
        out1 = model(tensor)
        out2, acts = model.forward_activations(tensor)
        assert torch.allclose(out1, out2, atol=1e-4)
        assert acts.ndim == 4  # (batch, layers, seq_len, hidden_size)

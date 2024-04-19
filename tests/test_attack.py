import requests as r
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from fsaa import attack
from fsaa.models import get_default_transform, get_model


def test_attack():
    torch.random.manual_seed(1)

    # Getting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Getting model and transform
    name = "facebook/dinov2-small"
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
        steps=3,
        max_img_mse=1e-4,
        pbar=True,
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_kwargs={"T_max": 3},
        do_flatten_features=True,
    )

    with torch.no_grad():
        y1 = model(transform(batch))
        y2 = model(transform(adv_batch))

        mses = (
            torch.nn.functional.mse_loss(batch, adv_batch, reduction="none")
            .mean(dim=(1, 2, 3))
            .cpu()
        )

        cossims = torch.nn.functional.cosine_similarity(
            y1.flatten(1), y2.flatten(1), dim=-1
        ).cpu()

    assert (mses < 1e-4).all(), "Mean squared error is higher than desired"
    assert (cossims < 1).all(), "Cosine similarity is not improving"

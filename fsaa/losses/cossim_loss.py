from torch import Tensor
from torch.nn import CosineSimilarity, Module


class CosSimLoss(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = CosineSimilarity()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.criterion(x, y)

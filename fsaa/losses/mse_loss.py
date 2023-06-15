from torch import Tensor
from torch.nn import Module, MSELoss


class MeanSquaredErrorLoss(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = MSELoss()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.criterion(x, y)

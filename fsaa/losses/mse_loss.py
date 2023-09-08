from torch import Tensor
from torch.nn import Module, MSELoss


class MeanSquaredErrorLoss(Module):
    """
    Computes the mean squared error loss between two tensors x and y.
    The reduction is set to none, so the output is a tensor of the same shape as the input.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = MSELoss(reduction="none")

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Computer the mean squared error loss between two tensors x and y.

        Args:
            x (Tensor): the first tensor
            y (Tensor): the second tensor

        Returns:
            Tensor: the mean squared error loss between the two tensors with the same shape as the input
        """
        return self.criterion(x, y)

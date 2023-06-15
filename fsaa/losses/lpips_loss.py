import torch
from lpips import LPIPS
from torch.nn import Module


class LPIPSAlexLoss(Module):
    """Official open-source implementation of the LPIPS loss."""

    def __init__(self):
        super(LPIPSAlexLoss, self).__init__()
        self.net = LPIPS(net="alex")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the LPIPS loss between two tensors x and y.
        Both tensors should be normalized in range [0, 1]."""
        return self.net(x, y, normalize=True).mean()


class LPIPSVGGLoss(Module):
    """Official open-source implementation of the LPIPS loss."""

    def __init__(self):
        super(LPIPSVGGLoss, self).__init__()
        self.net = LPIPS(net="vgg")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the LPIPS loss between two tensors x and y.
        Both tensors should be normalized in range [0, 1]."""
        return self.net(x, y, normalize=True).mean()

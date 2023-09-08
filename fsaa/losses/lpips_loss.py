from lpips import LPIPS
from torch import Tensor
from torch.nn import Module


class LPIPSAlexLoss(Module):
    """Official open-source implementation of the LPIPS loss with the AlexNet Network.
    It captures the perceptual similarity between two images."""

    def __init__(self):
        super(LPIPSAlexLoss, self).__init__()
        self.net = LPIPS(net="alex")

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Computes the LPIPS loss between two tensors x and y.
        Both tensors should be normalized in range [0, 1].

        Args:
            x (Tensor): the first tensor
            y (Tensor): the second tensor

        Returns:
            Tensor: the LPIPS loss between the two tensors using the AlexNet network
        """
        return self.net(x, y, normalize=True)


class LPIPSVGGLoss(Module):
    """Official open-source implementation of the LPIPS loss with the VGG Network.
    It captures the perceptual similarity between two images."""

    def __init__(self):
        super(LPIPSVGGLoss, self).__init__()
        self.net = LPIPS(net="vgg")

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Computes the LPIPS loss between two tensors x and y.
        Both tensors should be normalized in range [0, 1].

        Args:
            x (Tensor): the first tensor
            y (Tensor): the second tensor

        Returns:
            Tensor: the LPIPS loss between the two tensors using the VGG network
        """
        return self.net(x, y, normalize=True)

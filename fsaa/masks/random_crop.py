import torch
from torch import Tensor

from fsaa.core import PerceptualMask


class RandomCropMask(PerceptualMask):
    """Returns a random crop mask.

    Args:
        size (tuple): Size of the crop mask.
    """

    def __init__(self, size):
        super(RandomCropMask, self).__init__()
        self.size = size

    def mask(
        self,
        x: Tensor,
    ) -> Tensor:
        """Returns the random crop mask.

        Args:
            x (Tensor): The input tensor to be masked.

        Returns:
            Tensor: A mask of the same shape as x with ones in the crop region and zeros elsewhere.
        """

        # Get input dimensions
        N, _, H, W = x.shape
        crop_size = self.size

        # Generate random indices for cropping
        max_x = W - crop_size[1]
        max_y = H - crop_size[0]
        random_x = torch.randint(0, max_x + 1, (N,))
        random_y = torch.randint(0, max_y + 1, (N,))

        # Calculate the crop boundaries
        crop_left = random_x
        crop_top = random_y
        crop_right = random_x + crop_size[1]
        crop_bottom = random_y + crop_size[0]

        # Return mask
        mask = torch.zeros_like(x)
        for i in range(N):
            mask[i, :, crop_top[i]:crop_bottom[i],
                 crop_left[i]:crop_right[i]] = 1.0
        return mask

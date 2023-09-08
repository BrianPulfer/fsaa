from torch import Tensor
from torch.nn import CosineSimilarity, Module


class CosSimLoss(Module):
    """
    CosSimLoss is a loss function that computes the cosine similarity between two tensors.
    The cosine similarity is calculated between the flattened tensors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = CosineSimilarity(dim=-1)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Computes the cosine similarity between two tensors x and y.
        Tensors are flattened to 2D before computing the cosine similarity, so the first dimension is the batch size.

        Args:
            x (Tensor): the first tensor
            y (Tensor): the second tensor

        Returns:
            Tensor: the cosine similarity between the two tensors
        """
        return self.criterion(x.flatten(start_dim=1), y.flatten(start_dim=1))

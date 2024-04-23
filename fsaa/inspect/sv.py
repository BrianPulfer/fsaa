from typing import List, Tuple

import torch


def get_singular_values(model: torch.nn.Module) -> List[Tuple[str, List[float]]]:
    """Get singular values of the weight matrices of a model.

    Args:
        model: The model to inspect.

    Returns:
        A list of tuples, where each tuple contains the name of the layer and a list of singular values.
    """
    singular_values = []
    for name, param in model.named_modules():
        if isinstance(param, torch.nn.Linear):
            sv = torch.svd(param.weight, compute_uv=False).S.detach().tolist()
            singular_values.append((name, sv))
    return singular_values

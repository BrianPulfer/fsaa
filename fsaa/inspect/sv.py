from typing import List, Tuple

import torch


def get_singular_values(model: torch.nn.Module) -> List[Tuple[str, List[float]]]:
    """Get singular values of the weight matrices of a model.

    Args:
        model: The model to inspect.

    Returns:
        A list of tuples, where each tuple contains the name of the layer and a list of singular values.
    """
    svs = []
    for name, param in model.named_modules():
        if isinstance(param, torch.nn.Linear):
            # Unpack the weight matrix if it is an attention layer.
            if 'attn' in name and param.out_features == 3 * param.in_features:
                order = ["query", "key", "value"]
                for i in range(3):
                    sv = (
                        torch.svd(
                            param.weight[
                                i * param.in_features: (i + 1) * param.in_features
                            ],
                            compute_uv=False,
                        )
                        .S.detach()
                        .tolist()
                    )
                    svs.append((f"{name}_{order[i]}", sv))
            else:
                sv = torch.svd(
                    param.weight, compute_uv=False).S.detach().tolist()
                svs.append((name, sv))
    return svs

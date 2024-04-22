from typing import Any, Callable, Dict, Iterable, Optional, Union

import torch


class PGDOptimizer(torch.optim.Optimizer):
    """PGD-like optimizer. Updates parameters by adding the sign of the gradient multiplied by the learning rate. Does not update the parameters if the perturbation is greater than epsilon."""

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: float = 2 / 255,
        epsilon: float = 8 / 255,
    ):
        defaults = dict(lr=lr, epsilon=epsilon)
        super(PGDOptimizer, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Args:
            closure (Optional[Callable[[], float]], optional): A closure that reevaluates the model and returns the loss. Defaults to None.

        Returns:
            Optional[float]: The loss value if the closure is provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                pid = id(p)
                if "perturbation" not in group:
                    group["perturbation"] = {}
                    group["perturbation"][pid] = 0

                update = -p.grad.data.sign() * group["lr"]
                update = (
                    torch.clamp(
                        update + group["perturbation"][pid],
                        -group["epsilon"],
                        group["epsilon"],
                    )
                    - group["perturbation"][pid]
                )
                group["perturbation"][pid] += update
                p.data = p.data + update

        return loss

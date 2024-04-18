import torch


# PGD-like Optimizer
class PGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=2 / 255, epsilon=8 / 255):
        defaults = dict(lr=lr, epsilon=epsilon)
        super(PGDOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        # Does one step of PGD
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

                update = - p.grad.data.sign() * group["lr"]
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

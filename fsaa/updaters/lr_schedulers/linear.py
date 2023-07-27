from fsaa.core import Scheduler


class LinearScheduler(Scheduler):
    def __init__(self, base_lr: float = 2 / 255, *args, **kwargs):
        super(LinearScheduler, self).__init__(base_lr)
        self.base_lr = base_lr
        self.target_lr = kwargs.get("target_lr", 0)

    def get_step_lr(self, step: int, steps: int, *args, **kwargs) -> float:
        progress = (steps - step) / steps
        return (self.base_lr - self.target_lr) * progress + self.target_lr

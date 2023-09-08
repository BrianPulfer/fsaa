from fsaa.core import Scheduler


class LinearScheduler(Scheduler):
    r"""Linear learning rate scheduler. The learning rate is linearly decreased from the base learning rate to the target learning rate.
    The learning rate at step :math:`t` is computed as follows: :math:`lr_t = (base\_lr - target\_lr) * (steps - t) / steps + target\_lr`.

    Args:
        base_lr (float): Base learning rate.
        **kwargs: Additional keyword arguments. If 'target_lr' is specified, it is used as the target learning rate. Otherwise, the target learning rate is set to 0."""

    def __init__(self, base_lr: float = 2 / 255, *args, **kwargs):
        super(LinearScheduler, self).__init__(base_lr)
        self.base_lr = base_lr
        self.target_lr = kwargs.get("target_lr", 0)

    def get_step_lr(self, step: int, steps: int, *args, **kwargs) -> float:
        """Gets the learning rate at the current step.

        Args:
            step (int): Current step.
            steps (int): Total number of steps.

        Returns:
            float: Learning rate for the current step.
        """
        progress = (steps - step) / steps
        return (self.base_lr - self.target_lr) * progress + self.target_lr

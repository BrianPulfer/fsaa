from fsaa.core import Scheduler


class FunctionScheduler(Scheduler):
    """Learning rate scheduler that uses a custom function to compute the learning rate at the current step.

    Args:
        base_lr (float): Base learning rate.
        fn (function): Function to be used to compute the learning rate. The function must take a single argument, which is the current step divided by the total number of steps."""

    def __init__(self, base_lr: float = 2 / 255, *args, **kwargs):
        super(FunctionScheduler, self).__init__(base_lr)
        self.base_lr = base_lr
        self.fn = kwargs.get("fn", None)

        assert self.fn is not None, "'fn' keyword must be specified"

    def get_step_lr(self, step: int, steps: int, *args, **kwargs) -> float:
        """Computes the learning rate at the current step.

        Args:
            step (int): Current step.
            steps (int): Total number of steps.

        Returns:
            float: Learning rate for the current step."""
        return self.base_lr * self.fn(step / steps)

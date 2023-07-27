from fsaa.core import Scheduler

class FunctionScheduler(Scheduler):
    def __init__(self, base_lr: float = 2 / 255, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_lr = base_lr
        self.fn = kwargs.get("fn", None)
        
        assert self.fn is not None, "'fn' keyword must be specified for LambdaScheduler"
    
    def get_step_lr(self, step: int, steps: int, *args, **kwargs) -> float:
        return self.base_lr * self.fn(step / steps)

from typing import Dict, Callable, Any

from torch.nn import Module

from shion.core.cached_computation import TensorCachedComputationFunc
from shion.core.loss import Loss


class L2Loss(Loss):
    def __init__(self,
                 expected_func: TensorCachedComputationFunc,
                 actual_func: TensorCachedComputationFunc,
                 weight: float = 1.0):
        self.actual_func = actual_func
        self.expected_func = expected_func
        self.weight = weight

    def compute(
            self,
            modules: Dict[str, Module],
            batch: Any,
            outputs: Dict[str, Any],
            log_func: Callable[[str, float], None] = None):
        expected = self.expected_func(modules, batch, outputs)
        actual = self.actual_func(modules, batch, outputs)
        loss = self.weight * ((expected - actual) ** 2).mean()
        if log_func is not None:
            log_func("loss", loss.item())
        return loss

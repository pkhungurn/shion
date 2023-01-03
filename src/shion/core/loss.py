from abc import ABC, abstractmethod
from typing import Dict, Callable, Any

from torch import Tensor
from torch.nn import Module


class Loss(ABC):
    @abstractmethod
    def compute(
            self,
            modules: Dict[str, Module],
            batch: Any,
            outputs: Dict[str, Any],
            log_func: Callable[[str, float], None] = None) -> Tensor:
        pass

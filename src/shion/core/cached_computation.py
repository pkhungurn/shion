from abc import ABC, abstractmethod
from modulefinder import Module
from typing import Callable, Dict, Any, Optional

from torch import Tensor
from torch.nn import Module

CachedComputationFunc = Callable[
    [
        # modules
        Dict[str, Module],
        # batch
        Any,
        # cached outputs
        Dict[str, Any]
    ],
    Any
]

TensorCachedComputationFunc = Callable[
    [
        # modules
        Dict[str, Module],
        # batch
        Any,
        # cached outputs
        Dict[str, Any]
    ],
    Tensor
]


def create_get_item_func(func: CachedComputationFunc, index):
    def _f(modules: Dict[str, Module], batch: Any, outputs: Dict[str, Any]):
        output = func(modules, batch, outputs)
        return output[index]

    return _f


class CachedComputationProtocol(ABC):
    def get_output(
            self,
            key: str,
            modules: Dict[str, Module],
            batch: Any,
            outputs: Dict[str, Any]) -> Any:
        if key in outputs:
            return outputs[key]
        else:
            output = self.compute_output(key, modules, batch, outputs)
            outputs[key] = output
            return outputs[key]

    @abstractmethod
    def compute_output(
            self,
            key: str,
            modules: Dict[str, Module],
            batch: Any,
            outputs: Dict[str, Any]) -> Any:
        pass

    def get_output_func(self, key: str) -> CachedComputationFunc:
        def func(modules: Dict[str, Module], batch: Any, outputs: Dict[str, Any]):
            return self.get_output(key, modules, batch, outputs)

        return func


CachableCachedComputationStep = Callable[
    [
        CachedComputationProtocol,
        # modules
        Dict[str, Module],
        # batch
        Any,
        # cached outputs
        Dict[str, Any]
    ],
    Any
]


class ComposableCahedComputationProtocol(CachedComputationProtocol):
    def __init__(self, computation_steps: Optional[Dict[str, CachableCachedComputationStep]] = None):
        if computation_steps is None:
            computation_steps = {}
        self.computation_steps = computation_steps

    def compute_output(
            self,
            key: str,
            modules: Dict[str, Module],
            batch: Any,
            outputs: Dict[str, Any]) -> Any:
        if key in self.computation_steps:
            return self.computation_steps[key](self, modules, batch, outputs)
        else:
            raise RuntimeError("Computing output for key " + key + " is not supported!")

from abc import ABC, abstractmethod

from torch.nn import Module


class ModuleAccumulator(ABC):
    @abstractmethod
    def accumulate(self, module: Module, output: Module) -> Module:
        pass

from typing import List, Dict, Callable, Any

import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from shion.core.loss import Loss
from shion.core.optimizer_factory import OptimizerFactory
from shion.core.training.training_protocol import TrainingProtocol
from shion.core.training.validation_protocol import ValidationProtocol

KEY_NETWORK = "network"


class SingleNetworkTrainingProtocol(TrainingProtocol):
    def __init__(self,
                 check_point_examples: List[int],
                 batch_size: int,
                 learning_rate: Callable[[int], Dict[str, float]],
                 optimizer_factories: Dict[str, OptimizerFactory],
                 module_key: str = KEY_NETWORK,
                 random_seed: int = 39549059840):
        super().__init__()
        self.module_key = module_key
        self.optimizer_factories = optimizer_factories
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.check_point_examples = check_point_examples

    def get_optimizer_factories(self) -> Dict[str, OptimizerFactory]:
        return self.optimizer_factories

    def get_checkpoint_examples(self) -> List[int]:
        return self.check_point_examples

    def get_random_seed(self) -> int:
        return self.random_seed

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_learning_rate(self, examples_seen_so_far: int) -> Dict[str, float]:
        return self.learning_rate(examples_seen_so_far)

    def run_training_iteration(
            self,
            batch: Any,
            examples_seen_so_far: int,
            modules: Dict[str, Module],
            optimizers: Dict[str, Optimizer],
            losses: Dict[str, Loss],
            create_log_func: Callable[[str, int], Callable[[str, float], None]],
            device: torch.device):
        generator = modules[self.module_key]
        generator.train(True)
        optimizers[self.module_key].zero_grad()
        log_func = create_log_func("training_" + self.module_key, examples_seen_so_far)
        losses[self.module_key].compute(modules, batch, {}, log_func).backward()
        optimizers[self.module_key].step()


class SingleNetworkValidationProtocol(ValidationProtocol):
    def __init__(
            self,
            example_per_validation_iteration: int,
            batch_size: int,
            module_key: str = KEY_NETWORK):
        super().__init__()
        self.module_key = module_key
        self.batch_size = batch_size
        self.example_per_validation_iteration = example_per_validation_iteration

    def get_batch_size(self, ) -> int:
        return self.batch_size

    def get_examples_per_validation_iteration(self) -> int:
        return self.example_per_validation_iteration

    def run_validation_iteration(
            self,
            batch: Any,
            examples_seen_so_far: int,
            modules: Dict[str, Module],
            losses: Dict[str, Loss],
            create_log_func: Callable[[str, int], Callable[[str, float], None]],
            device: torch.device):
        module = modules[self.module_key]
        module.train(False)
        log_func = create_log_func("validation_" + self.module_key, examples_seen_so_far)
        losses[self.module_key].compute(modules, batch, {}, log_func)

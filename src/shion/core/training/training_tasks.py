import logging
import os
import time
from datetime import datetime
from typing import Optional, Dict, Callable, List

import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from shion.core.loss import Loss
from shion.core.module_accumulator import ModuleAccumulator
from shion.core.module_factory import ModuleFactory
from shion.core.optimizer_factory import OptimizerFactory
from shion.core.training.sample_output_protocol import SampleOutputProtocol
from shion.core.training.training_protocol import TrainingProtocol
from shion.core.training.validation_protocol import ValidationProtocol
from shion.core.load_save import torch_save, torch_load
from pytasuku import Workspace


def optimizer_to_device(optim: Optimizer, device: torch.device):
    for state in optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def zero_module(module: Module):
    parameters = dict(module.named_parameters())
    for k in parameters.keys():
        parameters[k].data.zero_()


class TrainingState:
    def __init__(self,
                 examples_seen_so_far: int,
                 modules: Dict[str, Module],
                 accumulated_modules: Dict[str, Module],
                 optimizers: Dict[str, Optimizer]):
        self.accumulated_modules = accumulated_modules
        self.optimizers = optimizers
        self.modules = modules
        self.examples_seen_so_far = examples_seen_so_far

    @staticmethod
    def get_examples_seen_so_far_file_name(prefix) -> str:
        return prefix + "/examples_seen_so_far.txt"

    @staticmethod
    def get_module_file_name(prefix, module_name) -> str:
        return "%s/module_%s.pt" % (prefix, module_name)

    @staticmethod
    def get_accumulated_module_file_name(prefix, module_name) -> str:
        return "%s/accumulated_%s.pt" % (prefix, module_name)

    @staticmethod
    def get_optimizer_file_name(prefix, module_name) -> str:
        return "%s/optimizer_%s.pt" % (prefix, module_name)

    @staticmethod
    def get_rng_state_file_name(prefix):
        return "%s/rng_state.pt" % prefix

    def save(self, prefix):
        logging.info("Saving training state to %s" % prefix)
        os.makedirs(prefix, exist_ok=True)
        with open(TrainingState.get_examples_seen_so_far_file_name(prefix), "wt") as fout:
            fout.write("%d\n" % self.examples_seen_so_far)
            logging.info("Saved %s" % TrainingState.get_examples_seen_so_far_file_name(prefix))
        for module_name in self.modules:
            file_name = TrainingState.get_module_file_name(prefix, module_name)
            torch_save(self.modules[module_name].state_dict(), file_name)
            logging.info("Saved %s" % file_name)
        for module_name in self.accumulated_modules:
            file_name = TrainingState.get_accumulated_module_file_name(prefix, module_name)
            torch_save(self.accumulated_modules[module_name].state_dict(), file_name)
            logging.info("Saved %s" % file_name)
        for module_name in self.optimizers:
            file_name = TrainingState.get_optimizer_file_name(prefix, module_name)
            torch_save(self.optimizers[module_name].state_dict(), file_name)
            logging.info("Saved %s" % file_name)
        torch_save(torch.get_rng_state(), TrainingState.get_rng_state_file_name(prefix))
        logging.info("Saved %s" % TrainingState.get_rng_state_file_name(prefix))
        logging.info("Done saving training state to %s" % prefix)

    @staticmethod
    def get_examples_seen_so_far(prefix: str) -> int:
        with open(TrainingState.get_examples_seen_so_far_file_name(prefix)) as fin:
            lines = fin.readlines()
            return int(lines[0])

    @staticmethod
    def load(prefix: str,
             module_factories: Dict[str, ModuleFactory],
             accumulators: Dict[str, ModuleAccumulator],
             optimizer_factories: Dict[str, OptimizerFactory],
             device: torch.device) -> 'TrainingState':
        logging.info("Loading training state from %s" % prefix)

        with open(TrainingState.get_examples_seen_so_far_file_name(prefix)) as fin:
            lines = fin.readlines()
            examples_seen_so_far = int(lines[0])
            logging.info("Loaded %s" % TrainingState.get_examples_seen_so_far_file_name(prefix))

        modules = {
            module_name: factory.create()
            for (module_name, factory) in module_factories.items()
        }
        for module_name in modules:
            file_name = TrainingState.get_module_file_name(prefix, module_name)
            modules[module_name].load_state_dict(torch_load(file_name))
            modules[module_name].to(device)
            logging.info("Loaded %s" % file_name)

        accumulated_modules = {}
        for module_name in accumulators:
            module_factory = module_factories[module_name]
            module = module_factory.create()
            file_name = TrainingState.get_accumulated_module_file_name(prefix, module_name)
            module.load_state_dict(torch_load(file_name))
            module.to(device)
            accumulated_modules[module_name] = module
            logging.info("Loaded %s" % file_name)

        optimizers = {}
        for module_name in optimizer_factories:
            optimizer = optimizer_factories[module_name].create(modules[module_name].parameters())
            file_name = TrainingState.get_optimizer_file_name(prefix, module_name)
            optimizer.load_state_dict(torch_load(file_name))
            optimizer_to_device(optimizer, device)
            optimizers[module_name] = optimizer
            logging.info("Loaded %s" % file_name)

        torch.set_rng_state(torch_load(TrainingState.get_rng_state_file_name(prefix)))
        logging.info("Loaded %s" % TrainingState.get_rng_state_file_name(prefix))

        logging.info("Done loading training state from %s" % prefix)

        return TrainingState(examples_seen_so_far, modules, accumulated_modules, optimizers)

    @staticmethod
    def new(module_factories: Dict[str, ModuleFactory],
            accumulators: Dict[str, ModuleAccumulator],
            optimizer_factories: Dict[str, OptimizerFactory],
            random_seed: int,
            device: torch.device) -> 'TrainingState':
        examples_seen_so_far = 0

        modules = {
            module_name: factory.create()
            for (module_name, factory) in module_factories.items()
        }
        for module_name in modules:
            modules[module_name].to(device)

        accumulated_modules = {}
        for module_name in accumulators:
            module_factory = module_factories[module_name]
            module = module_factory.create().to(device)
            zero_module(module)
            accumulated_modules[module_name] = module

        optimizers = {}
        for module_name in optimizer_factories:
            module = modules[module_name]
            optimizer = optimizer_factories[module_name].create(module.parameters())
            optimizer_to_device(optimizer, device)
            optimizers[module_name] = optimizer

        torch.manual_seed(random_seed)

        return TrainingState(examples_seen_so_far, modules, accumulated_modules, optimizers)

    @staticmethod
    def can_load(prefix: str,
                 module_factories: Dict[str, ModuleFactory],
                 accumulators: Dict[str, ModuleAccumulator],
                 optimizer_factories: Dict[str, OptimizerFactory]) -> bool:
        if not os.path.isdir(prefix):
            return False
        if not os.path.isfile(TrainingState.get_examples_seen_so_far_file_name(prefix)):
            return False
        for module_name in module_factories.keys():
            if not os.path.isfile(TrainingState.get_module_file_name(prefix, module_name)):
                return False
        for module_name in accumulators:
            if not os.path.isfile(TrainingState.get_accumulated_module_file_name(prefix, module_name)):
                return False
        for module_name in optimizer_factories:
            if not os.path.isfile(TrainingState.get_optimizer_file_name(prefix, module_name)):
                return False
        if not os.path.isfile(TrainingState.get_rng_state_file_name(prefix)):
            return False
        return True


def get_least_greater_multiple(x: int, m: int) -> int:
    """
    :param x: a non-negative integer
    :param m: a positive integer
    :return: the next multiple of m that is greater than x
    """
    assert x >= 0
    assert m > 0
    return (x // m + 1) * m


def create_log_func(summary_writer, prefix: str, examples_seen_so_far: int) -> Callable[[str, float], None]:
    def log_func(tag: str, value: float):
        summary_writer.add_scalar(prefix + "_" + tag, value, examples_seen_so_far)

    return log_func


def set_learning_rate(module, lr):
    for param_group in module.param_groups:
        param_group['lr'] = lr


class TrainingTasks:
    KEY_CHECKPOINT = 'checkpoint'
    KEY_SNAPSHOT = 'snapshot'
    KEY_VALIDATION = 'validation'
    KEY_SAMPLE_OUTPUT = 'sample_output'

    def __init__(
            self,
            workspace: Workspace,
            prefix: str,
            module_factories: Dict[str, ModuleFactory],
            accumulators: Dict[str, ModuleAccumulator],
            losses: Dict[str, Loss],
            training_dataset: Dataset,
            validation_dataset: Optional[Dataset],
            training_protocol: TrainingProtocol,
            validation_protocol: Optional[ValidationProtocol],
            sample_output_protocol: Optional[SampleOutputProtocol],
            pretrained_module_file_names: Dict[str, str],
            example_per_snapshot: int,
            device: torch.device,
            num_data_loader_workers: int = 8,
            dependencies: Optional[List[str]] = None):
        super().__init__()
        self.num_data_loader_workers = num_data_loader_workers
        self.accumulators = accumulators
        self.device = device
        self.sample_output_protocol = sample_output_protocol
        self.example_per_snapshot = example_per_snapshot
        self.pretrained_module_file_names = pretrained_module_file_names
        self.losses = losses
        self.validation_protocol = validation_protocol
        self.training_protocol = training_protocol
        self.module_factories = module_factories
        self.prefix = prefix
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

        self.checkpoint_examples = self.training_protocol.get_checkpoint_examples()
        assert len(self.checkpoint_examples) >= 1
        assert self.checkpoint_examples[0] > 0
        self.checkpoint_examples = [0] + self.checkpoint_examples

        self.module_names = self.module_factories.keys()
        assert len(self.module_names) > 0

        self.training_data_loader = None
        self.training_data_loader_iter = None
        self.training_data_loader_batch_size = None
        self.validation_data_loader = None
        self.validation_data_loader_iter = None
        self.validation_data_loader_batch_size = None
        self.sample_output_data = None
        self.summary_writer = None
        self.log_dir = None
        self.training_state = None

        if dependencies is None:
            dependencies = []
        self.sample_output_data_task = workspace.create_file_task(
            self.get_sample_output_data_file_name(), dependencies, self.save_sample_output_data)

        module_file_dependencies = [self.sample_output_data_task.name]
        for module_name in pretrained_module_file_names:
            module_file_dependencies.append(self.pretrained_module_file_names[module_name])

        def create_train_func(target_examples: int):
            return lambda: self.train(target_examples)

        train_tasks = []
        for checkpoint_index in range(1, len(self.checkpoint_examples)):
            for module_name in self.module_names:
                module_file_name = TrainingState.get_module_file_name(
                    self.get_checkpoint_prefix(checkpoint_index),
                    module_name)
                workspace.create_file_task(
                    module_file_name,
                    module_file_dependencies,
                    create_train_func(self.checkpoint_examples[checkpoint_index]))
            for module_name in self.accumulators:
                accumulated_module_file_name = TrainingState.get_accumulated_module_file_name(
                    self.get_checkpoint_prefix(checkpoint_index),
                    module_name)
                workspace.create_file_task(
                    accumulated_module_file_name,
                    module_file_dependencies,
                    create_train_func(self.checkpoint_examples[checkpoint_index]))
            workspace.create_command_task(
                self.get_checkpoint_prefix(checkpoint_index) + "/train",
                module_file_dependencies,
                create_train_func(self.checkpoint_examples[checkpoint_index]))
            train_tasks.append(self.get_checkpoint_prefix(checkpoint_index) + "/train")

        self.train_task = workspace.create_file_task(
            self.get_train_command_name(),
            module_file_dependencies,
            create_train_func(self.checkpoint_examples[-1]))

    def get_sample_output_data_file_name(self):
        return self.prefix + "/sample_output_data.pt"

    def save_sample_output_data(self):
        if self.sample_output_protocol is not None:
            torch.manual_seed(self.sample_output_protocol.get_random_seed())
            sample_output_data = self.sample_output_protocol.get_sample_output_data(self.validation_dataset)
            torch_save(sample_output_data, self.get_sample_output_data_file_name())
        else:
            torch_save({}, self.get_sample_output_data_file_name())

    def get_module_file_name(self, checkpoint_index, module_name):
        return TrainingState.get_module_file_name(self.get_checkpoint_prefix(checkpoint_index), module_name)

    def get_last_module_file_name(self, module_name):
        return self.get_module_file_name(len(self.checkpoint_examples) - 1, module_name)

    def get_log_dir(self):
        if self.log_dir is None:
            now = datetime.now()
            self.log_dir = self.prefix + "/log/" + now.strftime("%Y_%m_%d__%H_%M_%S")
        return self.log_dir

    def get_summary_writer(self) -> SummaryWriter:
        if self.summary_writer is None:
            self.summary_writer = SummaryWriter(log_dir=self.get_log_dir())
        return self.summary_writer

    def get_train_command_name(self) -> str:
        return self.prefix + "/train"

    def get_snapshot_prefix(self) -> str:
        return self.prefix + "/snapshot"

    def get_checkpoint_prefix(self, checkpoint_index) -> str:
        return "%s/checkpoint/%04d" % (self.prefix, checkpoint_index)

    def can_load_training_state(self, prefix) -> bool:
        return TrainingState.can_load(
            prefix,
            self.module_factories,
            self.accumulators,
            self.training_protocol.get_optimizer_factories())

    def load_training_state(self, prefix) -> TrainingState:
        return TrainingState.load(
            prefix,
            self.module_factories,
            self.accumulators,
            self.training_protocol.get_optimizer_factories(),
            self.device)

    def get_initial_training_state(self) -> TrainingState:
        training_state = TrainingState.new(
            self.module_factories,
            self.accumulators,
            self.training_protocol.get_optimizer_factories(),
            self.training_protocol.get_random_seed(),
            self.device)
        module_names = self.module_factories.keys()
        for module_name in module_names:
            if module_name in self.pretrained_module_file_names:
                file_name = self.pretrained_module_file_names[module_name]
                training_state.modules[module_name].load_state_dict(torch_load(file_name))
                logging.info("Loaded initial state from %s ..." % file_name)
        logging.info("Created a new initial training state.")
        return training_state

    def load_previous_training_state(self, target_checkpoint_examples: int) -> TrainingState:
        if self.can_load_training_state(self.get_snapshot_prefix()):
            examples_seen_so_far = TrainingState.get_examples_seen_so_far(self.get_snapshot_prefix())
            diff = examples_seen_so_far - target_checkpoint_examples
            if diff < self.training_protocol.get_batch_size():
                return self.load_training_state(self.get_snapshot_prefix())
        num_checkpoints = len(self.checkpoint_examples)
        for checkpoint_index in range(num_checkpoints - 1, -1, -1):
            if self.can_load_training_state(self.get_checkpoint_prefix(checkpoint_index)):
                examples_seen_so_far = TrainingState.get_examples_seen_so_far(
                    self.get_checkpoint_prefix(checkpoint_index))
                diff = examples_seen_so_far - target_checkpoint_examples
                if diff < self.training_protocol.get_batch_size():
                    return self.load_training_state(self.get_checkpoint_prefix(checkpoint_index))
        return self.get_initial_training_state()

    def get_next_checkpoint_num_examples(self, examples_seen_so_far) -> int:
        next_index = next(
            (i for i in range(len(self.checkpoint_examples)) if self.checkpoint_examples[i] > examples_seen_so_far),
            -1)
        return self.checkpoint_examples[next_index]

    def get_next_snapshot_num_examples(self, examples_seen_so_far) -> int:
        return get_least_greater_multiple(examples_seen_so_far, self.example_per_snapshot)

    def get_next_validation_num_examples(self, examples_seen_so_far) -> int:
        if self.validation_protocol is None:
            return -1
        return get_least_greater_multiple(examples_seen_so_far,
                                          self.validation_protocol.get_examples_per_validation_iteration())

    def get_next_sample_output_num_examples(self, examples_seen_so_far) -> int:
        if self.sample_output_protocol is None:
            return -1
        return get_least_greater_multiple(examples_seen_so_far,
                                          self.sample_output_protocol.get_examples_per_sample_output())

    def get_next_num_examples(self, examples_seen_so_far) -> Dict[str, int]:
        return {
            TrainingTasks.KEY_CHECKPOINT: self.get_next_checkpoint_num_examples(examples_seen_so_far),
            TrainingTasks.KEY_SNAPSHOT: self.get_next_snapshot_num_examples(examples_seen_so_far),
            TrainingTasks.KEY_VALIDATION: self.get_next_validation_num_examples(examples_seen_so_far),
            TrainingTasks.KEY_SAMPLE_OUTPUT: self.get_next_sample_output_num_examples(examples_seen_so_far)
        }

    def get_checkpoint_index_to_save(self, examples_seen_so_far: int) -> int:
        checkpoint_index = 0
        for i in range(len(self.checkpoint_examples)):
            if self.checkpoint_examples[i] <= examples_seen_so_far:
                checkpoint_index = i
        return checkpoint_index

    def get_next_training_batch(self):
        if self.training_data_loader is None:
            self.training_data_loader = DataLoader(
                self.training_dataset,
                batch_size=self.training_protocol.get_batch_size(),
                shuffle=True,
                num_workers=self.num_data_loader_workers,
                drop_last=True)
        if self.training_data_loader_iter is None:
            self.training_data_loader_iter = iter(self.training_data_loader)
        try:
            batch = next(self.training_data_loader_iter)
        except StopIteration:
            self.training_data_loader_iter = iter(self.training_data_loader)
            batch = next(self.training_data_loader_iter)
        return [x.to(self.device) for x in batch]

    def get_next_validation_batch(self):
        if self.validation_dataset is None:
            return None
        if self.validation_data_loader is None:
            self.validation_data_loader = DataLoader(
                self.validation_dataset,
                batch_size=self.validation_protocol.get_batch_size(),
                shuffle=True,
                num_workers=self.num_data_loader_workers,
                drop_last=True)
        if self.validation_data_loader_iter is None:
            self.validation_data_loader_iter = iter(self.validation_data_loader)
        try:
            batch = next(self.validation_data_loader_iter)
        except StopIteration:
            self.validation_data_loader_iter = iter(self.validation_data_loader)
            batch = next(self.validation_data_loader_iter)
        return [x.to(self.device) for x in batch]

    def get_checkpoint_index(self, target_checkpoint_examples: int):
        return self.checkpoint_examples.index(target_checkpoint_examples)

    def train(self, target_checkpoint_examples: Optional[int] = None):
        if target_checkpoint_examples is None:
            target_checkpoint_examples = self.checkpoint_examples[-1]

        sample_output_data = torch_load(self.get_sample_output_data_file_name())
        logging.info("Loaded sampled output data from %s", self.get_sample_output_data_file_name())
        training_state = self.load_previous_training_state(target_checkpoint_examples)
        summary_writer = self.get_summary_writer()
        last_time = time.time()

        while training_state.examples_seen_so_far < target_checkpoint_examples:
            # One training iteration
            learning_rate = self.training_protocol.get_learning_rate(training_state.examples_seen_so_far)
            for module_name in self.module_factories.keys():
                if module_name not in learning_rate or module_name not in training_state.optimizers:
                    continue
                lr = learning_rate[module_name]
                set_learning_rate(training_state.optimizers[module_name], lr)
                self.get_summary_writer().add_scalar(
                    module_name + "_learning_rate", lr, training_state.examples_seen_so_far)
            training_batch = self.get_next_training_batch()
            self.training_protocol.run_training_iteration(
                training_batch,
                training_state.examples_seen_so_far,
                training_state.modules,
                training_state.optimizers,
                self.losses,
                lambda name, num: create_log_func(summary_writer, name, num),
                self.device)

            # Accumulate model data
            for module_name in self.accumulators:
                new_module = training_state.modules[module_name]
                buffer_module = training_state.accumulated_modules[module_name]
                self.accumulators[module_name].accumulate(new_module, buffer_module)

            # Advance the number of examples seen so far
            next_num_examples = self.get_next_num_examples(training_state.examples_seen_so_far)
            training_state.examples_seen_so_far += self.training_protocol.get_batch_size()

            # Validation iteration
            if self.validation_protocol is not None \
                    and training_state.examples_seen_so_far >= next_num_examples[TrainingTasks.KEY_VALIDATION]:
                validation_batch = self.get_next_validation_batch()
                self.validation_protocol.run_validation_iteration(
                    validation_batch,
                    training_state.examples_seen_so_far,
                    training_state.modules,
                    self.losses,
                    lambda name, num: create_log_func(summary_writer, name, num),
                    self.device)

            # Save sample output
            if self.sample_output_protocol is not None \
                    and training_state.examples_seen_so_far >= next_num_examples[TrainingTasks.KEY_SAMPLE_OUTPUT]:
                self.sample_output_protocol.save_sample_output_data(
                    training_state.modules,
                    training_state.accumulated_modules,
                    sample_output_data,
                    self.prefix + "/sample_outputs",
                    training_state.examples_seen_so_far,
                    self.device)

            # Save checkpoint
            if training_state.examples_seen_so_far >= next_num_examples[TrainingTasks.KEY_CHECKPOINT]:
                checkpoint_index = self.get_checkpoint_index_to_save(training_state.examples_seen_so_far)
                training_state.save(self.get_checkpoint_prefix(checkpoint_index))

            # Save snapshot
            if training_state.examples_seen_so_far >= next_num_examples[TrainingTasks.KEY_SNAPSHOT]:
                training_state.save(self.get_snapshot_prefix())

            now = time.time()
            if now - last_time > 10:
                logging.info("Showed %d training examples." % training_state.examples_seen_so_far)
                last_time = now

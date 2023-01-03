from typing import List, Tuple, Dict, Callable, Any

import torch
from torch import Tensor
from torch.nn import Module

from shion.core.loss import Loss


class SumLoss(Loss):
    def __init__(self, losses: List[Tuple[str, Loss]]):
        self.losses = losses

    def compute(self,
                modules: Dict[str, Module],
                batch: Any,
                outputs: Dict[str, Any],
                log_func: Callable[[str, float], None] = None) -> Tensor:
        device = batch[0].device
        loss_value = torch.zeros(1, device=device)
        for loss_spec in self.losses:
            loss_name = loss_spec[0]
            loss = loss_spec[1]
            if log_func is not None:
                def loss_log_func(name, value):
                    log_func(loss_name + "_" + name, value)
            else:
                loss_log_func = None
            loss_value = loss_value + loss.compute(modules, batch, outputs, loss_log_func)

        if log_func is not None:
            log_func("loss", loss_value.item())

        return loss_value

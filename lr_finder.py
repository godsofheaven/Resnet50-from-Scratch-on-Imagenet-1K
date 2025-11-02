from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder as TorchLRFinder


def find_lr_torch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    loader: DataLoader,
    start_lr: float = 1e-5,
    end_lr: float = 1.0,
    num_iter: int = 200,
    step_mode: str = "exp",
) -> float:
    """
    Runs torch_lr_finder range test and returns a suggested learning rate.
    """
    model.train()
    # Ensure starting LR
    for pg in optimizer.param_groups:
        pg["lr"] = start_lr

    lr_finder = TorchLRFinder(model, optimizer, criterion, device=(device.type if isinstance(device, torch.device) else str(device)))
    lr_finder.range_test(loader, end_lr=end_lr, num_iter=num_iter, step_mode=step_mode)

    suggested: Optional[float] = None
    try:
        suggestion = lr_finder.suggest_lr()
        if isinstance(suggestion, dict) and "lr" in suggestion:
            suggested = float(suggestion["lr"])
        else:
            suggested = float(suggestion)
    except Exception:
        suggested = None

    if suggested is None:
        lrs = np.array(lr_finder.history.get("lr", []), dtype=float)
        losses = np.array(lr_finder.history.get("loss", []), dtype=float)
        if len(lrs) > 0 and len(losses) == len(lrs):
            suggested = float(lrs[int(np.argmin(losses))] / 10.0)
        else:
            suggested = start_lr

    lr_finder.reset()
    return suggested


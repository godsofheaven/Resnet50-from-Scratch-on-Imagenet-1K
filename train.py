import argparse
import os
import random
import time
from typing import Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR
from torch.utils.data import DataLoader
import torchvision.models as tvm
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import get_imagenet_dataloaders, mixup_collate_fn
from lr_finder import find_lr_torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ResNet50 on ImageNet-1k (from scratch)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=None, help="Base LR; if None and --auto-lr-find, uses LR finder")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true", default=True)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--auto-lr-find", action="store_true")
    p.add_argument("--lr-finder-steps", type=int, default=200)
    p.add_argument("--lr-finder-range", type=float, nargs=2, default=(1e-5, 1.0))
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup-alpha", type=float, default=0.0)
    p.add_argument("--random-erasing", type=float, default=0.1)
    p.add_argument("--use-autoaugment", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="runs")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--cache-dir", type=str, default=None)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))


def setup_distributed() -> Tuple[torch.device, int, int]:
    if is_distributed():
        torch.distributed.init_process_group(backend="nccl")
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1
    return device, rank, world_size


def build_model(num_classes: int = 1000) -> nn.Module:
    model = tvm.resnet50(weights=None)
    # Ensure classifier matches 1000 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_checkpoint(state: dict, is_best: bool, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    last_path = os.path.join(out_dir, "last.pth")
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(out_dir, "best.pth")
        torch.save(state, best_path)


def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scaler: Optional[GradScaler] = None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt and scaler.is_enabled():
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt.get("epoch", 0)
    return start_epoch


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    random_erasing_transform: Optional[torch.nn.Module] = None,
    mixup_alpha: float = 0.0,
    amp: bool = True,
    rank: int = 0,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    total = 0
    pbar = tqdm(loader, desc="train", leave=False, disable=(rank != 0))
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        # Apply RandomErasing on a per-image basis if enabled
        if random_erasing_transform is not None:
            for i in range(images.size(0)):
                images[i] = random_erasing_transform(images[i])

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            outputs = model(images)
            if isinstance(targets, tuple):
                y_a, y_b, lam = targets
                y_a = y_a.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                loss = lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)
                targets_for_acc = y_a
            else:
                targets = targets.to(device, non_blocking=True)
                loss = criterion(outputs, targets)
                targets_for_acc = targets

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        total += bs
        running_loss += loss.item() * bs
        top1 = accuracy(outputs.detach(), targets_for_acc.detach(), topk=(1,))[0]
        running_top1 += top1 * bs
        if rank == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "top1": f"{top1:.2f}%"})

    epoch_loss = running_loss / max(1, total)
    epoch_top1 = running_top1 / max(1, total)
    return epoch_loss, epoch_top1


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True, rank: int = 0) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_top1 = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(loader, desc="valid", leave=False, disable=(rank != 0))
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(enabled=amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        bs = images.size(0)
        total += bs
        running_loss += loss.item() * bs
        top1 = accuracy(outputs, targets, topk=(1,))[0]
        running_top1 += top1 * bs
        if rank == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "top1": f"{top1:.2f}%"})
    return running_loss / max(1, total), running_top1 / max(1, total)


def plot_metrics(output_dir: str, epochs: list, train_losses: list, val_losses: list, train_top1: list, val_top1: list, epoch_idx: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Loss plot
    axes[0].plot(epochs, train_losses, label="train")
    axes[0].plot(epochs, val_losses, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    # Top-1 plot
    axes[1].plot(epochs, train_top1, label="train")
    axes[1].plot(epochs, val_top1, label="val")
    axes[1].set_title("Top-1 Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Top-1 (%)")
    axes[1].legend()
    fig.tight_layout()
    latest_path = os.path.join(output_dir, "metrics_latest.png")
    fig.savefig(latest_path)
    periodic_path = os.path.join(output_dir, f"metrics_epoch_{epoch_idx}.png")
    fig.savefig(periodic_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device, rank, world_size = setup_distributed()

    if rank == 0:
        os.makedirs(args.output, exist_ok=True)

    # Data
    mixup_alpha = max(0.0, float(args.mixup_alpha))
    collate = mixup_collate_fn(mixup_alpha) if mixup_alpha > 0.0 else None
    train_loader, val_loader, random_erasing_transform, _ = get_imagenet_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        img_size=args.img_size,
        cache_dir=args.cache_dir,
        use_autoaugment=args.use_autoaugment,
        random_erasing_prob=args.random_erasing,
    )
    if collate is not None:
        train_loader.collate_fn = collate

    # Model
    model = build_model(num_classes=1000)
    model.to(device)
    if is_distributed():
        model = DDP(model, device_ids=[get_local_rank()], output_device=get_local_rank(), find_unused_parameters=False)

    # Optimizer and criterion
    base_lr = args.lr if args.lr is not None else 0.1
    optimizer = torch.optim.SGD(
        model.parameters(), lr=base_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    scaler = GradScaler(enabled=device.type == "cuda")

    start_epoch = 0

    # Auto LR finder
    if args.auto_lr_find and not args.evaluate and args.lr is None:
        if rank == 0:
            print("Running LR finder...")
        # For LR finder, avoid mixup to get clear signal
        tmp_collate = mixup_collate_fn(0.0)
        train_loader.collate_fn = tmp_collate
        suggested_lr = find_lr_torch(
            model if not isinstance(model, DDP) else model.module,
            optimizer,
            criterion,
            device,
            train_loader,
            start_lr=args.lr_finder_range[0],
            end_lr=args.lr_finder_range[1],
            num_iter=min(args.lr_finder_steps, len(train_loader)),
        )
        # Restore original collate
        if collate is not None:
            train_loader.collate_fn = collate
        else:
            train_loader.collate_fn = None
        if rank == 0:
            print(f"LR finder suggested lr: {suggested_lr:.6f}")
        for pg in optimizer.param_groups:
            pg["lr"] = suggested_lr

    # Scheduler: warmup + cosine
    if args.warmup_epochs > 0:
        def lr_lambda_warmup(epoch):
            return float(epoch + 1) / float(max(1, args.warmup_epochs))
        warmup = LambdaLR(optimizer, lr_lambda=lr_lambda_warmup)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model if not isinstance(model, DDP) else model.module, optimizer, scaler)
        if rank == 0:
            print(f"Resumed from {args.resume}, starting at epoch {start_epoch}")

    # Evaluate only
    if args.evaluate:
        val_loss, val_top1 = evaluate(model if not isinstance(model, DDP) else model.module, val_loader, device, rank=rank)
        if rank == 0:
            print(f"Eval-only: Val Loss {val_loss:.4f} | Val Top-1 {val_top1:.2f}%")
        return

    best_top1 = -1.0
    epoch_indices = []
    train_losses_hist = []
    train_top1_hist = []
    val_losses_hist = []
    val_top1_hist = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_top1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            random_erasing_transform=random_erasing_transform,
            mixup_alpha=mixup_alpha,
            amp=True,
            rank=rank,
        )
        scheduler.step()
        val_loss, val_top1 = evaluate(model if not isinstance(model, DDP) else model.module, val_loader, device, rank=rank)
        dt = time.time() - t0
        if rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"LR {current_lr:.6f} | "
                f"Train Loss {train_loss:.4f} | Train Top-1 {train_top1:.2f}% | "
                f"Val Loss {val_loss:.4f} | Val Top-1 {val_top1:.2f}% | "
                f"{dt/60.0:.2f} min"
            )

            # Record metrics
            epoch_indices.append(epoch + 1)
            train_losses_hist.append(train_loss)
            train_top1_hist.append(train_top1)
            val_losses_hist.append(val_loss)
            val_top1_hist.append(val_top1)

            # Plot every 5 epochs
            if ((epoch + 1) % 5) == 0:
                plot_metrics(args.output, epoch_indices, train_losses_hist, val_losses_hist, train_top1_hist, val_top1_hist, epoch + 1)

            is_best = val_top1 > best_top1
            if is_best:
                best_top1 = val_top1
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "best_top1": best_top1,
                    "args": vars(args),
                },
                is_best=is_best,
                out_dir=args.output,
            )

    if rank == 0:
        print(f"Training complete. Best Val Top-1: {best_top1:.2f}%")


if __name__ == "__main__":
    main()



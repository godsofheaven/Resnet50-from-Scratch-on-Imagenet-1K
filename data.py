import os
from typing import Optional, Tuple, Callable, Any

from datasets import load_dataset
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class HFImageNetDataset(Dataset):
    def __init__(
        self,
        hf_split,
        transform: Optional[Callable] = None,
    ) -> None:
        self.ds = hf_split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.ds[idx]
        image = sample["image"]
        label = int(sample["label"]) if sample["label"] != -1 else -1
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(
    img_size: int = 224,
    use_autoaugment: bool = True,
    random_erasing_prob: float = 0.1,
) -> Tuple[Callable, Callable]:
    train_list = [
        T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
    ]
    if use_autoaugment:
        # ImageNet AutoAugment policy
        train_list.append(T.AutoAugment(T.AutoAugmentPolicy.IMAGENET))
    train_list.extend([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    train_transform = T.Compose(train_list)

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # RandomErasing is applied in the training loop via transforms.RandomErasing on tensors
    # if random_erasing_prob > 0.
    return train_transform, val_transform


def get_imagenet_dataloaders(
    batch_size: int,
    num_workers: int,
    img_size: int = 224,
    cache_dir: Optional[str] = None,
    use_autoaugment: bool = True,
    random_erasing_prob: float = 0.1,
) -> Tuple[DataLoader, DataLoader, Callable, float]:
    """
    Returns train_loader, val_loader, random_erasing_transform, mixup_requires_numpy_flag
    """
    train_transform, val_transform = build_transforms(
        img_size=img_size,
        use_autoaugment=use_autoaugment,
        random_erasing_prob=random_erasing_prob,
    )

    train_hf = load_dataset(
        "ILSVRC/imagenet-1k",
        split="train",
        cache_dir=cache_dir,
    )
    val_hf = load_dataset(
        "ILSVRC/imagenet-1k",
        split="validation",
        cache_dir=cache_dir,
    )

    train_ds = HFImageNetDataset(train_hf, transform=train_transform)
    val_ds = HFImageNetDataset(val_hf, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    random_erasing_transform = None
    if random_erasing_prob and random_erasing_prob > 0.0:
        random_erasing_transform = T.RandomErasing(p=random_erasing_prob)

    return train_loader, val_loader, random_erasing_transform, False


def mixup_collate_fn(alpha: float) -> Callable:
    import numpy as np

    def collate(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        images, targets = list(zip(*batch))
        images = torch.stack(images, dim=0)
        targets = torch.tensor(targets, dtype=torch.long)

        if alpha <= 0.0:
            return images, targets

        lam = np.random.beta(alpha, alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        mixed_images = lam * images + (1.0 - lam) * images[index, :]

        # For mixup, targets are returned as a tuple for criterion handling
        return mixed_images, (targets, targets[index], lam)

    return collate


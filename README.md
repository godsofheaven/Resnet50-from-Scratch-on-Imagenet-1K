ResNet50 ImageNet Training (PyTorch)

This project trains a ResNet50 model from scratch on ImageNet-1k using PyTorch. It includes an LR finder to select a good initial learning rate, mixed precision (AMP), cosine LR schedule with warmup, label smoothing, optional mixup, random erasing, checkpointing, and optional multi-GPU distributed training.

Dataset

- Source: ImageNet-1k on Hugging Face: https://huggingface.co/datasets/ILSVRC/imagenet-1k
- You must log in to Hugging Face and accept the dataset terms to access it.

Quickstart (AWS EC2)

1) Provision an EC2 instance with a CUDA-capable GPU (e.g., g5.xlarge, p3.2xlarge) and NVIDIA drivers.

2) Install dependencies:

   ```bash
   sudo apt update -y
   sudo apt install -y python3-pip
   pip3 install --upgrade pip
   pip3 install -r requirements.txt
   ```

3) Authenticate with Hugging Face and accept dataset terms:

   ```bash
   huggingface-cli login
   # Then visit the dataset page and click "Access repository"
   # https://huggingface.co/datasets/ILSVRC/imagenet-1k
   ```

4) Run LR finder (optional; training script can also do this automatically):

   ```bash
   python3 train.py --auto-lr-find --lr-finder-steps 150 --batch-size 256 --epochs 50
   ```

5) Train (single GPU example):

   ```bash
   python3 train.py --epochs 50 --batch-size 256 --mixup-alpha 0.2 --label-smoothing 0.1 --random-erasing 0.1
   ```

6) Multi-GPU (DDP) example:

   ```bash
   torchrun --standalone --nproc_per_node=4 train.py --epochs 50 --batch-size 128 --mixup-alpha 0.2
   ```

Notes

- The script loads ImageNet directly from the Hugging Face datasets hub and caches locally. Ensure sufficient disk space (hundreds of GB recommended) and network bandwidth.
- Target: ~75%+ top-1 within ~50 epochs with strong aug (AutoAugment), mixup, label smoothing, cosine schedule, and AMP. Actual accuracy depends on hardware, batch size, and tuning.

Resume Training

```bash
python3 train.py --epochs 50 --resume runs/last.pth
```

Evaluate a Checkpoint

```bash
python3 train.py --evaluate --resume runs/best.pth
```

Common Flags

- `--auto-lr-find`: Runs an LR finder pass before training and uses the suggested LR.
- `--warmup-epochs`: Warmup epochs for LR schedule (default 5).
- `--weight-decay`: Weight decay (default 1e-4).
- `--momentum`: SGD momentum (default 0.9, with Nesterov).
- `--random-erasing`: RandomErasing probability (default 0.1; 0 disables).
- `--mixup-alpha`: Enable mixup with given alpha (default 0.0 disables).
- `--cache-dir`: Hugging Face datasets cache directory (default: HF default).
- `--workers`: DataLoader workers per process (default 8; tweak per CPU).

Citation

- ImageNet dataset card on Hugging Face: https://huggingface.co/datasets/ILSVRC/imagenet-1k


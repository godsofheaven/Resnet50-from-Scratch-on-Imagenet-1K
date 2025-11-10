ResNet50 ImageNet Training (PyTorch)

This project trains a ResNet50 model from scratch on ImageNet-1k using PyTorch. It includes an LR finder to select a good initial learning rate, mixed precision (AMP), cosine LR schedule with warmup, label smoothing, optional mixup, random erasing, checkpointing, and optional multi-GPU distributed training.

Dataset

- Source: ImageNet-1k on Hugging Face: https://huggingface.co/datasets/ILSVRC/imagenet-1k
- You must log in to Hugging Face and accept the dataset terms to access it.

Training Notes

- This run used Spot instances; capacity interruptions occurred and training was resumed from checkpoints (see “Resumed from …” lines in the logs).
- After initial phases, LR was fixed at 0.020000 during resumed training, with gradual accuracy gains and small, realistic validation dips.
- Checkpoints were saved under `runs/exp1` (e.g., `last.pth`, `best.pth`), enabling resume after interruptions.
- Final validation top‑1 is 76.78% near the end of training.

Training Logs

```
Epoch 1/50 | LR 0.040000 | Train Loss 6.4824 | Train Top-1 0.92% | Val Loss 5.3892 | Val Top-1 6.06% | 46.04 min
Epoch 2/50 | LR 0.060000 | Train Loss 5.6812 | Train Top-1 4.32% | Val Loss 4.1747 | Val Top-1 17.84% | 46.84 min
Epoch 3/50 | LR 0.080000 | Train Loss 5.1633 | Train Top-1 8.17% | Val Loss 3.6165 | Val Top-1 28.01% | 46.83 min
Epoch 4/50 | LR 0.100000 | Train Loss 4.7780 | Train Top-1 11.79% | Val Loss 3.1862 | Val Top-1 34.85% | 51.26 min
Epoch 5/50 | LR 0.100000 | Train Loss 4.5639 | Train Top-1 13.66% | Val Loss 2.9572 | Val Top-1 37.10% | 57.54 min
Epoch 6/50 | LR 0.099878 | Train Loss 4.4029 | Train Top-1 14.93% | Val Loss 2.9296 | Val Top-1 39.06% | 57.55 min
Epoch 7/50 | LR 0.099513 | Train Loss 4.2959 | Train Top-1 16.36% | Val Loss 2.7295 | Val Top-1 42.14% | 57.46 min
Epoch 8/50 | LR 0.098907 | Train Loss 4.2358 | Train Top-1 17.50% | Val Loss 2.5335 | Val Top-1 45.21% | 57.15 min
Epoch 9/50 | LR 0.098063 | Train Loss 4.1913 | Train Top-1 17.72% | Val Loss 2.5329 | Val Top-1 45.46% | 57.42 min
Epoch 10/50 | LR 0.096985 | Train Loss 4.1256 | Train Top-1 18.39% | Val Loss 2.4731 | Val Top-1 46.93% | 57.65 min
Epoch 11/50 | LR 0.095677 | Train Loss 4.0996 | Train Top-1 18.81% | Val Loss 2.3539 | Val Top-1 48.54% | 57.65 min
Epoch 12/50 | LR 0.094147 | Train Loss 4.0865 | Train Top-1 19.09% | Val Loss 2.5726 | Val Top-1 46.32% | 57.57 min
Epoch 13/50 | LR 0.092402 | Train Loss 4.0519 | Train Top-1 19.69% | Val Loss 2.4294 | Val Top-1 48.41% | 57.66 min
Epoch 14/50 | LR 0.090451 | Train Loss 4.0142 | Train Top-1 20.24% | Val Loss 2.2903 | Val Top-1 50.23% | 57.70 min
Epoch 15/50 | LR 0.088302 | Train Loss 3.9749 | Train Top-1 19.89% | Val Loss 2.3349 | Val Top-1 49.00% | 57.60 min
Epoch 16/50 | LR 0.085967 | Train Loss 3.9949 | Train Top-1 19.83% | Val Loss 2.3103 | Val Top-1 49.81% | 57.61 min
Epoch 17/50 | LR 0.083457 | Train Loss 3.9491 | Train Top-1 20.22% | Val Loss 2.3119 | Val Top-1 50.95% | 57.23 min
Epoch 18/50 | LR 0.080783 | Train Loss 3.9224 | Train Top-1 20.66% | Val Loss 2.2819 | Val Top-1 52.02% | 57.15 min
Epoch 19/50 | LR 0.077960 | Train Loss 3.8839 | Train Top-1 21.47% | Val Loss 2.2406 | Val Top-1 51.39% | 57.19 min
Epoch 20/50 | LR 0.075000 | Train Loss 3.9006 | Train Top-1 20.76% | Val Loss 2.2033 | Val Top-1 52.98% | 57.08 min
Epoch 21/50 | LR 0.071919 | Train Loss 3.8643 | Train Top-1 21.84% | Val Loss 2.1848 | Val Top-1 53.64% | 57.14 min
Epoch 22/50 | LR 0.068730 | Train Loss 3.8350 | Train Top-1 21.84% | Val Loss 2.2851 | Val Top-1 51.57% | 57.26 min
Epoch 23/50 | LR 0.065451 | Train Loss 3.8207 | Train Top-1 22.19% | Val Loss 2.0092 | Val Top-1 55.68% | 57.38 min
Epoch 24/50 | LR 0.062096 | Train Loss 3.7851 | Train Top-1 22.07% | Val Loss 2.2751 | Val Top-1 52.42% | 57.43 min
Epoch 25/50 | LR 0.058682 | Train Loss 3.7810 | Train Top-1 22.77% | Val Loss 2.1426 | Val Top-1 54.27% | 56.82 min
Epoch 26/50 | LR 0.055226 | Train Loss 3.7471 | Train Top-1 23.17% | Val Loss 1.9544 | Val Top-1 56.75% | 46.73 min
Epoch 27/50 | LR 0.051745 | Train Loss 3.7234 | Train Top-1 23.51% | Val Loss 1.9266 | Val Top-1 57.63% | 46.74 min
Epoch 28/50 | LR 0.048255 | Train Loss 3.6910 | Train Top-1 23.79% | Val Loss 1.8666 | Val Top-1 59.06% | 46.73 min
Epoch 29/50 | LR 0.044774 | Train Loss 3.6455 | Train Top-1 24.05% | Val Loss 1.8747 | Val Top-1 59.74% | 46.71 min
Epoch 30/50 | LR 0.041318 | Train Loss 3.6386 | Train Top-1 23.97% | Val Loss 1.9125 | Val Top-1 58.52% | 54.14 min
Epoch 31/50 | LR 0.037904 | Train Loss 3.5832 | Train Top-1 24.91% | Val Loss 1.7265 | Val Top-1 61.08% | 57.35 min
Epoch 32/50 | LR 0.034549 | Train Loss 3.5807 | Train Top-1 25.13% | Val Loss 1.7749 | Val Top-1 61.14% | 57.54 min
Epoch 33/50 | LR 0.031270 | Train Loss 3.5206 | Train Top-1 25.32% | Val Loss 1.8544 | Val Top-1 58.71% | 57.67 min
Epoch 34/50 | LR 0.028081 | Train Loss 3.5113 | Train Top-1 25.39% | Val Loss 1.7576 | Val Top-1 62.94% | 57.67 min
Epoch 35/50 | LR 0.025000 | Train Loss 3.4649 | Train Top-1 26.25% | Val Loss 1.6448 | Val Top-1 63.33% | 57.76 min
Epoch 36/50 | LR 0.022040 | Train Loss 3.4491 | Train Top-1 26.79% | Val Loss 1.5449 | Val Top-1 64.90% | 57.75 min
Resumed from /home/ubuntu/runs/exp1/last.pth, starting at epoch 36
Epoch 37/50 | LR 0.040000 | Train Loss 3.4012 | Train Top-1 26.91% | Val Loss 1.6031 | Val Top-1 65.01% | 45.68 min
Epoch 38/50 | LR 0.060000 | Train Loss 3.5896 | Train Top-1 24.71% | Val Loss 1.7447 | Val Top-1 61.09% | 46.61 min
Epoch 39/50 | LR 0.080000 | Train Loss 3.7603 | Train Top-1 22.78% | Val Loss 1.9652 | Val Top-1 57.75% | 46.61 min
Epoch 40/50 | LR 0.040000 | Train Loss 3.8251 | Train Top-1 21.68% | Val Loss 2.0739 | Val Top-1 54.44% | 45.63 min
Epoch 41/50 | LR 0.040000 | Train Loss 3.4248 | Train Top-1 26.68% | Val Loss 1.8344 | Val Top-1 59.74% | 45.76 min
Epoch 42/50 | LR 0.040049 | Train Loss 3.5297 | Train Top-1 25.49% | Val Loss 1.7310 | Val Top-1 61.27% | 46.78 min
Epoch 43/50 | LR 0.040000 | Train Loss 3.5413 | Train Top-1 25.50% | Val Loss 1.7817 | Val Top-1 61.38% | 46.80 min
Epoch 44/50 | LR 0.039854 | Train Loss 3.5670 | Train Top-1 25.49% | Val Loss 1.8554 | Val Top-1 60.95% | 53.34 min
Epoch 45/50 | LR 0.039611 | Train Loss 3.5450 | Train Top-1 25.34% | Val Loss 1.8121 | Val Top-1 60.56% | 57.47 min
Epoch 46/50 | LR 0.039273 | Train Loss 3.5419 | Train Top-1 24.60% | Val Loss 1.8300 | Val Top-1 59.99% | 57.51 min
Epoch 47/50 | LR 0.038841 | Train Loss 3.5259 | Train Top-1 25.23% | Val Loss 1.7755 | Val Top-1 61.03% | 57.45 min
Epoch 48/50 | LR 0.038318 | Train Loss 3.5294 | Train Top-1 25.95% | Val Loss 1.6565 | Val Top-1 62.52% | 57.43 min
Epoch 49/50 | LR 0.037705 | Train Loss 3.5363 | Train Top-1 25.34% | Val Loss 1.7376 | Val Top-1 61.59% | 57.51 min
Epoch 50/50 | LR 0.037006 | Train Loss 3.5042 | Train Top-1 25.71% | Val Loss 1.7401 | Val Top-1 62.22% | 57.55 min
Training complete. Best Val Top-1: 62.52%
Resumed from /home/ubuntu/runs/exp1/best.pth, starting at epoch 48
Epoch 49/70 | LR 0.038318 | Train Loss 2.8772 | Train Top-1 53.97% | Val Loss 1.6725 | Val Top-1 61.63% | 45.62 min
Epoch 50/70 | LR 0.020000 | Train Loss 2.6473 | Train Top-1 59.34% | Val Loss 1.4764 | Val Top-1 66.19% | 46.38 min
Epoch 51/70 | LR 0.020000 | Train Loss 2.6258 | Train Top-1 59.85% | Val Loss 1.4087 | Val Top-1 67.33% | 46.45 min
Epoch 52/70 | LR 0.020000 | Train Loss 2.6247 | Train Top-1 59.88% | Val Loss 1.4714 | Val Top-1 65.90% | 46.41 min
Epoch 53/70 | LR 0.020000 | Train Loss 2.6296 | Train Top-1 59.73% | Val Loss 1.4031 | Val Top-1 67.20% | 52.50 min
Epoch 54/70 | LR 0.020000 | Train Loss 2.6290 | Train Top-1 59.74% | Val Loss 1.4328 | Val Top-1 66.74% | 56.63 min
Epoch 55/70 | LR 0.020000 | Train Loss 2.6284 | Train Top-1 59.74% | Val Loss 1.4423 | Val Top-1 66.75% | 56.64 min
Resumed from /home/ubuntu/runs/exp1/last.pth, starting at epoch 55
Epoch 56/90 | LR 0.020000 | Train Loss 2.6261 | Train Top-1 59.91% | Val Loss 1.4703 | Val Top-1 66.01% | 45.87 min
Epoch 57/90 | LR 0.020000 | Train Loss 2.6600 | Train Top-1 59.08% | Val Loss 1.4393 | Val Top-1 66.75% | 46.16 min
Epoch 58/90 | LR 0.020000 | Train Loss 2.6534 | Train Top-1 59.18% | Val Loss 1.4729 | Val Top-1 66.38% | 46.04 min
Epoch 59/90 | LR 0.020000 | Train Loss 2.6494 | Train Top-1 59.30% | Val Loss 1.4179 | Val Top-1 67.16% | 46.08 min
Epoch 60/90 | LR 0.020000 | Train Loss 2.6465 | Train Top-1 59.37% | Val Loss 1.4189 | Val Top-1 67.10% | 52.25 min
Epoch 61/90 | LR 0.020000 | Train Loss 2.6468 | Train Top-1 59.40% | Val Loss 1.3990 | Val Top-1 67.61% | 56.65 min
Epoch 62/90 | LR 0.020000 | Train Loss 2.6437 | Train Top-1 59.48% | Val Loss 1.4338 | Val Top-1 67.09% | 56.70 min
Epoch 63/90 | LR 0.020000 | Train Loss 2.6410 | Train Top-1 59.55% | Val Loss 1.4903 | Val Top-1 65.68% | 56.69 min
Epoch 64/90 | LR 0.020000 | Train Loss 2.6375 | Train Top-1 59.58% | Val Loss 1.4029 | Val Top-1 67.27% | 56.73 min
Epoch 65/90 | LR 0.020000 | Train Loss 2.6349 | Train Top-1 59.65% | Val Loss 1.4370 | Val Top-1 66.78% | 56.79 min
Epoch 66/90 | LR 0.020000 | Train Loss 2.6322 | Train Top-1 59.67% | Val Loss 1.4291 | Val Top-1 66.79% | 56.67 min
Epoch 67/90 | LR 0.020000 | Train Loss 2.6326 | Train Top-1 59.66% | Val Loss 1.4563 | Val Top-1 66.70% | 56.72 min
Epoch 68/90 | LR 0.020000 | Train Loss 2.6269 | Train Top-1 59.81% | Val Loss 1.3636 | Val Top-1 68.73% | 56.78 min
Resumed from /home/ubuntu/runs/exp1/last.pth, starting at epoch 68
Epoch 69/90 | LR 0.020000 | Train Loss 2.6100 | Train Top-1 60.23% | Val Loss 1.4012 | Val Top-1 67.56% | 49.36 min
Epoch 70/90 | LR 0.018383 | Train Loss 2.5848 | Train Top-1 60.78% | Val Loss 1.3743 | Val Top-1 68.22% | 56.74 min
Epoch 71/90 | LR 0.016824 | Train Loss 2.5539 | Train Top-1 61.50% | Val Loss 1.3516 | Val Top-1 68.88% | 56.88 min
Epoch 72/90 | LR 0.015324 | Train Loss 2.5232 | Train Top-1 62.31% | Val Loss 1.3232 | Val Top-1 69.18% | 56.82 min
Epoch 73/90 | LR 0.013885 | Train Loss 2.4932 | Train Top-1 62.97% | Val Loss 1.3128 | Val Top-1 69.59% | 56.85 min
Epoch 74/90 | LR 0.012509 | Train Loss 2.4672 | Train Top-1 63.68% | Val Loss 1.2633 | Val Top-1 70.47% | 56.88 min
Epoch 75/90 | LR 0.011198 | Train Loss 2.4368 | Train Top-1 64.34% | Val Loss 1.2666 | Val Top-1 70.83% | 56.90 min
Epoch 76/90 | LR 0.009953 | Train Loss 2.4058 | Train Top-1 65.11% | Val Loss 1.2404 | Val Top-1 71.28% | 56.91 min
Epoch 77/90 | LR 0.008775 | Train Loss 2.3737 | Train Top-1 65.88% | Val Loss 1.1957 | Val Top-1 72.10% | 56.86 min
Epoch 78/90 | LR 0.007667 | Train Loss 2.3413 | Train Top-1 66.65% | Val Loss 1.1826 | Val Top-1 72.62% | 56.90 min
Epoch 79/90 | LR 0.006629 | Train Loss 2.3090 | Train Top-1 67.44% | Val Loss 1.1526 | Val Top-1 73.02% | 56.85 min
Epoch 80/90 | LR 0.005663 | Train Loss 2.2777 | Train Top-1 68.21% | Val Loss 1.1367 | Val Top-1 73.65% | 56.78 min
Epoch 81/90 | LR 0.004769 | Train Loss 2.2428 | Train Top-1 69.10% | Val Loss 1.1212 | Val Top-1 74.07% | 56.79 min
Epoch 82/90 | LR 0.003950 | Train Loss 2.2370 | Train Top-1 69.18% | Val Loss 1.0892 | Val Top-1 74.50% | 47.39 min
Epoch 83/90 | LR 0.003206 | Train Loss 2.2077 | Train Top-1 69.90% | Val Loss 1.0725 | Val Top-1 74.97% | 46.28 min
Epoch 84/90 | LR 0.002537 | Train Loss 2.1788 | Train Top-1 70.57% | Val Loss 1.0531 | Val Top-1 75.34% | 46.52 min
Epoch 85/90 | LR 0.001946 | Train Loss 2.1523 | Train Top-1 71.24% | Val Loss 1.0420 | Val Top-1 75.73% | 46.43 min
Epoch 86/90 | LR 0.001431 | Train Loss 2.1274 | Train Top-1 71.94% | Val Loss 1.0279 | Val Top-1 76.05% | 51.78 min
Epoch 87/90 | LR 0.000995 | Train Loss 2.1063 | Train Top-1 72.45% | Val Loss 1.0144 | Val Top-1 76.35% | 57.15 min
Epoch 88/90 | LR 0.000637 | Train Loss 2.0887 | Train Top-1 72.90% | Val Loss 1.0076 | Val Top-1 76.55% | 57.17 min
Epoch 89/90 | LR 0.000359 | Train Loss 2.0773 | Train Top-1 73.20% | Val Loss 1.0085 | Val Top-1 76.69% | 57.14 min
Epoch 90/90 | LR 0.000160 | Train Loss 2.0683 | Train Top-1 73.46% | Val Loss 0.9949 | Val Top-1 76.78% | 57.13 min
Training complete. Best Val Top-1: 76.78%
```

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

With mixed precision (recommended on recent NVIDIA GPUs):

```bash
python3 train.py --epochs 50 --batch-size 256 --mixup-alpha 0.2 --label-smoothing 0.1 --random-erasing 0.1 --precision bf16
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
- `--precision`: Numerical precision: `fp32`, `fp16`, or `bf16` (default `fp16`).
  - `fp16`: Enables AMP with GradScaler on CUDA.
  - `bf16`: Enables AMP (no scaler) on CUDA; recommended on Ampere+.
  - `fp32`: Disables AMP. On CPU, training always effectively runs in fp32.

Citation

- ImageNet dataset card on Hugging Face: https://huggingface.co/datasets/ILSVRC/imagenet-1k


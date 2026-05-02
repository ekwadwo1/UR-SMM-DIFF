# UR-SMM-DIFF

# UR-SSM-Diff: Uncertainty-Rectified State-Space Latent Diffusion for Physics-Consistent 3D MRI Artifact Correction and Tumor Quantification

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-red.svg)](https://pytorch.org/)

Official implementation of **UR-SSM-Diff**, a physics-grounded latent diffusion framework for joint 3D MRI artifact correction and brain tumor segmentation.

<p align="center">
  <img src="assets/fig_architecture.png" width="90%" alt="UR-SSM-Diff Architecture"/>
</p>

## Highlights

- **Physics-grounded diffusion**: Forward process derived from MRI acquisition physics (motion, ghosting, undersampling, noise) rather than isotropic Gaussian noise
- **State-space denoiser**: Mamba-based SSM backbone with tri-orientated scanning for efficient long-range 3D context at O(n) complexity
- **Uncertainty-Rectified Attention (URA) gate**: Learned gating between heteroscedastic variance (data-dependent) and physics-anchored variance schedules
- **Joint restoration + segmentation**: Single model, single forward pass — artifact correction and tumor quantification simultaneously
- **43M trainable parameters, 3.41 GB VRAM, 1.56 s/volume** inference

## Results

### BraTS 2021 — 5-Fold Cross-Validation

| Model | Category | DSC↑<sub>WT</sub> | DSC↑<sub>TC</sub> | DSC↑<sub>ET</sub> | HD95↓<sub>WT</sub> |
|-------|----------|-----------|-----------|-----------|------------|
| **UR-SSM-Diff (Ours)** | **Generative (SSM)** | **0.865±0.007** | **0.856±0.009** | **0.764±0.012** | **4.82±0.41** |
| SegMamba | SSM (discrim.) | 0.846±0.015 | 0.839±0.007 | 0.742±0.005 | 6.92±0.72 |
| nnU-Net | CNN (discrim.) | 0.831±0.011 | — | — | — |
| SwinUNETR | Transformer | 0.832±0.009 | — | — | — |
| Fast-DDPM | Generative | 0.809±0.012 | — | — | — |
| Standard 3D DDPM | Generative (SSM) | 0.745±0.014 | — | — | — |

## Repository Structure

```
UR-SSM-Diff/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
│
├── configs/                        # YAML configuration files
│   ├── default.yaml                #   Default hyperparameters
│   ├── vqgan.yaml                  #   VQ-GAN training config
│   └── train.yaml                  #   Main training config (3 phases)
│
├── data/                           # Data loading and preprocessing
│   ├── __init__.py
│   ├── brats_dataset.py            #   BraTS 2021 PyTorch Dataset
│   ├── brats_preprocess.py         #   BraTS NIfTI → .npy preprocessing
│   ├── fastmri_preprocess.py       #   fastMRI HDF5 → .npy preprocessing
│   └── run_all_preprocessing.py    #   Unified preprocessing entry point
│
├── physics/                        # MRI physics corruption model
│   ├── __init__.py
│   └── corruption_operator.py      #   PhysicsCorruptionOperator (H_ξ)
│
├── models/                         # Neural network architectures
│   ├── __init__.py
│   ├── vqgan3d.py                  #   3D VQ-GAN (encoder/decoder/quantizer)
│   ├── ssm_block.py                #   Mamba SSM blocks (tri-orientated scan)
│   ├── denoiser.py                 #   SSM-based ε_θ denoiser network
│   ├── ura_gate.py                 #   Uncertainty-Rectified Attention gate
│   ├── seg_head.py                 #   Segmentation head (latent → labels)
│   └── ur_ssm_diff.py              #   Full model wrapper (all components)
│
├── diffusion/                      # Diffusion process
│   ├── __init__.py
│   └── physics_forward.py          #   Physics-grounded forward process & Tweedie
│
├── losses/                         # Loss functions
│   ├── __init__.py
│   ├── diffusion_loss.py           #   ε-prediction + physics-consistency loss
│   └── segmentation_loss.py        #   Dice + CE segmentation loss
│
├── evaluation/                     # Evaluation and metrics
│   ├── __init__.py
│   └── metrics.py                  #   DSC, HD95, Surface Dice, PSNR, SSIM, ECE
│
├── experiments/                    # Training and evaluation scripts
│   ├── train.py                    #   Main 3-phase training pipeline (DDP)
│   ├── train_vqgan.py              #   VQ-GAN pretraining script
│   ├── evaluate.py                 #   Inference and evaluation runner
│   ├── baselines.py                #   Baselines 1–5 (nnU-Net, DDPM, etc.)
│   ├── new_baselines.py            #   Baselines 6–8 (SegMamba, TransBTS, etc.)
│   └── eval_baseline.py            #   Post-hoc baseline metric evaluation
│
├── visualization/                  # Figure generation
│   ├── __init__.py
│   └── figures.py                  #   Publication figures (main + supplementary)
│
├── scripts/                        # Utility shell scripts
│   ├── download_brats2021.sh       #   Download BraTS 2021 dataset
│   ├── preprocess.sh               #   Run full preprocessing pipeline
│   └── train_all_folds.sh          #   Launch 5-fold CV training
│
└── assets/                         # Images for README
    ├── fig_architecture.png
    └── fig_results.png
```

## Installation

### Prerequisites

- Linux (Ubuntu 20.04/22.04/24.04)
- NVIDIA GPU with ≥24 GB VRAM (tested on RTX 5880 Ada 48 GB)
- CUDA 11.8
- Python 3.10

### Setup

```bash
# Clone repository
git clone https://github.com/<your-username>/UR-SSM-Diff.git
cd UR-SSM-Diff

# Create conda environment
conda create -n ur_ssm_diff python=3.10 -y
conda activate ur_ssm_diff

# Install PyTorch (CUDA 11.8)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Install mamba-ssm (requires CUDA toolkit)
pip install mamba-ssm==2.2.4 causal-conv1d==1.4.0

# Install remaining dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "
import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
import mamba_ssm; print(f'mamba-ssm {mamba_ssm.__version__}')
from models.ur_ssm_diff import build_ur_ssm_diff; print('Model imports OK')
"
```

## Data Preparation

### BraTS 2021

1. Download BraTS 2021 Training Data from [Synapse](https://www.synapse.org/#!Synapse:syn27046444)

2. Preprocess:
```bash
python data/brats_preprocess.py \
    --input-dir /path/to/RSNA_ASNR_MICCAI_BraTS2021_TrainingData \
    --output-dir data/preprocessed/BraTS2021 \
    --target-shape 128 128 128
```

This produces per-subject directories with `image.npy` (4×128³ float32) and `label.npy` (128³ int64), plus `fold_0.json` through `fold_4.json` for cross-validation splits.

### fastMRI (Optional — for Generalization)

```bash
python data/fastmri_preprocess.py \
    --input-dir /path/to/fastMRI/brain \
    --output-dir data/preprocessed/fastMRI \
    --target-shape 128 128 128
```

## Training

Training proceeds in three phases, following the curriculum described in the paper.

### Phase 0: VQ-GAN Pretraining

```bash
torchrun --nproc_per_node=2 experiments/train_vqgan.py \
    --data-root data/preprocessed/BraTS2021 \
    --output-dir checkpoints/vqgan_r4 \
    --epochs 200 --codebook-size 8192 --commitment-weight 0.25
```

### Phases 1–3: Main Training (per fold)

```bash
# Single fold
torchrun --nproc_per_node=2 experiments/train.py \
    --fold 0 \
    --data-root data/preprocessed/BraTS2021 \
    --vqgan-ckpt checkpoints/vqgan_r4/best_vqgan.pt \
    --output-dir training/

# All 5 folds
for fold in 0 1 2 3 4; do
    torchrun --nproc_per_node=2 experiments/train.py \
        --fold $fold \
        --data-root data/preprocessed/BraTS2021 \
        --vqgan-ckpt checkpoints/vqgan_r4/best_vqgan.pt \
        --output-dir training/
done
```

**Phase schedule** (automated within `train.py`):
| Phase | Epochs | γ (seg weight) | Description |
|-------|--------|----------------|-------------|
| 1 | 1–50 | 0.0 | Diffusion-only (restoration) |
| 2 | 51–150 | 0.0→1.0 | Joint training (linear ramp) |
| 3 | 151–200 | 1.0 | Fine-tuning with full seg weight |

### Baselines

```bash
# nnU-Net, SwinUNETR, Standard DDPM, Fast-DDPM, MedSegDiff
torchrun --nproc_per_node=2 experiments/baselines.py \
    --baseline nnunet --fold 0 \
    --data-root data/preprocessed/BraTS2021 \
    --output-dir baselines/

# SegMamba, TransBTS, Restore→nnU-Net
torchrun --nproc_per_node=2 experiments/new_baselines.py \
    --baseline segmamba --fold 0 \
    --data-root data/preprocessed/BraTS2021 \
    --output-dir baselines/

# Single-GPU mode (when one GPU is occupied)
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 \
    experiments/new_baselines.py --baseline transbts --fold 0 \
    --data-root data/preprocessed/BraTS2021 --output-dir baselines/
```

## Evaluation

```bash
# Full evaluation with all metrics
python experiments/evaluate.py \
    --fold 0 \
    --data-root data/preprocessed/BraTS2021 \
    --vqgan-ckpt checkpoints/vqgan_r4/best_vqgan.pt \
    --ckpt training/fold_0/phase3/best_model.pt \
    --output-dir results/

# Baseline evaluation (post-hoc metrics)
python experiments/eval_baseline.py \
    --baseline segmamba --fold 0 \
    --data-root data/preprocessed/BraTS2021 \
    --output-dir baselines/
```

### Metrics

**Segmentation**: DSC (WT/TC/ET), HD95 (mm), Surface Dice τ=1 mm

**Restoration**: PSNR (dB), SSIM — both brain-masked 3D

**Calibration**: Expected Calibration Error (ECE)

**Efficiency**: Parameters, VRAM, inference time per volume

## Model Architecture

```
Input x_0 [B,4,128³]
    │
    ▼
┌─────────────────────────┐
│  Physics Corruption H_ξ │ ── motion, ghosting, undersampling, noise
└─────────┬───────────────┘
          │ y = H_ξ(x_0)
          ▼
┌─────────────────────────┐
│   VQ-GAN Encoder E_φ    │ ── 3D → latent space (4× downsampled)
└─────────┬───────────────┘
          │ z_obs = E_φ(y)
          ▼
┌─────────────────────────┐
│  Physics-Grounded       │
│  Forward Diffusion      │ ── q(z_t | z_0, ξ) with physics schedule
│  q(z_t | z_0, ξ)        │
└─────────┬───────────────┘
          │ z_t
          ▼
┌─────────────────────────┐
│  SSM Denoiser ε_θ       │ ── Mamba blocks, tri-orientated scanning
│  + URA Gate             │ ── σ²_het ↔ σ²_phys gating
│  + Seg Head             │ ── latent → class logits
└───┬─────────┬───────────┘
    │         │
    ▼         ▼
  ε̂, σ̂²    ŝ (seg logits)
    │
    ▼
┌─────────────────────────┐
│  Tweedie Estimate       │ ── x̂_0 = f(z_t, ε̂, t)
│  + VQ-GAN Decoder D_φ   │
└─────────┬───────────────┘
          │
          ▼
    x̂_0 [B,4,128³]  (restored)
    ŝ   [B,4,128³]  (segmentation)
```

## Pretrained Weights

| Model | Download | DSC<sub>WT</sub> | Size |
|-------|----------|-----------|------|
| VQ-GAN (r=4) | [Link]() | — | ~50 MB |
| UR-SSM-Diff (fold 0) | [Link]() | 0.866 | ~170 MB |
| UR-SSM-Diff (all folds) | [Link]() | 0.865±0.007 | ~850 MB |

*Links will be populated upon paper acceptance.*

## Citation

```bibtex
@article{your_name2026urssmdiff,
  title     = {{UR-SSM-Diff}: Uncertainty-Rectified State-Space Latent Diffusion
               for Physics-Consistent {3D} {MRI} Artifact Correction and
               Tumor Quantification},
  author    = {Your Name and Co-Author Names},
  journal   = {IEEE Transactions on Medical Imaging},
  year      = {2026},
  volume    = {},
  number    = {},
  pages     = {},
  doi       = {10.1109/TMI.2026.XXXXXXX}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- BraTS 2021 dataset organizers (RSNA-ASNR-MICCAI)
- [mamba-ssm](https://github.com/state-spaces/mamba) by Albert Gu and Tri Dao
- [MONAI](https://monai.io/) for medical imaging utilities
- TThis work was supported by the National Natural Science Foundation of China (62276116); Six talent peaks project in Jiangsu Province (DZXX-122), and the Jiangsu Province Maternal and Child Health Research Project (F202322), Zhanjiang Social and Development Project (SH2024091), and the China Postdoctoral Science Foundation (2024M751192).

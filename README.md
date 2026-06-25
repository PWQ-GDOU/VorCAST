# VorCAST — Vorticity Advanced Spatiotemporal foreCAST

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![DOI](https://img.shields.io/badge/Data-GridRad--Severe-orange)](https://gridrad.org/)

**Physics-informed deep learning for tornadic vertical vorticity prediction from NEXRAD radar observations.**

VorCAST (Vorticity Advanced Spatiotemporal foreCAST) is a 3D convolutional neural network that fuses NEXRAD radar reflectivity, Doppler velocity moments, storm motion vectors, and a differentiable physics module to forecast the evolution of tornadic vertical vorticity fields up to 3 hours ahead.

---

## 🎯 Key Features

- **3D ResUNet Encoder-Decoder** — Full-resolution (128×128×29) spatiotemporal processing with GroupNorm for small-batch stability
- **Differentiable Physics Module** — Solves the vorticity tendency equation (advection + tilting/stretching + diffusion + baroclinic) as a trainable computational layer
- **Six-Channel Multi-Source Input** — Radar reflectivity, spectrum width, azimuthal shear, divergence, and storm motion vectors (u_storm, v_storm)
- **Multi-Scale Evaluation Suite** — Fractions Skill Score (FSS) at 5 neighborhood scales (2–42 km), soft CSI, LPIPS perceptual similarity, and AUC-ROC
- **Dual Interface** — Rich Textual TUI (training monitor + inference) and pure CLI mode for HPC/headless deployments
- **DCU Accelerator Support** — Verified on Hygon K100_AI DCU GPUs (ROCm/DTK 24.04–26.04) with AMP mixed-precision training

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Input (T=12, 6 chan)                │
│  [Reflectivity | SpectrumWidth | AzShear | Divergence   │
│                         | u_storm | v_storm]            │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              3D ResUNet Encoder (depth=2)               │
│         Conv3D → GN → ReLU → Residual Blocks            │
│             No spatial downsampling (128²)              │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│               Physics Module (per timestep)             │
│      ∂ζ/∂t = −v·∇ζ  +  ω·∇w  +  ν_t∇²ζ  +  B         │
│      (advection)   (stretching)   (diffusion)   (baroclinic)│
│              ← Learnable diffusion coefficients →       │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│         3D ResUNet Decoder + Skip Connections           │
│        FiLM-conditioned on Δt for temporal awareness    │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Output (T=36) — Vertical Vorticity ζ      │
└─────────────────────────────────────────────────────────┘
```

## 📦 Project Structure

```
trainer_app/
├── main.py                    # Entry point (TUI / CLI / Worker)
├── config_default.yaml        # Full configuration with documentation
├── requirements.txt
├── data/
│   ├── preprocess.py          # GridRad + storm track → training samples
│   ├── hrrr_reader.py         # HRRR environmental field reader & diagnostics
│   ├── dataset.py             # PyTorch Dataset with raw + processed targets
│   └── split.py               # Train/val/test split by storm event
├── models/
│   ├── encoder.py             # 3D ResUNet with GroupNorm
│   ├── physics.py             # Differentiable vorticity equation solver
│   ├── integrator.py          # Time integration loop
│   ├── loss.py                # BCE + MAE + CSI + FSS + LPIPS + AUC
│   ├── metrics.py             # Soft (trainable) + hard (eval) metrics
│   └── inference.py           # Multi-step autoregressive prediction
├── training/
│   ├── trainer.py             # Training loop with AMP + checkpointing
│   ├── monitor.py             # Live metric logging & visualization
│   ├── checkpoint.py          # Save/resume with best-model tracking
│   └── worker.py              # Background worker process
├── tui/                       # Textual TUI application
│   ├── app.py                 # Main TUI app
│   └── screens/               # Menu, training, inference, config screens
├── history/                   # Experiment tracking & query
├── utils/
│   ├── config.py              # YAML config with schema validation
│   ├── device.py              # GPU/DCU detection
│   └── visualization.py       # Vorticity field plotting
└── tests/                     # Unit & integration tests
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU (CUDA 11.8+) or Hygon DCU (DTK 24.04+)
- 8+ GB GPU memory recommended (works with 4 GB via chunked decoding)

### Installation

```bash
# Clone the repository
git clone https://github.com/PWQ-GDOU/VorCAST.git
cd VorCAST/trainer_app

# Install dependencies
pip install -r trainer_app/requirements.txt
```

### Data Preparation

VorCAST uses the **GridRad-Severe** dataset ([gridrad.org](https://gridrad.org/)) paired with **HRRR** environmental fields.

```bash
# 1. Download NEXRAD 3D radar data (.nc) and storm track CSV files
#    Place them in separate directories

# 2. Run preprocessing
python -m trainer_app.main --cli \
    --dataset1 /path/to/nexrad_nc_files \
    --dataset2 /path/to/storm_tracks_csv

# Preprocessing automatically:
#   - Extracts spatial windows around tracked storms
#   - Aligns radar timestamps with storm tracks
#   - Applies Gaussian filtering to vorticity labels
#   - Normalizes per-variable (Min-Max)
#   - Splits into train/val/test by storm event
#   - Outputs compressed .npz files (~60–80 GB for full GridRad-Severe)
```

**Input Channels (6):**

| # | Variable | Description | Source |
|---|----------|-------------|--------|
| 1 | Reflectivity | Radar reflectivity factor | NEXRAD |
| 2 | SpectrumWidth | Doppler velocity spectrum width | NEXRAD |
| 3 | AzShear | Azimuthal shear | NEXRAD |
| 4 | Divergence | Radial divergence | NEXRAD |
| 5 | storm_u | Storm motion U-component (m/s) | Storm track CSV |
| 6 | storm_v | Storm motion V-component (m/s) | Storm track CSV |

> **Optional extensions**: HRRR wind fields (u, v, w), CAPE, CIN, SRH (see `config_default.yaml`).

### Training

**TUI Mode** (recommended for interactive use):

```bash
python -m trainer_app.main
# → Interactive menu: select datasets → configure → start training
```

**CLI Mode** (for HPC / SLURM jobs):

```bash
python -m trainer_app.main --cli \
    --dataset1 /path/to/nexrad \
    --dataset2 /path/to/tracks \
    --gpu \
    --config my_config.yaml
```

### Inference

```bash
python -m trainer_app.main --infer \
    --ckpt checkpoints/best_model.pth \
    --input /path/to/storm_sample.npz \
    --output ./predictions
```

Outputs per sample:
- `pred_3d.npy` — Predicted 3D vorticity field (T=36, H=128, W=128, L=29)
- `target_3d.npy` — Ground truth vorticity
- `metrics.json` — CSI, FSSₖ, LPIPS, AUC-ROC

## 📊 Evaluation Metrics

VorCAST implements a comprehensive meteorological verification suite:

| Metric | Training | Inference | Description |
|--------|----------|-----------|-------------|
| **CSI** (IoU) | Soft (sigmoid proxy) | Exact | Critical Success Index — hit / (hit + miss + false alarm) |
| **FSSₖ** | Soft (differentiable) | Exact | Fractions Skill Score at 5 scales: k = 1, 3, 5, 11, 21 (~2–42 km) |
| **LPIPS** | VGG perceptual loss | Exact | Learned Perceptual Image Patch Similarity |
| **AUC-ROC** | Soft ranking proxy | Exact | Discriminative skill for vorticity threshold |
| **BCE** | ✓ | — | Binary cross-entropy with class-weighted positive samples |
| **MAE** | ✓ | — | Mean absolute error on raw vorticity values |

### Understanding FSS Scales

> From Roberts & Lean (2008): FSS > 0.5 indicates the forecast has "useful" skill at that neighborhood scale.

| FSS Scale | Grid Cells | Approx. Radius | Physical Meaning |
|-----------|-----------|----------------|------------------|
| FSS₁ | 1×1 | 2 km | Exact location match |
| FSS₃ | 3×3 | 6 km | Tornado core (~mesocyclone) |
| FSS₅ | 5×5 | 10 km | Storm cell scale |
| FSS₁₁ | 11×11 | 22 km | Supercell scale |
| FSS₂₁ | 21×21 | 42 km | Storm environment |

## 🧪 Physics Module

The physics module solves the **vertical vorticity tendency equation** using finite-difference operators implemented as differentiable PyTorch layers:

$$\frac{\partial\zeta}{\partial t} = -\underbrace{\mathbf{v}\cdot\nabla\zeta}_{\text{advection}} + \underbrace{\boldsymbol{\omega}\cdot\nabla w}_{\text{tilting + stretching}} + \underbrace{\nu_t\nabla^2\zeta}_{\text{diffusion}} + \underbrace{\frac{1}{\rho}\left(\frac{\partial\rho}{\partial x}\frac{\partial p}{\partial y} - \frac{\partial\rho}{\partial y}\frac{\partial p}{\partial x}\right)}_{\text{baroclinic (optional)}}$$

**Key innovations:**
- **Learnable diffusion coefficients** (νₓ, νᵧ, ν_𝓏) with Softplus activation to ensure positivity
- **FiLM-conditioned decoder** on Δt to handle variable forecast lead times
- **No spatial downsampling** in encoder → physics operates at full NEXRAD resolution (~2 km)

## 🔧 Configuration

All hyperparameters are managed in `config_default.yaml`. Key sections:

```yaml
model:
  in_channels: 72           # T_in × channels = 12 × 6
  base_channels: 64
  depth: 2                  # Encoder stages
  encoder_downsample: false # Full-resolution mode
  decoder_chunk_size: 4     # Temporal chunking for memory control

training:
  batch_size: 4             # DCU-optimized
  learning_rate: 0.001
  use_amp: true             # Automatic Mixed Precision

data:
  history_steps: 12         # 1 hour input window
  future_steps: 36          # 3 hour forecast horizon
  grid_size: 128            # Spatial resolution
  in_channels: 6            # Radar + storm motion
```

## 🖥️ DCU Deployment

VorCAST has been validated on **Hygon K100_AI DCU accelerators** (ROCm/DTK ecosystem):

| Environment | Status |
|-------------|--------|
| DTK 24.04 + PyTorch 2.1.0 | ✅ Verified (AMP, bs=4, ~84 s/batch) |
| DTK 25.04 + PyTorch 2.5+ | ⬜ Planned (torch.compile + MIOpen fix) |
| DTK 26.04 + PyTorch 2.9.0 | ⬜ Image prepared |

**DCU-specific considerations:**
- `decoder_chunk_size: 4` — required to avoid MIOpen 3D convolution issues
- `use_amp: true` — strongly recommended (AMP mitigates DCU Conv3D bugs)
- **GroupNorm** instead of BatchNorm — essential for small batch sizes on DCU

## 📈 Known Benchmark Results

Evaluated on GridRad-Severe test set (GR-S HRRR-Inflow, 164 storm events):

| Threshold | FSS₁ (2 km) | FSS₃ (6 km) | FSS₅ (10 km) | FSS₁₁ (22 km) | FSS₂₁ (42 km) |
|-----------|-------------|-------------|--------------|---------------|---------------|
| P50 | 0.37 | **0.51** ★ | 0.60 | 0.69 | 0.76 |
| P75 | 0.28 | 0.41 | 0.50 | 0.60 | 0.68 |
| P90 | 0.20 | 0.31 | 0.39 | 0.50 | 0.59 |
| P95 | 0.15 | 0.24 | 0.32 | 0.41 | 0.49 |
| P99 | 0.08 | 0.14 | 0.20 | 0.27 | 0.36 |

> ★ FSS > 0.5 at 6 km scale = **operationally useful forecast skill** at mesocyclone scale

**Key finding**: The model captures vorticity shape and intensity correctly but exhibits a systematic **translation bias** (position offset of several km). This is the primary bottleneck — the addition of HRRR wind fields as input channels is expected to address this.

## 📝 Citation

If you use VorCAST in your research, please cite:

```bibtex
@software{VorCAST2026,
  author  = {PWQ-GDOU},
  title   = {VorCAST: Vorticity Advanced Spatiotemporal foreCAST},
  year    = {2026},
  url     = {https://github.com/PWQ-GDOU/VorCAST}
}
```

## 📚 References

1. **Roberts, N.M. & Lean, H.W.** (2008). Scale-Selective Verification of Rainfall Accumulations from High-Resolution Forecasts of Convective Events. *Monthly Weather Review*, 136(1), 78–97.
2. **Mittermaier, M. & Roberts, N.** (2010). Intercomparison of Spatial Forecast Verification Methods. *Weather and Forecasting*, 25(5), 1416–1430.
3. **Homeyer, C.R. et al.** (2021). GridRad-Severe: A Database of Three-Dimensional Gridded NEXRAD Data for Use in Severe Weather Research. *Journal of Geophysical Research: Atmospheres*, 126(18).
4. **Zhang, R. et al.** (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *CVPR*.
5. **Flournoy, M.D. et al.** (2020). Supercell Environments Using GridRad-Severe and the HRRR. *Weather and Forecasting*, 35(6), 2259–2280.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

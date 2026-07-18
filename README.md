<div align="center">

# TensorGated-SSM

### *TensorGated State-Space Model for Spatial-Spectral Representation Learning in Hyperspectral Image Classification*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Scientific%20Reports-red?logo=readthedocs)](https://www.nature.com/srep/)

<br/>

> A novel hyperspectral image classification framework that fuses a **Spectral-Gated Attention (SGA)** module with a **recurrent State-Space Model (SSM)** to jointly model long-range spatial dependencies and discriminative spectral signatures within a unified tensor-gated architecture.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Single-Model Reference (main.py)](#single-model-reference-mainpy)
  - [Unified Baseline Runner (baseline.py)](#unified-baseline-runner-baselinepy)
  - [Batch Scripts](#batch-scripts)
  - [Table Generation](#table-generation)
  - [Ablation Study (Notebook)](#ablation-study-notebook)
- [Outputs](#outputs)
- [Supported Datasets](#supported-datasets)
- [Supported Baseline Models](#supported-baseline-models)
- [Project Structure](#project-structure)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Overview

Hyperspectral image classification (HSIC) requires simultaneously exploiting the rich spectral signatures of hundreds of contiguous bands and the spatial context of neighboring pixels. Existing CNN-based approaches are limited by local receptive fields, while standard attention mechanisms scale poorly with the large number of spatial tokens.

**TensorGated-SSM** addresses these challenges through three tightly coupled components:

1. **Dual-path Tokenization** — Generates informationally distinct spatial tokens (from CNN features) and spectral tokens (from raw PCA bands) so that each branch carries a genuinely different signal.
2. **Spectral-Gated Attention (SGA)** — Computes per-band statistics (mean + std of each PCA component) across the patch, projects them into a channel gate, and uses that gate to selectively amplify or suppress attention outputs — making the model's attention *spectrally aware*.
3. **Recurrent State-Space Model (SSM)** — Scans the spatial token sequence with a learned state transition, capturing long-range dependencies across the full patch without the quadratic cost of full self-attention.

---

## Architecture

```
Input Patch [B, 14, 14, k]
        │
        ▼
┌──────────────────────┐
│  CNN Pre-Feature     │  Conv2D(16) → BN → Conv2D(32) → BN
│  Extractor           │  Output: [B, 14, 14, 32]
└──────────┬───────────┘
           │  x_cnn                    x_raw (original PCA patch)
           ▼                                  │
┌──────────────────────────────────────────────┐
│     Dual-Path Tokenization                   │
│  spatial_tokens ← CNN features  (proj→d)    │
│  spectral_tokens ← PCA values   (proj→d)    │
│  Output: [B, H×W, d]  ×2                   │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│   Spectral-Spatial Feature Enhancement       │
│   channel-wise sigmoid gate (center-token)   │
│   Output: enhanced spatial & spectral tokens │
└──────────────────┬───────────────────────────┘
                   │                   x_raw
                   ▼                     │
┌──────────────────────────────────────────────┐
│     Spectral-Gated Attention (SGA)           │
│  Q = spatial · K = spectral · V = spectral   │
│  gate ← f(mean(x_raw), std(x_raw))           │
│  out  ← gate ⊗ attn_out + (1−gate) ⊗ Q       │
│  Output: [B, H×W, d]                        │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│        State-Space Model (SSM)               │
│   h_t = W_h·h_{t-1} + W_x·x_t              │
│   Output: [B, state_dim]                     │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
             Classifier (Dense → Softmax)
             Output: [B, num_classes]
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Spectral-Gated Attention** | Per-band mean + std gate — spectral awareness without extra parameters |
| **Recurrent SSM** | Long-range spatial dependencies at linear cost |
| **Dual-Path Tokens** | CNN spatial features + raw PCA spectral values — informationally distinct branches |
| **Multi-Seed Training** | All models run N seeds → mean ± std reported (reviewer-ready) |
| **Auto Table Generation** | `.tex` + `.txt` comparison tables generated automatically after training |
| **Feature Interpretability** | t-SNE maps, spectral gate profiles, CNN activation maps per dataset |
| **Error Isolation** | Failed models are skipped with a traceback; other models continue uninterrupted |
| **19 Baseline Models** | One `baseline.py` runner covers all comparison methods |
| **Batch Scripts** | Single-line commands to run one or all datasets |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/TensorGatedSSM.git
cd TensorGatedSSM

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### requirements.txt (minimum)

```
tensorflow>=2.10
numpy
scipy
scikit-learn
matplotlib
seaborn
pandas
keras-cv-attention-models
PyWavelets
```

> **GPU note:** TensorFlow GPU support requires CUDA 11.2+ and cuDNN 8.1+. See the [TensorFlow install guide](https://www.tensorflow.org/install/gpu).

---

## Dataset Setup

Place all `.mat` dataset files in a `datasets/` folder **one level above** this repo:

```
parent_dir/
├── datasets/
│   ├── WHU_Hi_LongKou.mat
│   ├── WHU_Hi_LongKou_gt
│   ├── WHU_Hi_HongHu.mat
│   ├── WHU_Hi_HongHu_gt
│   ├── WHU_Hi_HanChuan.mat
│   ├── WHU_Hi_HanChuan_gt
│   ├── QUH-Qingyun.mat  +  QUH-Qingyun_GT.mat
│   ├── QUH-Pingan.mat   +  QUH-Pingan_GT.mat
│   ├── QUH-Tangdaowan.mat + QUH-Tangdaowan_GT.mat
│   ├── fei2.mat + FeiC_gt.mat          (SDFC)
│   ├── Indian_pines_corrected.mat + Indian_pines_gt.mat
│   └── ... (other classic benchmarks)
└── TensorGatedSSM/  ← this repo
```

| Dataset suite | Source |
|---------------|--------|
| WHU-Hi (LK, HH, HC) | [rsidea.whu.edu.cn](http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm) |
| QUH (Qingyun, Pingan, Tangdaowan) | QUH dataset paper |
| SDFC (Fei2) | SDFC dataset |
| Classic (IP, PU, SA, KSC, BS, PC, UH, SLA) | [IEEE GRSS](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) |

---

## Quick Start

```bash
# Run TensorGatedSSM + CNN2D on LongKou, 1 seed (fast test)
python baseline.py --dataset LK --seeds 1 --models CNN2D TensorGatedSSM

# Full run: all 19 models, 5 seeds on HanChuan
python baseline.py --dataset HC

# Full pipeline on all 7 WHU/QUH datasets
bash run_all_datasets.sh
```

---

## Usage

### Single-Model Reference (`main.py`)

`main.py` is a self-contained single-model script (standalone, no `models/` imports needed). Good for quickly testing the TensorGatedSSM model in isolation.

```bash
# Edit HSID at the top of main.py, then:
python main.py
```

It runs training → feature visualisations → multi-seed comparison → ablation → prediction map.

---

### Unified Baseline Runner (`baseline.py`)

The main entry point for paper experiments. Runs all (or selected) models with multi-seed training and generates tables automatically.

```
python baseline.py [OPTIONS]

Options:
  --dataset   / -d    Dataset name (default: LK)
                      Choices: LK HH HC Qingyun Pingan Tangdaowan SDFC
                               IP SA PU KSC BS PC UH SLA
  --window_size / -ws Patch size (default: 14)
  --dl_method  / -dlm Dimensionality reduction: PCA | iPCA (default: PCA)
  --test_ratio  / -te Test split ratio (default: 0.90)
  --train_ratio / -tr Train split ratio (default: 0.50)
  --val_ratio   / -vr Validation split ratio (default: 0.50)
  --num_bands   / -k  PCA components (default: 15)
  --epochs      / -e  Training epochs (default: 50)
  --batch_size  / -bs Batch size (default: 56)
  --seeds       / -s  Number of random seeds for mean±std (default: 5)
  --models      / -m  Subset of models to run, or 'all' (default: all)
  --gpu_id      / -gid GPU device index (default: 0)
  --skip_table        Skip .tex / .txt table generation
  --skip_viz          Skip t-SNE / gate / CNN-map visualisations
  --run_ablation      Also run TensorGatedSSM component ablation
```

**Examples:**

```bash
# All models, 5 seeds, dataset LK
python baseline.py --dataset LK

# Only SSMamba + TensorGatedSSM, 3 seeds
python baseline.py --dataset HC --seeds 3 --models SSMamba TensorGatedSSM

# Full run with ablation and specific GPU
python baseline.py --dataset HH --run_ablation --gpu_id 1

# Skip heavy visualisations for a quick results check
python baseline.py --dataset Qingyun --skip_viz
```

---

### Batch Scripts

```bash
# Run all models on ONE dataset (passes extra args to baseline.py)
bash run_single_dataset.sh LK
bash run_single_dataset.sh HC --seeds 3 --models CNN2D TensorGatedSSM

# Run all models on ALL 7 WHU/QUH datasets sequentially
bash run_all_datasets.sh
bash run_all_datasets.sh --seeds 3          # 3 seeds across all datasets
bash run_all_datasets.sh --skip_viz         # skip visualisations
```

Edit the `DATASETS` array at the top of `run_all_datasets.sh` to change which datasets are processed.

---

### Table Generation

Tables are generated automatically after training. To regenerate from existing JSON files:

```bash
python utils/generate_table.py \
    --dataset LK \
    --results_dir results/baseline_results/LK \
    --output_dir tables/
```

This reads every `*_results.json` file in the results directory (one per model) and produces:

| File | Format |
|------|--------|
| `LK_comparison_table.tex` | LaTeX table (booktabs, bold best per row, `\pm` notation) |
| `LK_comparison_table.txt` | Plain-text ASCII table |

The table structure:

```
+---------------------+-------------------+-------------------+-------------------+
| Class               | CNN2D             | SSMamba           | TensorGatedSSM    |
+=====================+===================+===================+===================+
| Corn                | 85.30 ± 1.20     | 89.10 ± 0.85     | 92.30 ± 0.70     |
| Cotton              | 82.10 ± 2.10     | 87.40 ± 1.10     | 91.50 ± 0.60     |
| ...                 |                  |                  |                  |
+---------------------+-------------------+-------------------+-------------------+
| OA (%)              | 87.50 ± 1.00     | 91.20 ± 0.75     | 93.10 ± 0.33     |
| AA (%)              | 85.20 ± 1.50     | 89.80 ± 0.90     | 92.00 ± 0.41     |
| Kappa (%)           | 86.10 ± 1.20     | 90.50 ± 0.80     | 92.53 ± 0.37     |
+---------------------+-------------------+-------------------+-------------------+
| Parameters          | 250.4K           | 890.1K           | 456.8K           |
+=====================+===================+===================+===================+
```

---

### Ablation Study (Notebook)

The ablation study is kept in the Jupyter notebook to avoid interfering with the main comparison pipeline.

```bash
jupyter notebook TensorGatedSSM_reference.ipynb
```

The notebook covers:
1. Imports & hyperparameters
2. Data loading & preprocessing
3. Single-seed training
4. **Ablation study** — 4 variants × 5 seeds

| Variant | Spectral Gate | Feature Enhancement |
|---------|:---:|:---:|
| Full TensorGated-SSM | ✓ | ✓ |
| w/o Spectral Gate | ✗ | ✓ |
| w/o Feature Enhancement | ✓ | ✗ |
| Base (MHA only) | ✗ | ✗ |

5. Multi-seed comparison (generates JSON for table generator)
6. Feature interpretability visualisations
7. Full-scene prediction map
8. Comparison table generation

---

## Outputs

After running `baseline.py` or the shell scripts, all outputs land in:

```
results/baseline_results/{DATASET}/
├── {MODEL}_results.json              ← per-class per-seed metrics (feeds table generator)
├── {DATASET}_{...}_{MODEL}_seed{N}.csv  ← per-seed classification report
├── {DATASET}_comparison_table.tex    ← LaTeX table (all models vs all classes)
├── {DATASET}_comparison_table.txt    ← Plain-text table
├── {DATASET}_AccLoss.png             ← Convergence curves (all models)
├── {DATASET}_{...}_PredMap_{MODEL}.png  ← Full-scene prediction map
├── {DATASET}_{...}_PredMap_{MODEL}_Legend.png
├── {DATASET}_tsne_features.png       ← t-SNE of TensorGatedSSM learned features
├── {DATASET}_spectral_gate_weights.png  ← Spectral gate activation profile
└── {DATASET}_cnn_activation_map.png  ← Per-class CNN spatial activation
```

---

## Supported Datasets

| ID | Dataset | Classes | Bands | Source |
|----|---------|:-------:|:-----:|--------|
| `LK` | WHU-Hi LongKou | 9 | 270 | WHU |
| `HH` | WHU-Hi HongHu | 22 | 270 | WHU |
| `HC` | WHU-Hi HanChuan | 16 | 274 | WHU |
| `Qingyun` | QUH-Qingyun | 6 | 250 | QUH |
| `Pingan` | QUH-Pingan | 10 | 250 | QUH |
| `Tangdaowan` | QUH-Tangdaowan | 18 | 250 | QUH |
| `SDFC` | SDFC (Fei2) | 20 | 198 | SDFC |
| `IP` | Indian Pines | 16 | 200 | AVIRIS |
| `PU` | Pavia University | 9 | 103 | ROSIS |
| `SA` | Salinas | 16 | 204 | AVIRIS |
| `KSC` | Kennedy Space Center | 13 | 176 | AVIRIS |
| `BS` | Botswana | 14 | 145 | Hyperion |
| `PC` | Pavia Centre | 9 | 102 | ROSIS |
| `UH` | University of Houston | 15 | 144 | ITRES |
| `SLA` | Salinas-A | 6 | 204 | AVIRIS |

---

## Supported Baseline Models

| Model | Category | Input |
|-------|----------|-------|
| CNN2D, IN2D, Exception2D, Xtreme\_Exception | 2D CNN | 5D |
| CNN3D, IN3D, HybCNN, HybIN, AttentionGCN | 3D / Hybrid CNN | 5D |
| SSFormer, WaveFormer, PyFormer, Hybrid\_ViT | Transformer | 4D |
| SSMamba | State-Space | 4D |
| WaveMamba | Wavelet + SSM | 4D |
| DSST (DiffFormer) | Differential Attention | 4D |
| DiMAMamba | Differential Memory Attention | 4D |
| MorpMamba | Morphological + SSM | 4D |
| **TensorGatedSSM** | **Spectral-Gated SSM (ours)** | **4D** |

---

## Project Structure

```
TensorGatedSSM/
│
├── baseline.py                    Unified baseline runner (all 19 models)
├── main.py                        Standalone single-model script (reference)
├── TensorGatedSSM_reference.ipynb Notebook: training + ablation + visualisation
├── run_single_dataset.sh          Run all models on one dataset
├── run_all_datasets.sh            Run all models on all datasets
│
├── models/
│   ├── tgssm.py                   TensorGatedSSM model class
│   ├── cnn.py                     CNN2D, IN2D, CNN3D, IN3D, Hybrids, AttentionGCN
│   ├── vit.py                     Hybrid_ViT, SSFormer, WaveFormer, PyFormer
│   ├── SSMamba.py                 Spatial-Spectral Mamba
│   ├── WaveMamba.py               Wavelet Mamba
│   ├── DiffFormer.py              Differential Attention Transformer (DSST)
│   ├── DiMA.py                    Differential Memory Attention Mamba
│   └── MorpMamba.py               Morphological Mamba
│
└── utils/
    ├── data_loader.py             LoadHSIData (15 datasets) + preprocessing
    ├── eval.py                    ClassificationReports, GT_Plot, save_prediction_map
    ├── generate_table.py          .tex + .txt comparison table generator
    └── visualize_features.py      t-SNE, spectral gate, CNN activation maps
```

---

## Reproducibility

All experiments use stratified splits and fixed random seeds. The default seed list is `[345, 445, 545, 645, 745]` (configurable via `--seeds`).

To exactly reproduce a single run:

```bash
python baseline.py \
    --dataset LK \
    --seeds 5 \
    --window_size 14 \
    --num_bands 15 \
    --test_ratio 0.90 \
    --epochs 50 \
    --batch_size 56
```

Results are reported as **mean ± std** across all seeds, addressing Scientific Reports reviewer requirement for statistical reliability.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{TariqTensorGatedSSM2025,
  title   = {TensorGated State-Space Model for Spatial Spectral Representation
             Learning in Hyperspectral Image Classification},
  author  = {Tariq, Hassan and ...},
  journal = {Scientific Reports},
  year    = {2025},
  note    = {Under Review}
}
```

---

## Acknowledgements

- **WHU-Hi datasets**: [rsidea.whu.edu.cn](http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm)
- **QUH datasets**: QUH dataset paper authors
- **Baseline model implementations** adapted from prior work in the `baseline_models_for_hsic` repository

---

<div align="center">

Made with care for reproducible remote sensing research.

</div>

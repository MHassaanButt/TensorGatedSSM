# TensorGated-SSM

Reproducibility code and results for the paper *TensorGated State-Space Model for
Spatial–Spectral Representation Learning in Hyperspectral Image Classification*
(under review, *Scientific Reports*).

This repository contains a **self-contained Jupyter notebook** with the complete model,
training and evaluation code, together with the **numerical results** reported in the paper.

## Model

TensorGated-SSM is a compact (~62K-parameter) hyperspectral image classifier:

1. **Dual tokenisation** — spatial tokens from a small convolutional stem and spectral
   tokens from the raw PCA cube, so the two streams carry different information.
2. **Global patch-context feature-enhancement gate** — a context vector pooled over the
   whole patch re-weights both streams channel-wise.
3. **Spectral–spatial cross-attention** — a single block where spatial tokens query
   spectral tokens.
4. **Linear state-space scan** — summarises the fused sequence into one state at O(N) cost,
   in place of a second quadratic attention layer.

## Contents

```
TensorGatedSSM_reproduce.ipynb   Self-contained notebook (model + data + training + evaluation)
results/
├── baseline_results/{DS}/       Per-seed metrics, comparison tables, prediction maps, t-SNE features
├── ablation_results/{DS}/       Component ablation (feature enhancement, tokenisation, aggregator)
└── sensitivity/                 Patch-size and label-ratio sweeps
```

Datasets (`{DS}`): `HC` (WHU-Hi HanChuan), `HH` (WHU-Hi HongHu), `Qingyun` and
`Tangdaowan` (QUH), `SA` (Salinas), `PU` (Pavia University).

## Usage

The notebook has no dependencies on other files in this repository.

```bash
pip install tensorflow numpy scipy scikit-learn matplotlib
jupyter notebook TensorGatedSSM_reproduce.ipynb
```

Set the dataset at the top of the notebook and run all cells. The datasets are public;
download links are given in the paper's Data Availability statement.

## Citation

```bibtex
@article{tensorgatedssm,
  title   = {TensorGated State-Space Model for Spatial--Spectral Representation
             Learning in Hyperspectral Image Classification},
  journal = {Scientific Reports},
  year    = {2025},
  note    = {Under review}
}
```

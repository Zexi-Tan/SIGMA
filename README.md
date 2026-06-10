# SIGMA: Semantic-Invariant Geometry-Aware Masked Admission for MTS Clustering

SIGMA is a PyTorch implementation of **Prune but Preserve: Decoupling Redundancy Suppression and Semantic Preservation for Efficient Time Series Clustering**. The project targets unsupervised multivariate time series clustering, where long routine intervals can dominate representation learning and short discriminative fragments are easy to dilute. SIGMA addresses this problem through confidence-calibrated patch admission, multi-endogenous view generation, reconstruction-based semantic preservation, and prototype-guided contrastive refinement.

## Highlights

- **Source-side patch admission**: selects compact high-confidence temporal patches before contextual encoding, reducing redundant encoder-side interaction.
- **Multi-endogenous views**: constructs complementary retained-token views to compensate for partial observation after pruning.
- **Semantic restoration route**: reconstructs the original sequence from admitted content, preserving temporal context under compression.
- **Prototype-guided contrastive route**: refreshes pseudo labels and aligns fused representations with latent cluster prototypes.
- **Scalable MTS clustering**: supports compact embeddings, retained-ratio diagnostics, noise robustness tests, and efficient mini-batch training.

## Repository Structure

```text
SIGMA-main/
├── Batch_Run.py          # Main training and evaluation script
├── mymodal.py            # SIGMA model definition
├── cov.py                # Transformer encoder and positional encoding modules
├── pooling.py            # View-level pooling operation
├── Contrastive_loss.py   # Prototype-style contrastive loss
├── utils.py              # Clustering metrics and evaluation utilities
├── Cluster_gpu.py        # Optional GPU k-means wrapper
└── ICDE_supplemental_material.pdf     # Supplementary experimental details
```

## Requirements

The code was developed with PyTorch and common scientific Python packages.

```bash
conda create -n sigma python=3.10 -y
conda activate sigma

pip install numpy pandas scipy scikit-learn matplotlib tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

The current code imports `kmeans_gpu` through `Cluster_gpu.py`. If your environment reports `ModuleNotFoundError: No module named 'kmeans_gpu'`, add the corresponding `kmeans_gpu.py` file to the project root or install the GPU k-means package used in your experimental environment. The main evaluation path in `Batch_Run.py` uses scikit-learn k-means through `utils.py`, but the import dependency still needs to be resolvable.

## Data Format

Place each dataset under the following directory structure:

```text
New_Data/
└── <DatasetName>/
    └── <DatasetName>.npy
```

Each `.npy` file should be saved as a Python dictionary with the following keys:

```python
{
    "train_X": train_X,   # shape: [N_train, T, D]
    "train_Y": train_Y,   # shape: [N_train]
    "test_X": test_X,     # shape: [N_test, T, D]
    "test_Y": test_Y      # shape: [N_test]
}
```

The training script concatenates train/test splits for unsupervised clustering. Labels are used only for evaluation.

## Quick Start

Run SIGMA on one dataset:

```bash
python Batch_Run.py \
  --dataname BasicMotions \
  --epoch_num 200 \
  --batch_size 64 \
  --patch_size 4 \
  --target_keep_rate 0.5 \
  --num_view 2 \
  --output_dims 64 \
  --hidden_dims 32
```

Run the default five-seed setting:

```bash
python Batch_Run.py \
  --dataname BasicMotions \
  --seeds 2022 2023 2024 2025 2026
```

Run a single seed for debugging:

```bash
python Batch_Run.py \
  --dataname BasicMotions \
  --seeds 2026 \
  --epoch_num 5
```

## Main Arguments

| Argument | Default | Description |
|---|---:|---|
| `--dataname` | `StandWalkJump` | Dataset name under `New_Data/<DatasetName>/` |
| `--seeds` | `2022 2023 2024 2025 2026` | Random seeds for repeated runs |
| `--use_mask` | `1` | Use SIGMA masked admission if `1`; use full-token processing if `0` |
| `--lr` | `0.001` | Adam learning rate |
| `--epoch_num` | `200` | Number of training epochs |
| `--batch_size` | `64` | Mini-batch size |
| `--patch_size` | `4` | Temporal patch length |
| `--target_keep_rate` | `0.5` | Retained patch ratio |
| `--num_view` | `2` | Number of endogenous views |
| `--output_dims` | `64` | Embedding dimension |
| `--hidden_dims` | `32` | Hidden size of the Transformer encoder |
| `--temperature` | `0.1` | Temperature for contrastive loss |
| `--alpha` | `1.0` | Reconstruction loss weight |
| `--gama` | `1.0` | Contrastive loss weight |
| `--beta` | `0.1` | Inter-view diversity loss weight |
| `--noise_std` | `0.0` | Gaussian noise standard deviation for robustness testing |

## Training Pipeline

`Batch_Run.py` follows the pipeline below:

1. Load a dataset from `New_Data/<dataname>/<dataname>.npy`.
2. Concatenate train/test splits and standardize variables.
3. Initialize SIGMA and obtain initial pseudo labels through k-means over fused embeddings.
4. Train with a reconstruction warm-up stage.
5. Periodically refresh pseudo labels and align cluster centers across refreshes.
6. Optimize the routed objective with reconstruction, contrastive, and inter-view diversity terms.
7. Report ACC, F1, NMI, and ARI at the final epoch for each seed.

## Method Overview

SIGMA receives an unlabeled MTS sample `x` with shape `[T, D]` and partitions it into temporal patches. A patch scorer estimates confidence for each patch. During training, stochastic perturbation encourages view-specific patch admission. During evaluation, deterministic top-k admission keeps the highest-confidence patches according to the target keep rate.

The admitted patches are encoded by multiple Transformer-based endogenous views. Retained tokens are projected into view-level representations, reconstructed back to the original sequence grid, and fused for clustering. The contrastive loss uses pseudo labels from periodically refreshed k-means assignments, which provides cluster-oriented guidance without label supervision.

## Expected Output

The script prints per-epoch training statistics and final clustering results, for example:

```text
Seed-2026 Epoch-200 | Loss: ... | Drop: 50.0% | Time: ...s
ACC 0.xxxx | F1 0.xxxx | ARI 0.xxxx | NMI 0.xxxx

Final Results for '<DatasetName>' (Mask: ON)
ACC:  mean ± std
F1:   mean ± std
NMI:  mean ± std
ARI:  mean ± std
Avg Optimal Drop Rate: ...%
Avg Time to Convergence: ... seconds
Avg Total Train Time: ... seconds
```

## Reproducing the Paper Setting

The main paper uses the following default configuration for SIGMA:

```bash
python Batch_Run.py \
  --dataname <DatasetName> \
  --seeds 2022 2023 2024 2025 2026 \
  --epoch_num 200 \
  --patch_size 4 \
  --target_keep_rate 0.5 \
  --num_view 2 \
  --hidden_dims 32 \
  --output_dims 64 \
  --batch_size 64
```

The paper evaluates SIGMA on 30 UEA multivariate time series benchmarks and real-world case-study datasets. Reported metrics include ACC, F1, NMI, ARI, runtime, GPU memory, noise robustness, embedding compactness, and retained-ratio diagnostics.

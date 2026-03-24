# TreeFinder-ATML-InfoBatch: Implementation Details

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Configuration](#3-configuration)
4. [Data Pipeline](#4-data-pipeline)
5. [Model: Mask2Former](#5-model-mask2former)
6. [Training Loop](#6-training-loop)
7. [InfoBatch: Lossless Training Acceleration](#7-infobatch-lossless-training-acceleration)
8. [Evaluation & Metrics](#8-evaluation--metrics)
9. [Dataset Structure](#9-dataset-structure)
10. [Output Artifacts](#10-output-artifacts)

---

## 1. Project Overview

**TreeFinder** is a PyTorch-based benchmarking pipeline for training and evaluating semantic segmentation models on **dead-tree mortality monitoring using high-resolution aerial imagery**. The dataset is from the NeurIPS 2025 datasets track and is hosted on Kaggle.

### Key Goals

- Train and evaluate Mask2Former for binary pixel-wise dead-tree segmentation
- Support multiple generalization scenarios (random, geographic, climate, species-based splits)
- Integrate InfoBatch for lossless training acceleration via intelligent dynamic data pruning
- Provide comprehensive evaluation with ROC, PR, calibration, and confusion matrix visualizations

### Core Technical Choices

| Concern | Choice | Rationale |
|---|---|---|
| Input channels | RGB + NIR + NDVI (5 channels) | NDVI captures vegetation health signal |
| Segmentation type | Binary (live/dead) | Dead tree = class 1 |
| Loss | BCE + Dice (weighted sum) | Handles class imbalance (dead trees are rare) |
| Training acceleration | InfoBatch pruning | ~2x speedup with <1% accuracy loss |
| Config management | YAML (`debug.yaml`) | Single reproducible experiment config |

---

## 2. Project Structure

```
treefinder-atml-infobatch/
├── main.py                    # Entry point: orchestrates training + evaluation
├── upd_info.py                # InfoBatch dataset wrapper
├── requirements.txt           # Python dependencies
├── readme.md                  # Basic project readme
│
├── configs/
│   └── debug.yaml             # Experiment configuration
│
├── data_loader/               # Data loading and splitting
│   ├── __init__.py            # Factory: get_dataloader()
│   ├── utils.py               # load_tile(), augment_tile()
│   ├── random_split.py        # RandomSplitDataset
│   ├── by_state_split.py      # StateSplitDataset
│   ├── by_climate_split.py    # ClimateSplitDataset
│   └── by_tree_split.py       # TreeSplitDataset
│
├── models/                    # Model architectures
│   ├── __init__.py            # Factory: get_model()
│   └── mask2former.py         # Mask2Former (Swin-T backbone)
│
├── exps/                      # Training and evaluation loops
│   ├── train.py               # train_model()
│   └── evaluate.py            # evaluate_model()
│
├── utils/                     # Shared utilities
│   └── tools.py               # Arg parsing, config loading, logging, seeding
│
├── splits/                    # Saved split JSON files (auto-generated)
│   └── random_seed2025.json
│
├── results/                   # Experiment outputs (auto-generated)
├── checkpoints/               # Saved model weights (auto-generated)
└── logs/                      # Training logs (auto-generated)
```

---

## 3. Configuration

The experiment is configured via `configs/debug.yaml`. All parameters are documented below.

### `experiment`

```yaml
experiment:
  id: 1                        # Unique identifier used in output paths
  seed: 2025                   # Base random seed
  gpu_id: 0                    # CUDA device
  num_workers: 16              # DataLoader workers
```

### `data`

```yaml
data:
  root_dir: /path/to/dataset
  dataset_dir: tiles224_v3
  dataset_info: tile_info224_v3.csv
  no_data_value: 255
  normalize: true
  train_fraction: 1.0          # Subsample training set (0.0–1.0)
  num_train_splits: 1          # Repeat training with different subsample seeds
  augmentation:
    random_flip: true
    rotation:
      type: "90"               # Discrete 0/90/180/270 degree rotations
  split:
    method: random             # Options: random, by_state, by_climate, by_tree_type
    pos_threshold: 100         # Min dead pixels to count as a positive tile
    pos_frac: 0.5              # Target fraction of positive samples
    train_ratio: 0.8
    train_val_ratio: 0.9
    shuffle_by_tile: true
```

### `model`

```yaml
model:
  name: mask2former
  in_channels: 5
  num_classes: 1
  image_size: 224
  mask2former_weights: facebook/mask2former-swin-tiny-ade-semantic
```

### `training`

```yaml
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  optimizer:
    type: AdamW
    weight_decay: 0.01
  criterion:
    type: BCEWithLogitsLoss     # BCEWithLogitsLoss | FocalLoss
    w_pos: 2.0                  # Positive class weight (0 = no weighting)
    alpha: 0.25                 # Focal loss alpha
    gamma: 2.0                  # Focal loss gamma
  w_dice: 0.5                   # Dice loss weight (0 = no Dice loss)
  scheduler:
    type: CosineAnnealingWarmRestarts
    gamma: 0.95
  early_stopping:
    enabled: true
    monitor: val_loss
    mode: min
    patience: 15
```

### `infobatch`

```yaml
infobatch:
  enabled: true
  prune_ratio: 0.5             # Fraction of well-learned samples to prune
  delta: 0.875                 # Active pruning for first delta * epochs epochs
```

### `output` / `logging`

```yaml
output:
  results_dir: results/
  checkpoint_dir: checkpoints/

logging:
  log_dir: logs/
  log_interval: 50             # Log every N batches
```

---

## 4. Data Pipeline

### 4.1 Entry Point: `get_dataloader()`

Located in `data_loader/__init__.py`. Creates train/val/test DataLoaders from config.

```
cfg, train_fraction_seed
       ↓
Instantiate split dataset (Random/State/Climate/Tree)
       ↓
If train_fraction < 1.0: subsample training indices (seeded)
       ↓
If infobatch.enabled: wrap with InfoBatch(dataset, num_epochs, prune_ratio, delta)
       ↓
Create DataLoader (train: IBSampler if InfoBatch, else shuffle=True)
       ↓
Return train_loader, val_loader, test_loader, train_dataset
```

### 4.2 Tile Loading: `load_tile(path)`

Each tile is a 5-band GeoTIFF:

| Band | Content | Normalization |
|---|---|---|
| 1 | Red | [0,255] → [0,1] |
| 2 | Green | [0,255] → [0,1] |
| 3 | Blue | [0,255] → [0,1] |
| 4 | NIR | [0,255] → [0,1] |
| 5 | Label (0=live, 1=dead, 255=no-data) | unchanged |

**NDVI** is computed and appended as a 5th image channel:

```
NDVI = (NIR - RED) / (NIR + RED + eps)     # range [-1, 1]
NDVI_normalized = (NDVI + 1) / 2           # range [0, 1]
```

**No-data mask** is derived from pixels where any channel equals 255. These pixels are excluded from all loss computations.

**Output tensors:**
- `img`: `[5, H, W]` float32 — 5 normalized channels (RGB + NIR + NDVI)
- `label`: `[H, W]` long — binary segmentation ground truth
- `no_data_mask`: `[H, W]` bool — True for invalid pixels

### 4.3 Data Augmentation: `augment_tile()`

Applied **only during training**, consistently across image, label, and mask:

- **Random horizontal flip** (50% probability)
- **Random vertical flip** (50% probability)
- **Random 90° rotation** (uniform over 0°, 90°, 180°, 270°)

### 4.4 Split Methods

#### Random Split (`random_split.py`)

- Shuffles all tiles and splits into train/val/test by ratio (default 80-10-10)
- Supports class balancing via `pos_threshold` (min dead pixels) and `pos_frac` (target ratio of positive tiles)
- Frozen split serialized to `splits/random_seed{seed}.json`

#### By State (`by_state_split.py`)

- Reads US state labels from the tile metadata CSV
- Assigns tiles to train/val/test based on `train_states`, `val_states`, `test_states`
- If `test_exclude_train=true`, test set includes all tiles not in training states

#### By Climate (`by_climate_split.py`)

- Reads Köppen climate classification from `*_aux.csv`
- Uses climate zone codes to define train/val/test splits
- Enables testing cross-climate generalization (e.g., train in Mediterranean, test in Continental)

#### By Tree Type (`by_tree_split.py`)

- Reads dominant tree species from `*_aux.csv`
- `train_exclude_test=true` ensures no training tiles contain test species
- Tests species-specific model generalization

### 4.5 Sample Dictionary Format

Each `__getitem__` call returns:

```python
{
    'image':        Tensor[5, H, W],    # RGB + NIR + NDVI, normalized
    'label':        Tensor[H, W],       # long, 0=live / 1=dead
    'no_data_mask': Tensor[H, W],       # bool, True = invalid pixel
    'cls_label':    Tensor[],           # scalar bool, True if tile has dead trees
}
```

When wrapped with InfoBatch, `__getitem__` also returns the sample's global index for weight tracking.

---

## 5. Model: Mask2Former

Located in `models/mask2former.py`.

Mask2Former is a set-prediction segmentation model that uses masked cross-attention to condition each query on a predicted spatial region, enabling precise binary segmentation of dead trees.

### Architecture

- **Backbone:** Swin-T (`facebook/mask2former-swin-tiny-ade-semantic`, loaded from HuggingFace)
- **Input:** Uses only the first 3 channels (RGB) from the 5-channel input tensor `[B, 5, H, W]`
- **Query-based prediction:** The transformer decoder produces `Q` object queries per image
- **Outputs:**
  - Class logits: `[B, Q, C]`
  - Binary mask logits: `[B, Q, H, W]`

### Segmentation Head

The final segmentation map is formed by fusing class probabilities with mask predictions via einsum:

```
seg_logits[b, h, w] = Σ_q class_probs[b, q] * mask_logits[b, q, h, w]
```

This collapses the query dimension, producing a single-channel logit map `[B, 1, H, W]` used for binary dead-tree prediction.

### Config

```yaml
model:
  name: mask2former
  mask2former_weights: facebook/mask2former-swin-tiny-ade-semantic
  in_channels: 5
  num_classes: 1
  image_size: 224
```

---

## 6. Training Loop

Located in `exps/train.py`.

### Full Epoch Flow

```
For each epoch:
  model.train()
  For each batch from train_loader:
    1. Unpack: image, label, no_data_mask [, indices if InfoBatch]
    2. Forward:  logits = model(image)
    3. Compute per-pixel loss (unreduced):
         raw_loss = criterion(logits, label.float())     # [B, 1, H, W]
    4. Apply no-data mask:
         valid_mask = ~no_data_mask.unsqueeze(1)         # [B, 1, H, W]
    5. If InfoBatch:
         per_img_loss = sum(raw_loss * valid_mask, dims=[1,2,3]) / valid_pixels
         bce_loss = train_dataset.update(per_img_loss, indices)
       Else:
         bce_loss = sum(raw_loss * valid_mask) / sum(valid_mask)
    6. If w_dice > 0:
         dice_loss = DiceLoss(logits, label, valid_mask)
         total_loss = bce_loss + w_dice * dice_loss
       Else:
         total_loss = bce_loss
    7. total_loss.backward()
    8. optimizer.step()

  Validate on val_loader (no grad):
    Compute val_loss → if improved: save checkpoint

  Scheduler step
  If early stopping: check patience → break if exceeded
```

### Loss Functions

**BCEWithLogitsLoss:** Standard binary cross-entropy with optional positive class weight.

```
loss = -[w_pos * y * log(σ(z)) + (1 - y) * log(1 - σ(z))]
```

**FocalLoss:** Downweights easy negatives to focus learning on hard examples.

```
p_t = σ(z) if y=1 else 1 - σ(z)
loss = -α * (1 - p_t)^γ * log(p_t)
```

**DiceLoss:** Directly optimizes a soft IoU-like objective; robust to class imbalance.

```
Dice = (2 * Σ(p * y)) / (Σ(p²) + Σ(y²) + ε)
loss = 1 - Dice
```

**Combined:**

```
total_loss = bce_loss + w_dice * dice_loss
```

### Optimizer & Scheduler

- **Optimizer:** AdamW (weight decay as L2 regularization)
- **Schedulers:** ExponentialLR, StepLR, or CosineAnnealingWarmRestarts

### Checkpointing

Best model weights (by validation loss) are saved to:

```
checkpoints/{exp_name}/{exp_name}_best.pth
```

---

## 7. InfoBatch: Lossless Training Acceleration

### Background

InfoBatch ([arXiv:2303.04947](https://arxiv.org/abs/2303.04947)) accelerates training by dynamically pruning well-learned ("easy") samples while **rescaling gradients** of remaining samples to preserve the expected gradient update. This makes pruning statistically unbiased and therefore lossless.

### Implementation: `upd_info.py`

**Initialization:**

```python
InfoBatch(dataset, num_epochs, prune_ratio=0.5, delta=0.875)
```

| Parameter | Description |
|---|---|
| `dataset` | Wrapped PyTorch Dataset |
| `num_epochs` | Total training epochs (used to set pruning window) |
| `prune_ratio` | Fraction of easy samples pruned each step |
| `delta` | Pruning is active for first `delta * num_epochs` epochs |

Internally maintains:
- `scores[N]`: Running loss score per sample (initialized to 3.0)
- `weights[N]`: Gradient rescaling factor per sample (initialized to 1.0)
- `active_indices`: Subset of non-pruned sample indices for the current epoch

**Per-batch update:**

```python
weighted_loss = train_dataset.update(per_img_loss, batch_indices)
```

1. Stores `per_img_loss` into `scores[batch_indices]`
2. Multiplies each sample's loss by `weights[batch_indices]`
3. Returns the mean of the reweighted losses (used as `bce_loss` in the total loss)

**Pruning (at epoch boundary):**

1. Compute `mean_score = mean(scores)` across active samples
2. Identify easy samples: `scores[i] < mean_score`
3. Randomly keep `keep_ratio = 1.0 - prune_ratio` fraction of easy samples
4. Set `weights[kept_easy_samples] = 1.0 / keep_ratio` (gradient upscaling)
5. Hard samples (score ≥ mean) are always kept with weight = 1.0
6. Return the new active index set for the next epoch

**IBSampler:**

A custom PyTorch sampler that iterates only over active (non-pruned) indices. The DataLoader is patched to expose batch indices so `update()` can track which samples were used.

### Effect on Training

- Easy samples (well below average loss) are gradually pruned from the training loop
- Gradient upscaling on retained easy samples ensures unbiased gradient estimation
- In tree mortality data: many tiles have zero dead trees (easy negatives) — these are pruned most aggressively
- Expected result: ~1.5–2× fewer training iterations per epoch with minimal accuracy impact

---

## 8. Evaluation & Metrics

Located in `exps/evaluate.py`.

### Metrics Computed

After loading the best checkpoint:

```
For each batch in test_loader:
  logits = model(image)
  probs = sigmoid(logits)
  preds = (probs > 0.5).long()

Aggregate across all valid (non-masked) pixels:
  TP, FP, FN, TN per class
  Precision, Recall, F1, IoU per class
  Macro-averaged Precision, Recall, F1, IoU
  Overall pixel accuracy
```

### Evaluation Plots

All saved to `results/{exp_name}/`:

| File | Description |
|---|---|
| `roc_curve.png` | FPR vs TPR with AUC score |
| `pr_curve.png` | Precision vs Recall curve with AP |
| `confusion_matrix.png` | Normalized heatmap (Live / Dead) |
| `threshold_metrics.png` | Precision, Recall, F1 across thresholds 0.0–1.0 |
| `calibration_curve.png` | Reliability diagram: mean predicted probability vs fraction of positives |

### Metric Output

Saved to `results/{exp_name}/{exp_name}_metrics.yaml`:

```yaml
class_0:
  precision: 0.94
  recall: 0.91
  f1: 0.92
  iou: 0.86
class_1:
  precision: 0.78
  recall: 0.82
  f1: 0.80
  iou: 0.67
macro:
  precision: 0.86
  recall: 0.87
  f1: 0.86
  iou: 0.77
overall_accuracy: 0.91
```

---

## 9. Dataset Structure

### Expected Layout

```
root_dir/
  tiles224_v3/                    # GeoTIFF tiles (one per sample)
    tile_00001.tif
    tile_00002.tif
    ...
  tile_info224_v3.csv             # Primary metadata
  tile_info224_v3_aux.csv         # Auxiliary metadata (climate, tree type)
```

### Metadata CSVs

**`tile_info224_v3.csv`** (primary):

| Column | Description |
|---|---|
| FileName | Tile filename (e.g., `tile_00001.tif`) |
| State | US state abbreviation (e.g., `CA`) |
| Region | Geographic region |
| ImageRawPath | Original source path |
| LabelSize | Number of dead-tree pixels in the tile |

**`tile_info224_v3_aux.csv`** (auxiliary, for climate and tree-type splits):

| Column | Description |
|---|---|
| FileName | Tile filename (matches primary CSV) |
| ClimateType | Köppen climate code (e.g., `Csa`, `Dfb`) |
| TreeTypes | Dominant tree species |

### GeoTIFF Band Layout

| Band | Content | Value Range |
|---|---|---|
| 1 | Red | 0–255 |
| 2 | Green | 0–255 |
| 3 | Blue | 0–255 |
| 4 | NIR | 0–255 |
| 5 | Label | 0 = live, 1 = dead, 255 = no-data |

---

## 10. Output Artifacts

```
results/
  {exp_name}/
    {exp_name}_metrics.yaml         # Final test metrics (per-class + macro)
    {exp_name}_metrics.csv          # Per-epoch training metrics
    {exp_name}_loss_curve.png       # Train vs validation loss over epochs
    roc_curve.png
    pr_curve.png
    confusion_matrix.png
    threshold_metrics.png
    calibration_curve.png

checkpoints/
  {exp_name}/
    {exp_name}_best.pth             # Best model weights (by val loss)

logs/
  {exp_name}.log                    # Full training + evaluation log

splits/
  random_seed{seed}.json            # Frozen random split (reproducibility)
  trainfrac_{frac}_seed{seed}.json  # Subsampled training indices (if train_fraction < 1.0)
```

---

## Full Training Pipeline Diagram

```
main.py
  │
  ├─ load_config(configs/debug.yaml)
  ├─ set_seed(base_seed)
  │
  └─ for run in range(num_train_splits):
       │
       ├─ set_seed(base_seed + run)
       │
       ├─ get_dataloader(cfg)
       │    ├─ Split dataset (Random / State / Climate / Tree)
       │    ├─ Subsample train_fraction (optional, seeded)
       │    ├─ Wrap with InfoBatch (optional)
       │    └─ Create train / val / test DataLoaders
       │
       ├─ get_model(cfg['model'])
       │    └─ Mask2Former (Swin-T backbone)
       │
       ├─ train_model(model, train_loader, val_loader, train_dataset, cfg)
       │    └─ AdamW + scheduler + early stopping
       │       BCE + Dice loss + no-data masking
       │       InfoBatch.update() per batch (if enabled)
       │       Save best checkpoint
       │
       └─ evaluate_model(model, test_loader, cfg)
            └─ Load best checkpoint
               Compute pixel-wise metrics
               Generate 5 evaluation plots
               Save metrics YAML
```

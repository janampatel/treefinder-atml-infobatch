# Implementation Plan: Minority-Class Protection for InfoBatch

## Background

Experimental results show that InfoBatch significantly degrades performance on the minority class (dead trees) as pruning ratio increases:

| Configuration     | F1 Dead Tree | Recall Dead Tree | IoU Dead Tree | Train Time |
|-------------------|-------------|-----------------|---------------|------------|
| No InfoBatch      | 47.2        | 32.1            | 30.9          | 2.83h      |
| + InfoBatch 10%   | 19.1        | 11.1            | 10.6          | 2.77h      |
| + InfoBatch 30%   | 41.7        | 27.8            | 26.3          | 2.74h      |
| + InfoBatch 50%   | 6.0         | 3.3             | 3.1           | 2.67h      |

### Root Cause

InfoBatch scores each tile by its **average focal loss over all valid pixels**. For a tile containing 95% background and 5% dead trees, the background pixels (easy, low loss) dominate the average, pulling the tile's score below the mean threshold. The tile is classified as "well-learned" and pruned — even though it contains valuable minority-class examples. This is a fundamental mismatch between InfoBatch's average-based scoring and the per-pixel imbalance structure of the task.

The near-identical training times (2.83h → 2.67h, ~6% reduction) also indicate that InfoBatch is providing almost no acceleration benefit, making the performance cost unjustifiable in its current form.

---

## Proposed Fix: Minority-Class Protection

Compute a boolean mask once before training that marks which tiles contain at least one tree pixel. During `prune()`, exclude those tiles from the well-learned candidate pool so they are **always retained with weight 1.0**. Only pure background tiles are eligible for pruning.

### What changes and what stays the same

| Aspect | Status | Detail |
|---|---|---|
| Scoring H(z) | **Unchanged** | All tiles are still scored; minority tiles contribute to the mean threshold |
| Pruning pool | **Changed** | Only background-only tiles are pruning candidates |
| Rescaling factor | **Unchanged** | `1/keep_ratio` still applied to retained easy background tiles |
| Gradient expectation | **Preserved** | Expectation is preserved within the background-only subpool |
| Annealing | **Unchanged** | Full dataset used after `delta × num_epochs` epochs |

---

## Files to Modify

| File | Change |
|---|---|
| `configs/debug.yaml` | Add `protect_minority` flag |
| `data_loader/__init__.py` | Add `compute_minority_mask()`, pass mask to InfoBatch |
| `upd_info.py` | Update `InfoBatch.__init__()` and `prune()` |

---

## Step-by-Step Changes

### Step 1 — Add `protect_minority` to config

**File:** `configs/debug.yaml`

Add the flag under the `infobatch` block:

```yaml
infobatch:
  enabled: true
  prune_ratio: 0.5
  delta: 0.875
  protect_minority: true   # Never prune tiles containing tree pixels
```

`protect_minority: false` preserves the original behaviour for ablation purposes.

---

### Step 2 — Add `compute_minority_mask` and pass mask to InfoBatch

**File:** `data_loader/__init__.py`

Add a utility function that iterates through training labels once before training begins and builds a boolean array where `True` = tile contains at least one positive (tree) pixel:

```python
def compute_minority_mask(dataset):
    """
    Returns a boolean numpy array of shape (N,) where True means the tile
    contains at least one positive (tree) pixel. Iterated once before
    training; used to protect minority-class tiles from InfoBatch pruning.
    """
    mask = np.zeros(len(dataset), dtype=bool)
    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample['label']
        if isinstance(label, torch.Tensor):
            mask[i] = label.any().item()
        else:
            mask[i] = np.any(label > 0)
    return mask
```

Update the InfoBatch wrapping block to compute and pass the mask:

```python
if use_infobatch:
    protect_minority = cfg['infobatch'].get('protect_minority', False)
    minority_mask = compute_minority_mask(train_ds) if protect_minority else None
    train_ds = InfoBatch(train_ds, num_epochs=num_epochs,
                         prune_ratio=prune_ratio, delta=delta,
                         minority_mask=minority_mask)
```

---

### Step 3 — Update `InfoBatch.__init__`

**File:** `upd_info.py`

Accept and store the optional minority mask:

```python
def __init__(self, dataset: Dataset, num_epochs: int,
             prune_ratio: float = 0.5, delta: float = 0.875,
             minority_mask=None):
    ...
    # minority_mask[i] = True means tile i contains tree pixels and is
    # never a pruning candidate, regardless of its loss score.
    if minority_mask is not None:
        self.minority_mask = np.asarray(minority_mask, dtype=bool)
    else:
        self.minority_mask = None
```

---

### Step 4 — Update `InfoBatch.prune()`

**File:** `upd_info.py`

Exclude minority tiles from the well-learned candidate pool before random selection:

```python
def prune(self):
    scores_np = self.scores.numpy()
    mean_score = scores_np.mean()

    # Well-learned candidates: score below the mean
    well_learned_mask = scores_np < mean_score

    # Exclude minority-class tiles from the pruning pool entirely.
    # They are moved directly to remained_indices with weight 1.0.
    if self.minority_mask is not None:
        well_learned_mask = well_learned_mask & ~self.minority_mask

    well_learned_indices = np.where(well_learned_mask)[0]
    remained_indices = np.where(~well_learned_mask)[0].tolist()

    selected_indices = np.random.choice(
        well_learned_indices,
        int(self.keep_ratio * len(well_learned_indices)),
        replace=False
    )
    self.reset_weights()
    if len(selected_indices) > 0:
        self.weights[selected_indices] = 1 / self.keep_ratio
        remained_indices.extend(selected_indices)
    self.num_pruned_samples += len(self.dataset) - len(remained_indices)
    np.random.shuffle(remained_indices)
    return remained_indices
```

---

## Expected Behaviour After Fix

- **Dead tree recall should recover** toward the no-InfoBatch baseline since tree-containing tiles are never dropped from training.
- **Speed benefit will be smaller** — the prunable pool is now restricted to background-only tiles. If a large proportion of tiles contain trees, the prunable pool may be small and InfoBatch's acceleration effect will be limited. This is useful information in itself about whether InfoBatch is the right fit for this dataset.
- **Background-only tiles still benefit** from InfoBatch's pruning and rescaling as intended.
- **The fix is fully ablatable** — setting `protect_minority: false` in the config restores the original behaviour exactly.

---

## Suggested Ablation Runs

Once implemented, run the following to isolate the effect:

| Configuration | Purpose |
|---|---|
| No InfoBatch | Baseline |
| InfoBatch 50% + `protect_minority: false` | Confirm original degradation is reproducible |
| InfoBatch 50% + `protect_minority: true` | Validate fix recovers dead tree metrics |
| InfoBatch 30% + `protect_minority: true` | Check if lower prune ratio improves further |
| InfoBatch 10% + `protect_minority: true` | Check lower bound of pruning aggressiveness |

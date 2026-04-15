# InfoBatch × TreeFinder Integration — Complete Technical Documentation

**Purpose:** Exhaustive record of every change made to integrate InfoBatch into the TreeFinder  
segmentation pipeline, with a frank analysis of what is likely causing the poor results.

---

## Table of Contents

1. [Repository Map](#1-repository-map)
2. [What InfoBatch Is Supposed to Do](#2-what-infobatch-is-supposed-to-do)
3. [File-by-File Change Log](#3-file-by-file-change-log)
   - 3.1 `main.py`
   - 3.2 `data_loader/__init__.py`
   - 3.3 `exps/train.py`
   - 3.4 `upd_info.py` (new file — adapted InfoBatch)
   - 3.5 `configs/debug.yaml`
4. [Key Divergences from Original InfoBatch](#4-key-divergences-from-original-infobatch)
5. [Identified Bugs and Likely Causes of Poor Results](#5-identified-bugs-and-likely-causes-of-poor-results)
6. [Recommendations for the Next Iteration](#6-recommendations-for-the-next-iteration)

---

## 1. Repository Map

```
treefinder-atml-infobatch/        ← OUR INTEGRATED PROJECT
├── main.py                       ← Modified
├── upd_info.py                   ← NEW (adapted InfoBatch core)
├── data_loader/__init__.py       ← Modified
├── exps/train.py                 ← Modified
├── configs/debug.yaml            ← Modified
├── exps/evaluate.py              ← Unchanged
├── models/                       ← Unchanged
├── data_loader/random_split.py   ← Unchanged
├── data_loader/by_*.py           ← Unchanged
└── utils/                        ← Unchanged

treefinder-main/treefinder-main/  ← ORIGINAL TREEFINDER (baseline, read-only)
InfoBatch-master/InfoBatch-master/← ORIGINAL INFOBATCH  (reference, read-only)
    └── infobatch/infobatch.py    ← The canonical implementation we adapted
    └── examples/cifar_example.py ← The three-line integration template
```

---

## 2. What InfoBatch Is Supposed to Do

InfoBatch (ICLR 2024) speeds up training **without accuracy loss** via two mechanisms:

1. **Pruning:** Each epoch, samples whose loss is *below* the batch mean are considered "well-learned."  
   A fraction (`prune_ratio`) of those are randomly dropped from that epoch.

2. **Gradient rescaling:** The samples that *are* kept (including the retained well-learned ones)  
   have their per-sample loss multiplied by `1 / keep_ratio` so that the gradient magnitude  
   approximates what it would have been if the full dataset was seen.

The promised integration is literally three lines (from their CIFAR example):
```python
# Line 1 – wrap dataset
trainset = InfoBatch(trainset, num_epochs, ratio, delta)
# Line 2 – use InfoBatch's sampler
sampler = trainset.sampler
# Line 3 – call update() per batch with per-sample loss
loss = trainset.update(loss)
```

The rest of this document explains what we did differently and what went wrong.

---

## 3. File-by-File Change Log

### 3.1 `main.py`

| | Original TreeFinder | Our Version |
|---|---|---|
| **Signature of `get_dataloader()`** | `(cfg)` → returns 3 loaders | `(cfg, train_fraction_seed)` → returns 3 loaders + `train_dataset` |
| **`train_model()` call** | 4 args: `model, train_loader, val_loader, cfg, exp_name` | 5 args: adds `train_dataset` |
| **Run loop** | Single run | Loop over `num_train_splits` runs, each with `seed + run_idx` |
| **GDrive upload** | Not present | Added at end of all runs |
| **Cross-run summary YAML** | Not present | Written when `num_splits > 1` |

**What changed and why:**  
`train_dataset` must be passed to `train_model` because the training loop needs to call  
`train_dataset.update(per_img_bce, scores=per_img_score)` per batch, and to call  
`train_loader.sampler.reset()` at the start of each epoch.

---

### 3.2 `data_loader/__init__.py`

#### Added: `_compute_minority_mask(dataset)` (lines 8–23)

```python
def _compute_minority_mask(dataset):
    mask = np.zeros(len(dataset), dtype=bool)
    for i in range(len(dataset)):
        label = dataset[i]['label']
        mask[i] = label.any().item()  # True if tile contains at least one tree pixel
    return mask
```

**Purpose:** Prevent InfoBatch from pruning tiles that contain any positive (tree) pixels.  
**Problem:** This function iterates over the *entire* training set at startup. For large datasets  
this is very slow and blocks training from starting. More critically, see Bug #1 below.

#### Added: Train-fraction subsampling (lines 51–78)

```python
train_fraction = cfg['data'].get('train_fraction', 1.0)
if 0.0 < train_fraction < 1.0:
    train_ds = Subset(train_ds, indices)
```

Saves the selected indices to `splits/trainfrac_{fraction}_seed{seed}.json` for reproducibility.  
**This is independent of InfoBatch and correct.** Currently set to `0.7` in `debug.yaml`.

#### Added: InfoBatch wrapping (lines 80–101)

```python
if use_infobatch:
    train_ds = InfoBatch(train_ds, num_epochs=num_epochs, prune_ratio=prune_ratio,
                         delta=delta, minority_mask=minority_mask)
    train_sampler = train_ds.sampler
    train_shuffle = False
```

**Correct pattern.** Matches the InfoBatch reference example.

#### Changed: DataLoader creation

- Sets `shuffle=train_shuffle` (False when InfoBatch is active — correct, sampler handles ordering).
- Sets `sampler=train_sampler` (None when InfoBatch disabled — correct).
- Added `persistent_workers=persistent` (good for performance).

---

### 3.3 `exps/train.py`

This is the most heavily modified file.

#### Added: `train_dataset` parameter to `train_model()`

Original signature: `train_model(model, train_loader, val_loader, cfg, exp_name)`  
New signature: `train_model(model, train_loader, val_loader, train_dataset, cfg, exp_name)`

#### Added: `use_infobatch` flag

```python
use_infobatch = cfg.get('infobatch', {}).get('enabled', False)
```

Used to gate all InfoBatch-specific logic.

#### Added: `sampler.reset()` call at epoch start (lines 179–184)

```python
if use_infobatch:
    sampler = train_loader.sampler
    if hasattr(sampler, 'reset'):
        sampler.reset()
```

**This is a critical bug fix** relative to the original InfoBatch library.  
In the original, `IBSampler.__iter__()` calls `self.reset()`, which means `prune()` fires  
every time PyTorch internally calls `iter(sampler)` — which can happen multiple times per epoch  
(e.g., inside the `_sampler_iter is None` guard in `info_hack_indices`). This causes scores  
to be computed against stale loss values and epoch counting to be wrong.  
Our fix: `IBSampler.__iter__()` does **not** call `reset()`; instead we call it once explicitly here.

#### Added: Per-image BCE computation (lines 218–220)

```python
valid_pixels_per_img = valid_mask.view(valid_mask.shape[0], -1).sum(dim=1).clamp(min=1)
per_img_bce = (raw_loss * valid_mask).view(raw_loss.shape[0], -1).sum(dim=1) / valid_pixels_per_img
```

**This converts the (B,1,H,W) raw loss into a per-image scalar** used by InfoBatch to rank samples.  
The original InfoBatch example (CIFAR-100 classification) has a natural per-sample scalar loss.  
For segmentation, we must aggregate over spatial dimensions — this is the correct approach.

#### Added: Batch weights retrieval before `update()` (line 223)

```python
batch_weights = train_dataset.weights[train_dataset.cur_batch_index].to(device)
```

**Note:** This line captures `cur_batch_index` *before* `update()` clears it (line 126 of  
`upd_info.py` sets `self.cur_batch_index = []`). See Bug #3 for a subtle issue here.

#### Added: Optional per-image Dice scoring (lines 225–239)

```python
if w_dice > 0:
    per_img_score = per_img_bce + w_dice * per_img_dice
    dloss = (per_img_dice * batch_weights).mean()
else:
    dloss = torch.tensor(0.0, device=device)
    per_img_score = per_img_bce
```

When Dice loss is active, the pruning score is `H(z) = BCE + w_dice * Dice`.  
Gradient rescaling is applied to both BCE and Dice independently.  
**Note:** In the current config, `w_dice: 0`, so this path is never taken.

#### Added: `train_dataset.update(per_img_bce, scores=per_img_score)` (line 243)

```python
bce_loss = train_dataset.update(per_img_bce, scores=per_img_score)
```

This is the core InfoBatch call. It:
1. Stores `per_img_score` as the new loss score for each sample in the batch.
2. Multiplies `per_img_bce` by the sample weights (rescaling).
3. Returns the weighted mean BCE for `total_loss.backward()`.

#### Added: Training accuracy tracking (lines 261–266)

Added `train_correct` and `train_total_px` counters — these were absent in the original.

#### Added: Validation accuracy tracking (lines 319–323)

Same addition as training.

#### Added: CSV metrics logging (lines 156–166, 373–387)

Every epoch writes a row to `{exp_name}_metrics.csv`:
```
epoch, train_loss, val_loss, train_acc, val_acc,
train_bce, train_dice, val_bce, val_dice,
epoch_time_s, pruned_count
```

This is purely observational and does not affect training.

#### Added: `get_pruned_count()` logging (lines 367–371)

```python
pruned_count = train_dataset.get_pruned_count() if use_infobatch else 0
```

---

### 3.4 `upd_info.py` (New File — Adapted InfoBatch)

This is a copy of the original `infobatch/infobatch.py` with several targeted modifications.

#### Change 1: Score initialization

| Original | Ours |
|---|---|
| `self.scores = torch.ones(len(self.dataset)) * 3` | `self.scores = torch.ones(len(self.dataset))` |

**Rationale stated in code:** Initialize to 1.0 so that epoch 1 sees mean=1.0, all samples  
score == mean, `well_learned_mask` is all-False, and full dataset is used in epoch 1 without  
any pruning. After epoch 1, actual losses populate scores.

**Actual effect:** The original uses 3.0. If actual losses are typically in the range [0.1, 1.0]  
(which focal loss and BCE commonly produce), then in epoch 1 with scores=3.0, ALL samples have  
score < mean=3.0, so `well_learned_mask` is all-True, and InfoBatch would try to prune *everything*.  
Initializing to 1.0 avoids this and is the correct choice — unless actual losses are regularly > 1.0.

#### Change 2: Added `minority_mask` parameter to `InfoBatch.__init__`

```python
def __init__(self, dataset, num_epochs, prune_ratio=0.5, delta=0.875, minority_mask=None):
    ...
    self.minority_mask = np.asarray(minority_mask, dtype=bool) if minority_mask is not None else None
```

In `prune()`:
```python
if self.minority_mask is not None:
    well_learned_mask = well_learned_mask & ~self.minority_mask
```

**Purpose:** Never prune tiles that contain tree pixels (positive class).  
**Potential issue:** The minority mask is computed at initialization time. Any resampling via  
`Subset` (from `train_fraction`) happens *before* InfoBatch wrapping, so the mask size matches.  
However, see Bug #1 for the correctness problem.

#### Change 3: `update()` takes two arguments

```python
# Original
def update(self, values):
    loss_val = values.detach().clone()
    self.scores[...] = loss_val.cpu()
    values.mul_(weights)      # in-place multiply
    return values.mean()

# Ours
def update(self, values, scores=None):
    score_val = (scores if scores is not None else values).detach().clone()
    self.scores[...] = score_val.cpu()
    return (values * weights).mean()   # out-of-place multiply
```

Two differences:
1. **Separate scoring tensor** — allows using combined loss (BCE+Dice) for scoring while  
   rescaling only the BCE component.
2. **Out-of-place multiplication** — original modifies `values` in-place with `values.mul_(weights)`,  
   then returns `values.mean()`. Our version computes `(values * weights).mean()` without  
   modifying `values`. Both are mathematically identical but our version is safer.

#### Change 4: `IBSampler.__iter__()` does not call `reset()`

```python
# Original
def __iter__(self):
    self.reset()   # ← causes spurious prune() on every internal iter(sampler)
    return self

# Ours
def __iter__(self):
    # Do NOT call reset() here. reset() is called explicitly once per epoch.
    self.iter_obj = iter(self.sample_indices)
    return self
```

**This is the most important correctness fix.** See detailed explanation in Section 3.3.

#### Change 5: `stop_prune` boundary check

```python
# Original
if self.iterations == self.stop_prune + 1:   # float arithmetic — can miss the transition
    self.dataset.reset_weights()

# Ours
if self.iterations == int(self.stop_prune) + 1:   # explicit int cast
    self.dataset.reset_weights()
```

Small but important: `stop_prune = num_epochs * delta` is a float. The original compares  
`int == float`, which only works when the product is a whole number (e.g., 100 * 0.875 = 87.5  
is NOT an integer, so `self.stop_prune + 1 = 88.5`, and `self.iterations` (an int) will  
**never equal 88.5**). This means `reset_weights()` is **never called** in the original.  
Our `int(self.stop_prune) + 1 = 88` correctly triggers the reset.

#### Change 6: Added `__getitems__()` override

```python
def __getitems__(self, indices):
    return [self.__getitem__(idx) for idx in indices]
```

PyTorch >= 2.x's `_MapDatasetFetcher` calls `__getitems__` directly when present, bypassing  
`__getitem__`. Without this override, the `(index, data)` tuple contract that  
`info_hack_indices` relies on would be broken.

#### Change 7: Added multiprocessing safety guard in `__getattr__`

```python
def __getattr__(self, name):
    if name == 'dataset':
        raise AttributeError(name)   # prevents infinite recursion during unpickling
    return getattr(self.dataset, name)
```

The original version has no such guard, which can cause a `RecursionError` in  
multiprocessing worker processes when the dataset is pickled/unpickled.

---

### 3.5 `configs/debug.yaml`

| Key | Original | Ours |
|---|---|---|
| `data.train_fraction` | Not present | `0.7` |
| `data.num_train_splits` | Not present | `1` |
| `infobatch` block | Not present | Added (see below) |
| `gdrive` block | Not present | Added |
| `training.w_dice` | `0.5` | `0` |

**Added InfoBatch config block:**
```yaml
infobatch:
  enabled: true
  prune_ratio: 0.5      # 50% of well-learned samples are dropped each epoch
  delta: 0.875          # Pruning active for first 87.5% of epochs
  protect_minority: true
```

**Note on `w_dice: 0`:** The original TreeFinder `debug.yaml` also has `w_dice: 0.5`.  
We changed it to `0`. With `w_dice=0`, the Dice scoring path in `train.py` is  
skipped entirely, and `per_img_score = per_img_bce`. This is a simplification that  
removes one potential source of bugs during debugging.

---

## 4. Key Divergences from Original InfoBatch

The original InfoBatch was designed and validated on **image classification** (CIFAR-100,  
ImageNet). We are applying it to **binary semantic segmentation** with a spatial loss.  
This is a fundamentally different task with several implications:

| Dimension | InfoBatch Classification | Our Segmentation Adaptation |
|---|---|---|
| Loss shape | Scalar per sample | (B,1,H,W) — must be aggregated |
| Loss aggregation | Direct CE loss per sample | Mean over valid (non-nodata) pixels |
| Label balance | Balanced CIFAR classes | Highly imbalanced: most tiles are mostly background |
| Score meaning | How well model classifies image | How well model predicts pixels in tile |
| "Easy" sample risk | Low-loss samples are well-learned | Low-loss tiles might just be empty (all background) |
| Pruning risk | Prunes easy images | May prune ALL-background tiles that should be kept |

---

## 5. Identified Bugs and Likely Causes of Poor Results

### Bug #1 (HIGH SEVERITY): `_compute_minority_mask` is called on the WRONG dataset

**Location:** `data_loader/__init__.py`, lines 93

```python
minority_mask = _compute_minority_mask(train_ds) if protect_minority else None
...
train_ds = InfoBatch(train_ds, ..., minority_mask=minority_mask)
```

`train_ds` at this point is already a `Subset` (after `train_fraction` slicing). `_compute_minority_mask`  
calls `dataset[i]['label']` using indices `0..len(subset)-1`, which are the **subset-local** indices.  
The `Subset.__getitem__` correctly maps these to underlying indices, so the labels themselves  
are correct. **This part is fine.**

However, `InfoBatch.prune()` references `self.minority_mask` against `self.scores`, where  
`self.scores` has length `len(self.dataset) = len(subset)`. The mask also has length `len(subset)`.  
**This is consistent and correct.**

**BUT:** `_compute_minority_mask` is an O(N) scan that loads every tile's label from disk.  
For large datasets this adds significant startup time. More importantly, if the dataset  
has the GeoTIFF label format, calling `dataset[i]` inside `_compute_minority_mask` while  
the main process hasn't initialized CUDA yet may cause subtle ordering issues.

**Actual critical issue:** With `protect_minority: true` and highly imbalanced data (most  
tiles being mostly background), a large fraction of tiles will have `minority_mask[i] = False`  
(no tree pixels). InfoBatch will freely prune these. This means **the model sees even fewer  
positive-containing tiles**, making the class imbalance problem worse. The protection only  
helps tiles *with* trees, but the damage is in over-pruning the already-rare positive tiles  
through secondary effects (their context tiles being removed, batch statistics being skewed).

---

### Bug #2 (HIGH SEVERITY): `train_dataset.weights` indexing uses `cur_batch_index` before InfoBatch sets it

**Location:** `exps/train.py`, lines 219–223

```python
# ← raw_loss = criterion(outputs, labels) happens first
# ← Then we compute per_img_bce from raw_loss

batch_weights = train_dataset.weights[train_dataset.cur_batch_index].to(device)  # line 223
```

`cur_batch_index` is set by `info_hack_indices` (the monkey-patched `DataLoader.__next__`).  
It is set *when the batch is fetched*, not when the loss is computed.  
`batch_weights` is read at line 223, which is **after** `raw_loss = criterion(outputs, labels)`  
at line 215 — so by this point, `cur_batch_index` should already be set for this batch.  
**This is technically fine for a single-GPU synchronous training loop.**

However, when `update()` is called at line 243:
```python
bce_loss = train_dataset.update(per_img_bce, scores=per_img_score)
```

Inside `update()`, line 134:
```python
return (values * weights).mean()
```

The `weights` used here are the **same** weights read from `self.weights[self.cur_batch_index]`  
(computed at the start of `update()`). But we already captured them in `batch_weights` at  
line 223 for use in the Dice calculation. This duplication is harmless.

**Actual problem:** `batch_weights` at line 223 are used for `dloss` (Dice rescaling) at line 236,  
but since `w_dice=0`, this path is never taken. **Not a live bug with current config.**

---

### Bug #3 (HIGH SEVERITY): Loss function mismatch between scoring and gradient signal

**Location:** `exps/train.py`, line 243 and `configs/debug.yaml`

The config uses `criterion: FocalLoss` with `alpha=0.25, gamma=2`.  
Focal loss **downweights easy samples** by multiplying the cross-entropy by  
`(1 - p)^gamma`. This means samples the model already classifies well have  
**very low focal loss**.

InfoBatch's pruning criterion is: samples with `loss < mean(losses)` are "well-learned".  
With focal loss, **tiles the model predicts confidently** will always have low loss —  
including tiles that are all-background (trivially easy, not informative).

This creates a **conflict**: InfoBatch prunes low-loss samples, but focal loss  
already explicitly downweights them. The combination means:
- Easy background tiles have near-zero focal loss → InfoBatch identifies them as well-learned → prunes them
- Remaining dataset is disproportionately hard/ambiguous tiles
- But focal loss on hard tiles was already high → gradient signal is not meaningfully changed
- The gradient rescaling (`1/keep_ratio`) upweights the gradients of kept easy samples,  
  but those samples had near-zero focal loss to begin with — so the "rescaled" gradient  
  is still near-zero

**Root cause:** InfoBatch was designed for plain cross-entropy, where the loss directly  
reflects how much information a sample provides. Focal loss pre-adjusts this signal;  
combining both pruning mechanisms leads to double-discounting of easy samples.

**Recommendation:** Use `BCEWithLogitsLoss` (plain BCE) when InfoBatch is active, or  
use BCE for *scoring* (what we pass to InfoBatch) even if Dice or another loss is used  
for training. The current code already separates scoring from gradient loss, but  
the scoring uses focal loss values, which is the incorrect metric.

---

### Bug #4 (MEDIUM SEVERITY): `train_fraction: 0.7` compounds the data scarcity

**Location:** `configs/debug.yaml`, line 17

The config uses `train_fraction: 0.7`, meaning only 70% of the already-limited training  
set is used. InfoBatch is then applied on top of this 70%, further pruning ~50% each epoch  
(before the scores stabilize). In early epochs, roughly `0.7 × 0.5 = 35%` of the original  
training data is seen per epoch. This is far below what the model needs to learn meaningful  
features for a segmentation task.

For comparison, the original InfoBatch paper on CIFAR-100 uses 50,000 samples with pruning  
starting after scores are stable. Starting with only 35% of data from epoch 1 onward is  
likely to cause underfitting and instability.

**Recommendation:** Set `train_fraction: 1.0` when debugging InfoBatch correctness, then  
re-introduce the fraction once InfoBatch is confirmed working.

---

### Bug #5 (MEDIUM SEVERITY): IBSampler calls `reset()` once in `__init__`

**Location:** `upd_info.py`, lines 211–213

```python
def __init__(self, dataset: InfoBatch):
    ...
    self.reset()   # ← epoch 0 prune() is called here, before any training has happened
```

When the sampler is constructed, `reset()` is called immediately. At this point all  
scores are 1.0 (uniform), so `prune()` sees mean=1.0, and `well_learned_mask` is  
all-False (no sample has score < mean when all are equal to mean). **This means epoch 0  
is safe** — `prune()` returns the full dataset with no pruning.

However, `self.iterations` is incremented to 1 after this init call. When  
`train.py` calls `sampler.reset()` at the start of epoch 1 (the first actual epoch),  
`iterations` is already 1, so this call advances to `iterations=2` and prunes using  
the scores from epoch 0 (which are still all 1.0 from init). **Epoch 1 training runs on**  
**the output of the epoch-0 reset** (full data, correct), but the scores used for pruning  
before epoch 2 are still all 1.0.

**Net effect:** Pruning does not effectively start until epoch 2 (after real losses from  
epoch 1 have been recorded). This is a one-epoch offset that is unlikely to cause  
significant accuracy loss but wastes one epoch.

---

### Bug #6 (MEDIUM SEVERITY): `num_workers > 0` with monkey-patched `__next__` is risky

**Location:** `upd_info.py`, line 41 + `data_loader/__init__.py`, line 103

```python
_BaseDataLoaderIter.__next__ = info_hack_indices
```

This monkey-patch replaces the DataLoader's internal `__next__` globally. When  
`num_workers > 0`, each worker process has its own DataLoader iterator, but the  
monkey-patch is applied in the main process. The crucial question is whether  
`self._dataset` in a worker's iterator refers to the `InfoBatch` instance from the  
main process or a pickled copy.

With `num_workers=16`, each worker gets a **separate pickled copy** of the dataset.  
Calls to `set_active_indices()` in the worker will modify **the worker's copy**,  
not the main-process `train_dataset`. When `train.py` calls  
`train_dataset.weights[train_dataset.cur_batch_index]`, `cur_batch_index` may be  
`None` or stale because the indices were set in a worker, not the main process.

**This is likely causing runtime errors or silently incorrect behavior when `num_workers > 0`.**  
The original InfoBatch CIFAR example uses `num_workers=0`.  
Our config uses `num_workers=16`.

---

### Bug #7 (LOW SEVERITY): ExponentialLR scheduler with default gamma=0.95 over 100 epochs

**Location:** `configs/debug.yaml`, `training.scheduler.type: ExponentialLR`

With `gamma=0.95` and 100 epochs, the LR decays to `lr * 0.95^100 ≈ lr * 0.006`.  
Starting at `lr=1e-4`, by epoch 100 the LR is ~6e-7. Combined with InfoBatch pruning  
which reduces effective dataset size, the model may stall completely in later epochs.  
This is not an InfoBatch bug but interacts poorly with aggressive pruning.

---

### Bug #8 (LOW SEVERITY): `val_acc` in training loop uses binary threshold 0.5 on logits

**Location:** `exps/train.py`, line 262

```python
preds = (torch.sigmoid(outputs) > 0.5).long()
```

This is correct, but the accuracy metric is pixel-wise and extremely dominated by the  
background class (which may be >95% of pixels). A model that predicts all-background  
achieves >95% pixel accuracy. **Accuracy is not a meaningful metric for this task.**  
This does not affect training but makes it hard to detect whether InfoBatch is helping.

---

## 6. Recommendations for the Next Iteration

### Priority 1: Fix the num_workers issue (Bug #6)

The monkey-patch approach only works with `num_workers=0`. Set `num_workers=0` in the  
config, or replace the monkey-patch with a proper wrapper that collects indices in the  
main process. This is likely the most impactful single fix.

```yaml
experiment:
  num_workers: 0    # Required for InfoBatch monkey-patch to work correctly
```

### Priority 2: Use BCE (not FocalLoss) as the InfoBatch scoring loss (Bug #3)

Change `configs/debug.yaml`:
```yaml
training:
  criterion:
    type: "BCEWithLogitsLoss"
    w_pos: 5  # upweight positive class instead of using focal loss
```

Or keep FocalLoss for training gradients but compute BCE-only scores for InfoBatch:
```python
# In train.py, compute a separate BCE loss for scoring
bce_for_score = nn.BCEWithLogitsLoss(reduction='none')(outputs, labels)
per_img_score = (bce_for_score * valid_mask).view(B, -1).sum(dim=1) / valid_pixels_per_img
bce_loss = train_dataset.update(per_img_bce_focal, scores=per_img_score)
```

### Priority 3: Disable `train_fraction` during InfoBatch debugging (Bug #4)

```yaml
data:
  train_fraction: 1.0   # Use full dataset when validating InfoBatch correctness
```

### Priority 4: Validate InfoBatch is actually pruning

Add a sanity check after a few epochs: inspect `train_dataset.scores` distribution  
and verify `pruned_count` is increasing. If `pruned_count` stays at 0, InfoBatch is  
not functioning (likely due to Bug #6 — scores are never updated from workers).

```python
# After training epoch 2+, check:
print(f"Score range: {train_dataset.scores.min():.4f} - {train_dataset.scores.max():.4f}")
print(f"Score mean: {train_dataset.scores.mean():.4f}")
print(f"Samples with score < mean: {(train_dataset.scores < train_dataset.scores.mean()).sum()}")
```

### Priority 5: Reconsider `protect_minority` with imbalanced data (Bug #1)

With segmentation and heavy class imbalance, protecting only "minority tiles" (tiles  
with any tree pixel) may not be enough. Consider:
1. Protecting all tiles where tree pixels exceed a threshold (e.g., >1% of tile)
2. Tracking per-tile class balance and using it in the pruning score

### Priority 6: Verify `stop_prune` boundary (Change 5 in Section 3.4)

Confirm that our fix `int(self.stop_prune) + 1` fires correctly:
```python
# With num_epochs=100, delta=0.875:
# stop_prune = 87.5
# int(87.5) + 1 = 88
# This fires when iterations == 88, i.e., after epoch 88
# ✓ Correct — pruning active for epochs 1-87, full data from epoch 88 onward
```

### Priority 7: Replace pixel accuracy with IoU for monitoring

The `val_acc` metric is dominated by background pixels and is not meaningful.  
Track IoU (already computed in `exps/evaluate.py`) during training for useful monitoring.

---

## Summary Table: What Changed vs. Original TreeFinder

| File | Nature of Change | Risk Level |
|---|---|---|
| `main.py` | Added multi-run loop, `train_dataset` return, GDrive | Low |
| `data_loader/__init__.py` | Added InfoBatch wrapping, minority mask, train_fraction | Medium |
| `exps/train.py` | Added per-image loss, `update()` call, `reset()`, CSV logging | High |
| `upd_info.py` | New file: InfoBatch adapted for segmentation | High |
| `configs/debug.yaml` | Added infobatch section, train_fraction, gdrive | Medium |

## Summary Table: What Changed vs. Original InfoBatch

| Aspect | Original InfoBatch | Our Version | Verdict |
|---|---|---|---|
| Score init | 3.0 | 1.0 | **Ours is better** for focal/BCE losses in [0,1] range |
| `__iter__` reset | Calls `reset()` | Does NOT call `reset()` | **Ours is correct** — prevents spurious pruning |
| `stop_prune` check | `== self.stop_prune + 1` (float) | `== int(self.stop_prune) + 1` | **Ours is correct** |
| `update()` signature | Single `values` arg | Separate `values` and `scores` | **Ours is more flexible** |
| `update()` in-place | `values.mul_(weights)` | `(values * weights).mean()` | Equivalent, ours is safer |
| minority_mask | Not present | Present | **New feature** — correctness depends on data |
| `__getitems__` | Not present | Present | **Required** for PyTorch >= 2.x |
| `__getattr__` guard | No recursion guard | Recursion guard added | **Ours is correct** |
| num_workers compatibility | Designed for 0 | Used with 16 | **BROKEN** — see Bug #6 |

# InfoBatch Bug Fixes

This document summarises the bugs found in the InfoBatch implementation, why they produced wrong results, and what was changed to fix them.

---

## Fix 1 — Score Initialisation: zeros → ones

### What was broken
`self.scores` was initialised to `torch.zeros(...)`. With all scores at 0, the pruning threshold (mean) was also 0. The pruning condition `score < mean` becomes `0 < 0`, which is **always False** — so the full dataset was used and no pruning occurred in epoch 1. This was accidentally correct for epoch 1, but it violated the spec and made the initial state fragile: any small numerical noise could make some scores slightly negative, causing those samples to be pruned before any real training had occurred.

### What was fixed
Initialise to `torch.ones(...)` per the spec. With all scores at 1 and mean = 1, `score < mean` is `1 < 1` — always False — so epoch 1 correctly uses the full dataset. After epoch 1, scores are replaced with real loss values and pruning begins from epoch 2 onward.

**File:** `upd_info.py`, `InfoBatch.__init__`

---

## Fix 2 — Representative Score H(z) only used BCE, ignored dice loss

### What was broken
The per-sample pruning score passed to `update()` was only the per-image focal/BCE loss:
```python
per_img_loss = (raw_loss * valid_mask).view(...).sum(dim=1) / valid_pixels_per_img
bce_loss = train_dataset.update(per_img_loss)
```
The dice loss was computed separately as a scalar and added to the total loss for backprop, but **never incorporated into the InfoBatch score**. This meant pruning decisions ignored the dice component entirely. A sample that was easy on BCE but hard on dice would be incorrectly scored as "easy" and pruned, even though it still carried useful gradient signal in the dice term.

### What was fixed
Per-image dice is now computed alongside per-image BCE, and the combined H(z) = BCE + w_dice × dice is passed as the pruning score:
```python
per_img_score = per_img_bce + w_dice * per_img_dice
bce_loss = train_dataset.update(per_img_bce, scores=per_img_score)
```
`update()` was extended with an optional `scores` parameter so that H(z) can differ from the value used for gradient rescaling. The BCE component still drives rescaling (keeping the existing loss structure), while the combined score drives pruning decisions.

**Files:** `upd_info.py` (`update()` signature), `exps/train.py` (InfoBatch branch)

---

## Fix 3 — Annealing weight reset never fired (float vs int comparison bug)

### What was broken
When entering the annealing phase, `reset_weights()` is supposed to be called exactly once to set all sample weights back to 1.0. The guard was:
```python
if self.iterations == self.stop_prune + 1:
    self.dataset.reset_weights()
```
`self.iterations` is an integer; `self.stop_prune` is a float (`num_epochs * delta`, e.g. `100 * 0.875 = 87.5`). The comparison `88 == 88.5` is **always False** in Python. `reset_weights()` was therefore **never called**.

As a result, samples that were assigned `weight = 1/keep_ratio` (e.g. 2.0) in the last pruning epoch retained those elevated weights throughout the entire annealing phase. Every batch during annealing had some samples contributing 2× their actual gradient, corrupting the unbiased full-dataset training the annealing phase is meant to provide.

### What was fixed
Cast `stop_prune` to int before the comparison:
```python
if self.iterations == int(self.stop_prune) + 1:
    self.dataset.reset_weights()
```
For `stop_prune = 87.5`, `int(87.5) + 1 = 88`, so `88 == 88` correctly fires once when annealing begins and resets all weights to 1.0.

**File:** `upd_info.py`, `IBSampler.reset`

---

## Fix 4 — Dice loss gradient rescaling used mean weight instead of per-sample weights

### What was broken
After Fix 2 introduced per-image dice for scoring, the dice loss for backprop was rescaled as:
```python
dloss = dloss * batch_weights.mean()
```
`dloss` here was the original **scalar** batch-level dice (from `DiceLoss.forward()`), multiplied by the mean of the batch weights. This is not equivalent to per-sample weighting:

- BCE rescaling: `mean(per_img_bce[i] * weight[i])` — each sample's loss scaled by its own weight
- Dice rescaling: `scalar_dloss * mean(weight)` — a single scalar scaled by the average weight

These are only equal when all per-sample losses are identical. In practice, when samples have different losses (which is the entire point of InfoBatch), the dice gradient was under- or over-compensated relative to BCE, producing biased gradient estimates.

### What was fixed
Per-image dice (already computed for scoring) is now used directly for the backprop loss as well, with the same per-sample weight applied:
```python
dloss = (per_img_dice * batch_weights).mean()
```
This makes BCE and dice rescaling consistent: both apply per-sample InfoBatch weights before taking the mean. The call to the scalar `DiceLoss` module in the InfoBatch branch was removed since `per_img_dice` already provides the correct values.

**File:** `exps/train.py`, InfoBatch training branch

---

## Fix 5 — Distributed training: `reset()` call raised AttributeError

### What was broken
The training loop called `train_loader.sampler.reset()` unconditionally. In non-distributed mode the sampler is `IBSampler`, which has `reset()` — no problem. In distributed mode the sampler is `DistributedIBSampler`, which inherits from PyTorch's `DistributedSampler` and **has no `reset()` method**. This would raise `AttributeError` immediately at the start of every epoch in any distributed training run, making InfoBatch completely unusable in multi-GPU setups.

(`DistributedIBSampler` correctly handles pruning inside its own `__iter__` via `self.sampler.reset()` → `DatasetFromSampler.reset()` → `IBSampler.reset()`, so the explicit call from the training loop is not needed in that path.)

### What was fixed
Added a `hasattr` guard so the explicit reset is only called when the sampler exposes it:
```python
sampler = train_loader.sampler
if hasattr(sampler, 'reset'):
    sampler.reset()
```
`IBSampler` gets the explicit per-epoch reset as before. `DistributedIBSampler` skips it and relies on the reset inside `__iter__`.

**File:** `exps/train.py`, per-epoch sampler reset block

---

## On Hold — DeepSMOTE for Class Imbalance (74.4% background / 25.6% tree)

### What DeepSMOTE does
DeepSMOTE trains an encoder-decoder to map samples into a latent space, applies SMOTE-style interpolation between minority-class samples in that space, and decodes the interpolated vectors back to pixel space. For segmentation, this means generating synthetic **image-mask pairs** where the minority class (tree pixels) is more prevalent, effectively shifting the tile-level class distribution toward the minority.

### Why we are not implementing it yet

**1. The existing loss stack already addresses pixel-level imbalance directly**
Focal loss (via α and γ) down-weights easy background pixels and focuses gradient on hard examples. Dice loss is independent of the true-negative count, so the 74% background does not dominate it. Together they handle within-tile pixel imbalance at exactly the right level — the loss function — without requiring synthetic data generation.

**2. DeepSMOTE targets sample-level imbalance; our primary imbalance is pixel-level**
DeepSMOTE generates new tree-heavy tiles to increase minority-class exposure at the sample level. This is most valuable when the model sees too few tree-containing tiles. Before adding a generative pipeline, it is worth confirming through the tile-level class distribution and per-class validation metrics whether sample-level exposure is actually the bottleneck.

**3. InfoBatch already acts as a soft positive-sample filter**
By pruning low-loss (well-learned) samples first, InfoBatch disproportionately retains tree-containing tiles, which tend to have higher loss. This partially compensates for sample-level imbalance without any additional mechanism.

**4. Non-trivial pipeline overhead**
DeepSMOTE requires training a separate encoder-decoder on satellite imagery and generating a synthetic augmentation dataset before main training begins. For remote sensing data with complex spectral and spatial structure, this adds meaningful engineering cost and the quality of synthetic tiles needs careful validation before they can safely be mixed into training.

### When we would revisit it
If per-class validation metrics show the model is still under-performing on tree pixels after tuning focal loss α and confirming that tile-level oversampling does not close the gap, DeepSMOTE becomes a justified next step — specifically if the tile-level analysis shows a large fraction of training tiles contain no trees at all.

---

## Copy-Paste Augmentation — Worth Trying

### What it does
Cuts annotated tree regions from one tile, pastes them into a background-heavy tile, and updates the mask accordingly. No generative model required — patches are real, spatially coherent, and correctly labelled by construction.

### Why it is a good fit
- Low implementation cost and easy to ablate
- Directly increases tree pixel exposure in background-dominant tiles
- Produces realistic patches since they come from real imagery
- Addresses sample-level exposure without the overhead of DeepSMOTE

### Key concern for satellite imagery
Trees have specific canopy textures, shadow patterns, and spectral signatures tied to the surrounding landscape (forest type, season, soil, illumination angle). Pasting a tree region from one tile into a spectrally different background can create visible boundary seams that the model may learn to detect rather than learning genuine tree appearance. This risk is lower if tiles are from similar geographic regions and acquisition conditions, and can be mitigated with Gaussian boundary blending.

### Verdict
The most justified augmentation addition given the current setup. Should be tried before anything heavier, and the impact assessed through per-class validation metrics.

---

## On Hold — OHEM (Online Hard Example Mining)

### What it does
During each forward pass, OHEM computes the loss for all pixels (or samples) in a batch, selects the top-K hardest (highest-loss) elements, and restricts gradient updates to those — discarding easy examples within the batch.

### Why it overlaps heavily with the existing stack

| | InfoBatch | Focal Loss | OHEM |
|---|---|---|---|
| Granularity | Tile / sample level | Pixel level (soft) | Pixel level (hard) |
| Timing | Across epochs | Every forward pass | Every forward pass |
| Mechanism | Prune easy tiles from the epoch | Re-weight each pixel by (1−p_t)^γ | Hard-threshold; drop easy pixels from loss |

Focal loss already performs a continuous, soft version of pixel-level hard example mining on every forward pass via the `(1 − p_t)^γ` modulating factor. OHEM is a hard-threshold version of the same idea applied at the same granularity. Adding OHEM on top means the same intent is covered three times (focal loss, InfoBatch, OHEM).

### Risk of double filtering
InfoBatch has already filtered for hard tiles at the epoch level before a batch is formed. OHEM would then further filter for hard pixels within those batches. This double concentration of gradient updates onto a small subset of very hard examples risks destabilising training, particularly in early epochs when loss values are noisy and unreliable.

### Verdict
Not justified given the current stack. The combination of focal loss (pixel-level soft mining) and InfoBatch (tile-level hard mining) already covers the intent of OHEM from two directions. Introducing OHEM adds complexity and instability risk without a clear gap to fill.

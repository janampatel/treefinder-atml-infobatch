# CS-IB: Class-Stratified InfoBatch with Warmup

## Research Goal

Outperform the TreeFinder author's baseline result **and** standard InfoBatch, using **less training data and less wall-clock time**. Algorithm must be model-agnostic and generalise to any segmentation task with class imbalance.

---

## Diagnosed Flaws in Standard InfoBatch (for Imbalanced Segmentation)

| # | Flaw | Root Cause |
|---|------|-----------|
| F1 | Score drowns minority class | H(zᵢ) = mean over all pixels; with 99.63% background pixels, foreground signal is ~0 contribution |
| F2 | Biased pruning threshold | Global mean threshold is dominated by majority class — positive tiles unfairly compete against negative tiles |
| F3 | `protect_minority` suppresses effective pruning | Blocking ALL positive tiles from pruning drops effective pruning rate from ~50% to ~38% (only negatives pruned) |
| F4 | Pruning on noisy early scores | Scores in epochs 1–10 reflect random-init noise, not true sample difficulty; unreliable pruning decisions |
| F5 | Constant prune ratio from epoch 1 | No ramp-up: aggressive pruning starts before model has seen all samples even once |

---

## Algorithm: CS-IB (Class-Stratified InfoBatch with Warmup)

Four targeted mathematical fixes. No architectural changes. Drop-in replacement for InfoBatch.

---

### Fix 1 — Class-Balanced Score Function (addresses F1, F3)

**Current (broken for imbalanced segmentation):**
```
H(zᵢ) = (1/Nᵢ) Σⱼ lⱼ     ← mean over all pixels, foreground drowned
```

**Proposed:**
```
H(zᵢ) = wᶠᵍ · Lᶠᵍ(zᵢ)  +  wᵇᵍ · Lᵇᵍ(zᵢ)

where:
  Lᶠᵍ(zᵢ) = mean loss over foreground pixels in image i  (0 if none)
  Lᵇᵍ(zᵢ) = mean loss over background pixels in image i
  wᶠᵍ     = 1 / (2 · pᶠᵍ)       ← upweights rare foreground
  wᵇᵍ     = 1 / (2 · pᵇᵍ)       ← downweights dominant background
  pᶠᵍ     = global fraction of foreground pixels across training set
  pᵇᵍ     = 1 - pᶠᵍ
```

For TreeFinder: pᶠᵍ = 0.003722 → wᶠᵍ ≈ 134, wᵇᵍ ≈ 0.50

**Consequence for negative tiles** (no foreground pixels, Lᶠᵍ = 0):
```
H(zᵢ) = wᵇᵍ · Lᵇᵍ(zᵢ)     ← same as before, scaled
```

**Consequence for positive tiles:**
```
H(zᵢ) ≈ 134 · Lᶠᵍ(zᵢ) + 0.5 · Lᵇᵍ(zᵢ)   ← dominated by foreground loss
```

A positive tile is now only "easy" (prunable) when the model has actually learned its tree pixels — not just when the background is easy. `protect_minority` becomes unnecessary.

**Generalises to K classes:**
```
H(zᵢ) = Σₖ  (1 / (K · pₖ)) · Lₖ(zᵢ)    for each class k
```

---

### Fix 2 — Class-Stratified Pruning Threshold (addresses F2)

**Current (biased):**
```
T = mean{ H(zⱼ) : all j }     ← dominated by majority class
Prune zᵢ if H(zᵢ) < T
```

**Proposed:**
```
Tᵖᵒˢ = mean{ H(zⱼ) : cls_label[j] = 1 }   ← threshold for positive tiles
Tⁿᵉᵍ = mean{ H(zⱼ) : cls_label[j] = 0 }   ← threshold for negative tiles

Prune zᵢ if H(zᵢ) < T^{cls_label[i]}
```

Pruning is applied separately within each image-level class, maintaining approximately equal pruning rates across classes. The pruning pool is split: keep_ratio fraction of easy positives and keep_ratio fraction of easy negatives are retained and upweighted.

---

### Fix 3 — Three-Phase Schedule with Progressive Ramp (addresses F4, F5)

**Three phases (parameters: α = warmup fraction, δ = anneal fraction, ρ_max = max prune ratio):**

```
Phase 1 — Warmup      [epochs 1   →  ⌊αT⌋   ]   no pruning; scores accumulate via EMA
Phase 2 — CS-Pruning  [epochs ⌊αT⌋+1 → ⌊δT⌋ ]   class-stratified pruning with progressive ratio
Phase 3 — Annealing   [epochs ⌊δT⌋+1 → T    ]   full dataset; weights reset to 1.0
```

**Progressive prune ratio during Phase 2:**
```
ρ(t) = ρ_max · (t − αT) / (δT − αT)

  At t = αT:  ρ = 0       (no pruning, smooth entry into Phase 2)
  At t = δT:  ρ = ρ_max   (full configured pruning at Phase 2 end)
```

**Effective keep ratio passed to prune():**
```
keep_ratio(t) = 1 − ρ(t)
```

Defaults: α = 0.1, δ = 0.875 (same as current), ρ_max = 0.5 (same as current).
Effective pruning window: 77.5% of epochs (vs 87.5% in standard InfoBatch), but every pruning decision is trustworthy.

---

### Fix 4 — EMA Score Smoothing (addresses F4)

**Current:**
```
self.scores[i] ← h^t(zᵢ)     ← single-epoch raw score, high variance
```

**Proposed:**
```
self.scores[i] ← β · self.scores[i]  +  (1−β) · h^t(zᵢ)

β = 0.7   (half-life ≈ 2 epochs; scores settle well within α·T = 10 epochs of warmup)
```

During warmup, EMA accumulates reliable baseline scores. Phase 2 pruning starts with stable multi-epoch estimates rather than single-epoch noise.

**Note on β:** β = 0.7 is preferred over 0.9. With T=100, αT=10 epochs of warmup, β=0.7 reduces initial score bias to 0.7^10 ≈ 3% (negligible). β=0.9 leaves ~35% bias at warmup end.

---

## Summary: CS-IB vs Standard InfoBatch

| Flaw | Standard InfoBatch | CS-IB |
|------|--------------------|-------|
| Pixel-level imbalance in score | Ignored (mean over all pixels) | Class-balanced score wᶠᵍ·Lᶠᵍ + wᵇᵍ·Lᵇᵍ |
| Image-level imbalance in threshold | Single biased global mean | Separate threshold per image class |
| Positive tiles never pruned | `protect_minority` blocks all | Prunable when foreground loss is low |
| Effective pruning rate | ~38% (negatives only) | ~50% (both classes) |
| Score reliability in early epochs | Pruning on noisy init scores | Warmup phase defers pruning |
| Prune ratio ramp | Constant from epoch 1 | Linear ramp 0 → ρ_max over Phase 2 |
| Score stability | Raw single-epoch value | EMA β=0.7 |

---

## Implementation Plan

Files that change: `upd_info.py`, `exps/train.py`, `data_loader/__init__.py`, `configs/*.yaml`

---

### Step 1 — Update `InfoBatch.__init__` signature

**File:** `upd_info.py`

Replace the `minority_mask` parameter with `cls_labels` and add new parameters:

```python
def __init__(self, dataset, num_epochs, prune_ratio=0.5, delta=0.875,
             cls_labels=None, fg_pixel_fraction=0.003722,
             warmup_fraction=0.1, ema_beta=0.7):
```

- `cls_labels`: numpy bool array, shape (N,); True if tile contains any foreground pixel. Replaces `minority_mask`. Source: reuse `_compute_minority_mask()` output from `data_loader/__init__.py` (same logic, new name).
- `fg_pixel_fraction`: global fraction of foreground pixels across training set. Used to compute wᶠᵍ and wᵇᵍ. Accept from config; default matches TreeFinder EDA result.
- `warmup_fraction`: fraction of total epochs used as Phase 1 warmup.
- `ema_beta`: EMA decay for score smoothing.

Compute and store in `__init__`:
```python
self.cls_labels   = np.asarray(cls_labels, dtype=bool) if cls_labels is not None else None
self.w_fg         = 1.0 / (2.0 * fg_pixel_fraction)
self.w_bg         = 1.0 / (2.0 * (1.0 - fg_pixel_fraction))
self.warmup_epochs = max(1, int(warmup_fraction * num_epochs))
self.ema_beta     = ema_beta
```

Remove `minority_mask` attribute and its guard logic entirely.

**Off-by-one note:** `IBSampler.__init__` calls `self.reset()` once at construction before training begins. This consumes `iterations=0`. The first training-loop call to `sampler.reset()` runs at `iterations=1`. Account for this in the warmup check: prune only when `self.iterations > self.warmup_epochs` (strictly greater, matching the existing `> self.stop_prune` pattern).

---

### Step 2 — Add EMA to `InfoBatch.update()`

**File:** `upd_info.py`, `update()` method, currently line 133:

```python
# Current:
self.scores[indices.cpu().long()] = score_val.cpu()

# Replace with:
idx = indices.cpu().long()
self.scores[idx] = self.ema_beta * self.scores[idx] + (1.0 - self.ema_beta) * score_val.cpu()
```

This applies to both single-GPU and distributed paths (the distributed gather happens before this line, so the same replacement applies).

---

### Step 3 — Implement class-stratified threshold in `InfoBatch.prune()`

**File:** `upd_info.py`, `prune()` method. Replace entire method body.

New signature: `def prune(self, effective_keep_ratio=None):`

Logic:
1. Use `effective_keep_ratio` if provided, else fall back to `self.keep_ratio`.
2. If `cls_labels` is available: split indices into positive group (cls_labels=True) and negative group (cls_labels=False). Compute threshold separately for each group. Apply pruning within each group independently. Combine remained_indices.
3. If `cls_labels` is None: fall back to current global-mean threshold behaviour (backward compat).
4. Assign `1/effective_keep_ratio` weight to retained easy samples in each group.
5. Remove all `minority_mask` logic (replaced by class-stratified approach).

Pseudo-code:
```python
def prune(self, effective_keep_ratio=None):
    kr = effective_keep_ratio if effective_keep_ratio is not None else self.keep_ratio
    scores_np = self.scores.numpy()
    remained = []
    self.reset_weights()

    if self.cls_labels is not None:
        for group_flag in [False, True]:           # negatives first, then positives
            idx = np.where(self.cls_labels == group_flag)[0]
            if len(idx) == 0:
                continue
            group_scores = scores_np[idx]
            threshold = group_scores.mean()
            easy_mask = group_scores < threshold
            hard_idx  = idx[~easy_mask]
            easy_idx  = idx[easy_mask]
            remained.extend(hard_idx.tolist())
            if len(easy_idx) > 0:
                n_keep = max(0, int(kr * len(easy_idx)))
                kept = np.random.choice(easy_idx, n_keep, replace=False)
                if len(kept) > 0:
                    self.weights[kept] = 1.0 / kr
                    remained.extend(kept.tolist())
    else:
        # fallback: original global-threshold logic (no cls_labels provided)
        threshold = scores_np.mean()
        easy_idx  = np.where(scores_np < threshold)[0]
        hard_idx  = np.where(scores_np >= threshold)[0]
        remained.extend(hard_idx.tolist())
        n_keep = int(kr * len(easy_idx))
        kept = np.random.choice(easy_idx, n_keep, replace=False)
        if len(kept) > 0:
            self.weights[kept] = 1.0 / kr
            remained.extend(kept.tolist())

    self.num_pruned_samples += len(self.dataset) - len(remained)
    np.random.shuffle(remained)
    return remained
```

---

### Step 4 — Implement three-phase schedule in `IBSampler.reset()`

**File:** `upd_info.py`, `IBSampler.reset()` method.

```python
def reset(self):
    np.random.seed(self.iterations)
    warmup_stop = self.dataset.warmup_epochs      # Phase 1 end
    prune_stop  = self.stop_prune                 # Phase 2 end (= delta * num_epochs)

    if self.iterations <= warmup_stop:
        # Phase 1: warmup — full dataset, no pruning, scores accumulate via EMA
        self.sample_indices = self.dataset.no_prune()

    elif self.iterations <= prune_stop:
        # Phase 2: CS-pruning with progressive ratio ramp
        #   ρ(t) = ρ_max * (t - warmup_stop) / (prune_stop - warmup_stop)
        #   keep_ratio(t) = 1 - ρ(t)
        ramp = (self.iterations - warmup_stop) / max(1, prune_stop - warmup_stop)
        prune_ratio_max = 1.0 - self.dataset.keep_ratio   # ρ_max from config
        effective_prune = prune_ratio_max * ramp
        effective_keep  = 1.0 - effective_prune
        effective_keep  = max(self.dataset.keep_ratio, effective_keep)  # floor at keep_ratio
        self.sample_indices = self.dataset.prune(effective_keep_ratio=effective_keep)

    else:
        # Phase 3: annealing — full dataset
        if self.iterations == int(prune_stop) + 1:
            self.dataset.reset_weights()
        self.sample_indices = self.dataset.no_prune()

    self.iter_obj = iter(self.sample_indices)
    self.iterations += 1
```

---

### Step 5 — Update score computation in `train.py`

**File:** `exps/train.py`, inside the `if use_infobatch:` block (currently lines 218–239).

Replace `per_img_score` computation with the class-balanced score:

```python
# Existing: per_img_bce (mean over all valid pixels) — keep for gradient rescaling
valid_pixels_per_img = valid_mask.view(B, -1).sum(dim=1).clamp(min=1)
per_img_bce = (raw_loss * valid_mask).view(B, -1).sum(dim=1) / valid_pixels_per_img

# New: class-balanced score H(zᵢ) = w_fg * L_fg + w_bg * L_bg
fg_mask  = (labels > 0.5).float() * valid_mask          # foreground valid pixels
bg_mask  = (labels <= 0.5).float() * valid_mask         # background valid pixels
n_fg     = fg_mask.view(B, -1).sum(dim=1).clamp(min=1)
n_bg     = bg_mask.view(B, -1).sum(dim=1).clamp(min=1)
L_fg     = (raw_loss * fg_mask).view(B, -1).sum(dim=1) / n_fg
L_bg     = (raw_loss * bg_mask).view(B, -1).sum(dim=1) / n_bg
# Zero out L_fg contribution for negative tiles (no foreground pixels present)
has_fg   = (fg_mask.view(B, -1).sum(dim=1) > 0).float()
w_fg     = train_dataset.w_fg   # pre-computed in InfoBatch.__init__
w_bg     = train_dataset.w_bg
per_img_score = w_fg * L_fg * has_fg + w_bg * L_bg

# Pass class-balanced score as H(z); per_img_bce used for gradient rescaling only
bce_loss = train_dataset.update(per_img_bce, scores=per_img_score)
```

The dice score path (when w_dice > 0) stays the same — it contributes to `per_img_score` via addition after the class-balanced BCE score, same as the current `per_img_score = per_img_bce + w_dice * per_img_dice` pattern.

---

### Step 6 — Update `data_loader/__init__.py`

**File:** `data_loader/__init__.py`

1. Rename `_compute_minority_mask` → `_compute_cls_labels` (same logic, same output, clearer name). Keep the old name as an alias if needed for backward compat.

2. In `get_dataloader()`, replace InfoBatch construction:

```python
# Old:
protect_minority = ib_cfg.get('protect_minority', False)
minority_mask = _compute_cls_labels(train_ds) if protect_minority else None
train_ds = InfoBatch(train_ds, num_epochs=num_epochs, prune_ratio=prune_ratio,
                     delta=delta, minority_mask=minority_mask)

# New:
cls_labels = _compute_cls_labels(train_ds)   # always compute; used for stratified threshold
fg_pix_frac   = ib_cfg.get('fg_pixel_fraction', 0.003722)
warmup_frac   = ib_cfg.get('warmup_fraction', 0.1)
ema_beta      = ib_cfg.get('ema_beta', 0.7)
train_ds = InfoBatch(
    train_ds,
    num_epochs=num_epochs,
    prune_ratio=prune_ratio,
    delta=delta,
    cls_labels=cls_labels,
    fg_pixel_fraction=fg_pix_frac,
    warmup_fraction=warmup_frac,
    ema_beta=ema_beta,
)
```

Remove the `protect_minority` config read and the conditional guard.

---

### Step 7 — Update config files

**Files:** `configs/debug.yaml` and all `configs/res_*.yaml`

Under the `infobatch:` block, make these changes:

```yaml
infobatch:
  enabled: true
  prune_ratio: 0.5
  delta: 0.875
  warmup_fraction: 0.1        # NEW: Phase 1 length as fraction of total epochs
  ema_beta: 0.7               # NEW: EMA decay for score smoothing
  fg_pixel_fraction: 0.003722 # NEW: global foreground pixel fraction (from EDA)
  # protect_minority: REMOVE  # no longer needed; handled by class-stratified logic
```

---

### Step 8 — Validation Checklist

After implementation, verify the following before running full experiments:

1. **Warmup produces no pruning**: Add a print/assert in `IBSampler.reset()`. For the first `warmup_epochs` calls, `len(sample_indices)` must equal `len(dataset)`.

2. **Progressive ramp is monotonic**: Log `effective_keep_ratio` each epoch. Should decrease from 1.0 to `keep_ratio` over Phase 2.

3. **Class-stratified thresholds differ**: Log `Tᵖᵒˢ` and `Tⁿᵉᵍ` in `prune()`. They should diverge once EMA scores have settled.

4. **Positive tiles are now prunable**: Log pruned count per class. Expect some positive tiles pruned in mid-to-late Phase 2 (unlike current where 0 positives are ever pruned).

5. **EMA settling**: Log mean and std of `self.scores` each epoch. Should be noisy in epochs 1–5, stable by epoch ~8.

6. **Integration smoke test**: Run `configs/debug.yaml` for 20 epochs. Check that:
   - Epochs 1–10: pruned_count = 0 in CSV
   - Epochs 11–15: pruned_count increases gradually
   - Epochs 16–17.5 (87.5% of 20): near-max pruning
   - Epochs 18–20: pruned_count = 0 again (annealing)

---

## Files Modified Summary

| File | Change |
|------|--------|
| `upd_info.py` | New `__init__` params; EMA in `update()`; stratified threshold in `prune(effective_keep_ratio)`; three-phase schedule in `IBSampler.reset()` |
| `exps/train.py` | Replace per_img_score with class-balanced H(zᵢ) = w_fg·L_fg + w_bg·L_bg |
| `data_loader/__init__.py` | Rename fn, always compute cls_labels, pass new params to InfoBatch, remove protect_minority |
| `configs/*.yaml` | Add warmup_fraction, ema_beta, fg_pixel_fraction; remove protect_minority |

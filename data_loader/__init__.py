import json
import math
import os
import numpy as np
from torch.utils.data import DataLoader


def _compute_cls_labels(dataset):
    """Return a boolean numpy array of shape (N,) where True means the tile
    contains at least one positive (tree) pixel.

    Iterated once before training begins. Used by CS-IB for class-stratified
    pruning thresholds: positive tiles and negative tiles are pruned against
    their own per-class mean, preventing the majority class from dominating.
    """
    import torch
    labels = np.zeros(len(dataset), dtype=bool)
    for i in range(len(dataset)):
        label = dataset[i]['label']
        if isinstance(label, torch.Tensor):
            labels[i] = label.any().item()
        else:
            labels[i] = np.any(label > 0)
    return labels


def get_dataloader(cfg: dict, train_fraction_seed: int = None):
    """Return train, val, test DataLoaders and the train dataset (possibly InfoBatch-wrapped).

    Args:
        cfg: full config dict
        train_fraction_seed: seed used to subsample train_fraction from the train split.
            Defaults to cfg['experiment']['seed'] when None.
    """
    method = cfg["data"]["split"]["method"]
    if method == "random":
        from .random_split import RandomSplitDataset as DS
    elif method == "by_state":
        from .by_state_split import StateSplitDataset as DS
    elif method == "by_climate":
        from .by_climate_split import ClimateSplitDataset as DS
    elif method == "by_tree_type":
        from .by_tree_split import TreeSplitDataset as DS
    else:
        raise ValueError(f"Unknown split method: {method}")

    train_ds = DS(cfg, split="train")
    val_ds   = DS(cfg, split="val")
    test_ds  = DS(cfg, split="test")

    # Optionally subsample the training set (val/test are always kept full)
    train_fraction = cfg['data'].get('train_fraction', 1.0)
    if 0.0 < train_fraction < 1.0:
        from torch.utils.data import Subset
        n_total = len(train_ds)
        n_keep = max(1, math.floor(n_total * train_fraction))
        seed = train_fraction_seed if train_fraction_seed is not None else cfg['experiment']['seed']

        os.makedirs('splits', exist_ok=True)
        split_save_path = os.path.join('splits', f'trainfrac_{train_fraction:.2f}_seed{seed}.json')

        if os.path.exists(split_save_path):
            # Reuse the existing split — guarantees identical data across runs
            with open(split_save_path) as f:
                split_payload = json.load(f)
            indices = split_payload['indices']
            # Sanity-check: dataset size must match what was recorded
            if split_payload['n_total'] != n_total:
                raise ValueError(
                    f"[TrainFraction] Cached split at {split_save_path} was built on "
                    f"{split_payload['n_total']} samples but current dataset has {n_total}. "
                    f"Delete the file to regenerate."
                )
            print(f"[TrainFraction] Loaded existing split from {split_save_path} "
                  f"({len(indices)}/{n_total} samples, seed={seed})")
        else:
            # First run: generate and save the split
            rng = np.random.RandomState(seed)
            indices = rng.permutation(n_total)[:n_keep].tolist()
            split_payload = {
                'train_fraction': train_fraction,
                'seed': seed,
                'n_total': n_total,
                'n_selected': n_keep,
                'indices': indices,
            }
            try:
                split_payload['paths'] = [train_ds.paths[i] for i in indices]
            except AttributeError:
                pass  # dataset type doesn't expose .paths
            with open(split_save_path, 'w') as f:
                json.dump(split_payload, f, indent=2)
            print(f"[TrainFraction] Created new split, saved to {split_save_path} "
                  f"({n_keep}/{n_total} samples, seed={seed})")

        train_ds = Subset(train_ds, indices)

    # Optionally wrap train dataset with InfoBatch
    ib_cfg = cfg.get('infobatch', {})
    use_infobatch = ib_cfg.get('enabled', False)
    train_sampler = None
    train_shuffle = True

    if use_infobatch:
        print("Using CS-IB (Class-Stratified InfoBatch) for training")
        from upd_info import InfoBatch
        num_epochs = cfg['training']['epochs']
        prune_ratio  = ib_cfg.get('prune_ratio', 0.5)
        delta        = ib_cfg.get('delta', 0.875)
        fg_pix_frac  = ib_cfg.get('fg_pixel_fraction', 0.003722)
        warmup_frac  = ib_cfg.get('warmup_fraction', 0.1)
        ema_beta     = ib_cfg.get('ema_beta', 0.7)
        # Always compute cls_labels — used for class-stratified pruning thresholds
        cls_labels = _compute_cls_labels(train_ds)
        n_pos = int(cls_labels.sum())
        print(f"[CS-IB] cls_labels computed: {n_pos}/{len(train_ds)} positive tiles "
              f"({100*n_pos/max(len(train_ds),1):.1f}%)")
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
        train_sampler = train_ds.sampler
        train_shuffle = False  # sampler handles ordering

    num_workers = cfg['experiment'].get("num_workers", 16)
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training'].get("batch_size", 32),
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=persistent)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training'].get("batch_size", 32),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=persistent)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg['training'].get("batch_size", 32),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=persistent)

    return train_loader, val_loader, test_loader, train_ds

#!/usr/bin/env python3
"""
main.py: Training and evaluation pipeline for dead tree dataset.

This script supports flexible train-test splits (random, geographic, state-based) and multiple model backbones (CNNs, ViTs, SegFormer, etc.).
Usage:
    python main.py --config configs/debug.yaml --exp_name exp1

Recommendation: use a YAML config (e.g., configs.yaml) to centralize parameters and ensure reproducibility, following common ML conventions.
"""


import yaml
import logging
from pathlib import Path
import time

from utils.tools import (
    parse_args, load_config, overwrite_config,
    setup_logging, set_seed, save_results
)
from data_loader import get_dataloader
from models import get_model
from exps.train import train_model
from exps.evaluate import evaluate_model


def main():
    # Parse command-line args and load config
    args = parse_args()
    cfg = load_config(args.config)
    if args.overwrite_cfg:
        cfg = overwrite_config(args, cfg)

    base_seed = cfg['experiment']['seed']
    num_splits = cfg['data'].get('num_train_splits', 1)

    exp_name_base = f"exp{cfg['experiment']['id'].zfill(3)}_{cfg['model']['name']}"

    # Log experiment configuration once
    logger = setup_logging(cfg['logging']['log_dir'], exp_name_base)
    logger.info(f"Starting experiment: {exp_name_base}")
    logger.info(f"Number of train splits: {num_splits}")
    logger.info("Configuration:\n" + yaml.dump(cfg, sort_keys=False))

    all_run_results = {}

    for run_idx in range(num_splits):
        run_seed = base_seed + run_idx
        run_suffix = f"_run{run_idx + 1}" if num_splits > 1 else ""
        exp_name = exp_name_base + run_suffix

        logger.info(
            f"=== Run {run_idx + 1}/{num_splits} | exp: {exp_name} | train_fraction_seed: {run_seed} ==="
        )

        # Set seed for model init and training dynamics
        set_seed(run_seed)

        # Build data loaders — test split is the same across all runs (fixed by base_seed split file)
        train_loader, val_loader, test_loader, train_dataset = get_dataloader(
            cfg, train_fraction_seed=run_seed
        )

        # Initialize a fresh model for each run
        model = get_model(cfg['model'])

        # Train or skip training
        if args.eval_only:
            logger.info("Evaluation only mode: Skipping training.")
            train_metrics = {}
            train_time = 0.0
        else:
            t0 = time.time()
            train_metrics = train_model(
                model,
                train_loader,
                val_loader,
                train_dataset,
                cfg,
                exp_name
            )
            train_time = time.time() - t0
            logger.info(f"Training completed in {train_time/3600:.2f}h ({train_time/60:.1f}m).")

        # Evaluate model on the fixed test set
        t0 = time.time()
        eval_metrics = evaluate_model(
            model,
            test_loader,
            cfg,
            exp_name
        )
        eval_time = time.time() - t0
        logger.info(f"Evaluation completed in {eval_time/3600:.2f}h ({eval_time/60:.1f}m).")

        # Combine and save per-run metrics
        results_root = Path(cfg['output']['results_dir']) / exp_name
        all_metrics = {
            **train_metrics,
            **eval_metrics,
            'train_fraction_seed': run_seed,
            'train_time_h': train_time,
            'eval_time_h': eval_time,
        }
        save_results(all_metrics, exp_name, str(results_root))
        logger.info(f"Run {run_idx + 1} completed. Metrics: {all_metrics}")

        all_run_results[exp_name] = all_metrics

    # Save cross-run comparison summary when more than one split
    if num_splits > 1:
        summary_dir = Path(cfg['output']['results_dir']) / exp_name_base
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"{exp_name_base}_split_comparison.yaml"
        with open(summary_path, 'w') as f:
            yaml.dump(all_run_results, f, sort_keys=False)
        logger.info(f"Split comparison summary saved to: {summary_path}")

    # Auto-upload outputs to Google Drive (if enabled in config)
    gdrive_cfg = cfg.get('gdrive', {})
    if gdrive_cfg.get('enabled', False):
        try:
            from utils.gdrive_upload import upload_experiment
            upload_experiment(exp_name_base, cfg, gdrive_cfg)
        except Exception as e:
            logger.warning(f"[GDrive] Unexpected error during upload: {e}")
    else:
        logger.info("[GDrive] Upload skipped (gdrive.enabled is false or not configured).")


if __name__ == "__main__":
    main()

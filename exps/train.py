import os
import csv
import time
import logging
import itertools
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.ops import sigmoid_focal_loss

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets, mask=None):
        probs = torch.sigmoid(logits)          # shape (B,1,H,W)
        if mask is not None:
            probs = probs * mask
            
        p_flat = probs.view(-1)
        t_flat = targets.float().view(-1)
        
        inter = (p_flat * t_flat).sum()
        union = p_flat.sum() + t_flat.sum()
        dice = (inter + self.eps) / (union + self.eps)
        return 1 - dice
    
class FocalLoss(nn.Module):
  def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction  # 'none' is recommended for external masking

  def forward(self, logits, targets):
    return sigmoid_focal_loss(
      inputs=logits,
      targets=targets.float(),
      alpha=self.alpha,
      gamma=self.gamma,
      reduction=self.reduction
    )
    

def train_model(model, train_loader, val_loader, train_dataset, cfg, exp_name):
    """
    Train the model with validation and early stopping.

    Args:
      model: nn.Module
      train_loader: DataLoader for training
      val_loader: DataLoader for validation
      train_dataset: raw or InfoBatch-wrapped training dataset
      cfg: full config dict
      exp_name: experiment identifier

    Returns:
      dict containing best monitored metric
    """
    logger = logging.getLogger(__name__)
    # Device
    gpu_id = cfg['experiment'].get('gpu_id', 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    # Determine if InfoBatch is active
    use_infobatch = cfg.get('infobatch', {}).get('enabled', False)
    print("-----------------------------------------")
    print(f"Using infobatch: {use_infobatch}")
    print("-----------------------------------------")
    # Unpack training cfg
    tr_cfg = cfg['training']
    criterion_type = tr_cfg['criterion'].get('type', 'BCEWithLogitsLoss')
    if criterion_type == 'BCEWithLogitsLoss':
        if tr_cfg['criterion'].get('w_pos', 0) > 0:
            pos_weight = torch.tensor(tr_cfg['criterion']['w_pos'], device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        else:
            criterion = nn.BCEWithLogitsLoss(reduction='none')
    elif criterion_type == 'FocalLoss':
        alpha = tr_cfg['criterion'].get('alpha', 0.25)
        gamma = tr_cfg['criterion'].get('gamma', 2.0)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unsupported loss type: {criterion_type}")
    dice_loss = DiceLoss()
    w_dice = tr_cfg.get('w_dice', 0.5)
    
    optimizer = getattr(optim, tr_cfg['optimizer']['type'])(
        model.parameters(),
        lr=float(tr_cfg['learning_rate']),
        weight_decay=tr_cfg['optimizer'].get('weight_decay', 0)
    )
    scheduler = None
    if 'scheduler' in tr_cfg:
        sc = tr_cfg['scheduler']        
        sched_type = sc['type']
        SchedulerClass = getattr(optim.lr_scheduler, sched_type)
        
        if sched_type == 'ExponentialLR':
            scheduler = SchedulerClass(
            optimizer,
            gamma=sc.get('gamma', 0.95)
        )
        elif sched_type == 'StepLR':
            scheduler = SchedulerClass(
            optimizer,
            step_size=sc.get('step_size', 10),
            gamma=sc.get('gamma', 0.1)
        )
        elif sched_type == 'CosineAnnealingWarmRestarts':
            scheduler = SchedulerClass(
            optimizer,
            T_0=sc.get('T_0', 10),
            T_mult=sc.get('T_mult', 1),
            eta_min=float(sc.get('eta_min', 0))
        )
        else:
            raise ValueError(f"Unsupported scheduler type: {sched_type}")


    # Early stopping config
    es_cfg = tr_cfg['early_stopping']
    es_enabled = es_cfg.get('enabled', False)
    monitor = es_cfg.get('monitor', 'val_loss')
    mode = es_cfg.get('mode', 'min')
    patience = es_cfg.get('patience', 5)
    if mode == 'min':
        best_val = float('inf')
        improve = lambda cur, best: cur < best
    else:
        best_val = -float('inf')
        improve = lambda cur, best: cur > best
    epochs_no_improve = 0

    # Checkpoint directory
    ckpt_root = Path(cfg['output']['checkpoint_dir'])
    ckpt_dir = ckpt_root / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results_root = Path(cfg['output']['results_dir'])
    results_dir = results_root / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = tr_cfg['epochs']
    log_interval = cfg['logging'].get('log_interval', 100)
    
    best_metric = None
    train_losses = []
    val_losses = []

    # CSV metrics file — written incrementally, one row per epoch
    csv_path = results_dir / f"{exp_name}_metrics.csv"
    csv_fields = [
        'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc',
        'train_bce', 'train_dice', 'val_bce', 'val_dice',
        'epoch_time_s', 'pruned_count'
    ]
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()
    logger.info(f"Metrics CSV will be saved to: {csv_path}")

    # Total training time tracker
    train_total_start = time.time()

    # Training loop
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        batch_size = cfg['training'].get('batch_size', 32)
        # Explicitly call reset() once per epoch so prune() runs exactly once
        # with this epoch's scores. __iter__ on IBSampler no longer calls
        # reset(), preventing the spurious prune() calls that PyTorch triggers
        # internally when it re-creates the sampler iterator mid-epoch.
        if use_infobatch:
            train_loader.sampler.reset()
        train_iter = iter(train_loader)
        total_steps = len(train_loader)
        logger.info("-----------------------------------------")
        logger.info(f"Using infobatch: {use_infobatch}")
        logger.info("-----------------------------------------")
        logger.info(f"Epoch {epoch} - Starting training with ~{total_steps} steps.")

        # Training
        model.train()
        train_loss_total = 0.0
        train_loss_bce   = 0.0
        train_loss_dice  = 0.0
        train_correct    = 0
        train_total_px   = 0
        step_times   = []
        for step, batch in enumerate(itertools.islice(train_iter, total_steps), 1):
            step_start = time.time()

            # InfoBatch monkey-patches DataLoader.__next__ to extract indices
            # automatically via set_active_indices(), so the batch here is already
            # the plain collated sample dict — no manual unpacking needed.

            imgs = batch['image'].to(device)
            labels = batch['label'].unsqueeze(1).float().to(device)
            no_data = batch['no_data_mask'].unsqueeze(1).to(device)
            valid_mask = (~no_data).float()
            
            optimizer.zero_grad()
            outputs = model(imgs)
            
            raw_loss = criterion(outputs, labels) # (B,1,H,W)

            if use_infobatch:
                # Per-image BCE score: average focal loss over valid pixels.
                valid_pixels_per_img = valid_mask.view(valid_mask.shape[0], -1).sum(dim=1).clamp(min=1)
                per_img_bce = (raw_loss * valid_mask).view(raw_loss.shape[0], -1).sum(dim=1) / valid_pixels_per_img

                if w_dice > 0:
                    dloss = dice_loss(outputs, labels, valid_mask)
                    # Per-image dice score for H(z) = BCE + w_dice * dice (spec §1).
                    B = outputs.shape[0]
                    p_flat = (torch.sigmoid(outputs) * valid_mask).view(B, -1)
                    t_flat = (labels * valid_mask).view(B, -1)
                    eps = dice_loss.eps
                    inter = (p_flat * t_flat).sum(dim=1)
                    union = p_flat.sum(dim=1) + t_flat.sum(dim=1)
                    per_img_dice = 1 - (inter + eps) / (union + eps)
                    per_img_score = per_img_bce + w_dice * per_img_dice
                else:
                    dloss = torch.tensor(0.0, device=device)
                    per_img_score = per_img_bce

                # Pass combined H(z) as the pruning score; BCE component is
                # used for gradient rescaling so the dice term stays separate.
                bce_loss = train_dataset.update(per_img_bce, scores=per_img_score)
            else:
                bce_loss = (raw_loss * valid_mask).sum() / valid_mask.sum()
                if w_dice > 0:
                    dloss = dice_loss(outputs, labels, valid_mask)
                else:
                    dloss = torch.tensor(0.0, device=device)
                
            total_loss = bce_loss + w_dice * dloss
            total_loss.backward()
            optimizer.step()
            
            train_loss_total += total_loss.item()
            train_loss_bce += bce_loss.item()
            train_loss_dice += dloss.item()
            step_times.append(time.time() - step_start)

            # Track training accuracy
            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).long()
                gt = labels.long()
                flat_valid = valid_mask.bool().view(-1)
                train_correct += (preds.view(-1)[flat_valid] == gt.view(-1)[flat_valid]).sum().item()
                train_total_px += flat_valid.sum().item()
            
            if step % log_interval == 0:
                avg_total = train_loss_total / step
                avg_bce = train_loss_bce / step
                avg_dice = train_loss_dice / step
                log_str = (
                    f"[Epoch {epoch}/{num_epochs}] "
                    f"Step {step}/{total_steps} - "
                    f"Time Spent: {sum(step_times)/60:.1f}m - "
                    f"Train Total Loss: {avg_total:.4f}"
                )
                if w_dice > 0:
                    log_str += f" - BCE Loss: {avg_bce:.4f} - Dice Loss: {avg_dice:.4f}"
                logger.info(log_str)
                step_times = []

        avg_train = train_loss_total / len(train_loader)
        avg_train_bce = train_loss_bce / len(train_loader)
        avg_train_dice = train_loss_dice / len(train_loader)
        train_acc = train_correct / max(train_total_px, 1)
        train_losses.append(avg_train)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_loss_bce = 0.0
        val_loss_dice = 0.0
        val_correct = 0
        val_total_px = 0
        val_time = time.time()
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                labels = batch['label'].unsqueeze(1).float().to(device)
                no_data = batch['no_data_mask'].unsqueeze(1).to(device)
                valid_mask = (~no_data).float()
                
                outputs = model(imgs)
                
                raw_loss = criterion(outputs, labels)
                bce_loss = (raw_loss * valid_mask).sum() / valid_mask.sum()
                if w_dice > 0:
                    dloss = dice_loss(outputs, labels, valid_mask)
                else:
                    dloss = torch.tensor(0.0, device=device)
                total_loss = bce_loss + w_dice * dloss
                
                val_loss_total += total_loss.item()
                val_loss_bce += bce_loss.item()
                val_loss_dice += dloss.item()

                # Track validation accuracy
                preds = (torch.sigmoid(outputs) > 0.5).long()
                gt = labels.long()
                flat_valid = valid_mask.bool().view(-1)
                val_correct += (preds.view(-1)[flat_valid] == gt.view(-1)[flat_valid]).sum().item()
                val_total_px += flat_valid.sum().item()
                
        avg_val = val_loss_total / len(val_loader)
        avg_val_bce = val_loss_bce / len(val_loader)
        avg_val_dice = val_loss_dice / len(val_loader)
        val_acc = val_correct / max(val_total_px, 1)
        
        val_losses.append(avg_val)
        val_time = time.time() - val_time
        log_str = (
            f"[Epoch {epoch}/{num_epochs}] "
            f"Time Spent: {val_time/60:.1f}m - "
            f"Avg Val Total Loss: {avg_val:.4f} - "
            f"Acc: {val_acc:.4f}"
            )
        if w_dice > 0:
            log_str += f" - BCE Loss: {avg_val_bce:.4f} - Dice Loss: {avg_val_dice:.4f}"
        logger.info(log_str)

        # Save best weights and early stopping check
        current = avg_val if monitor == 'val_loss' else None
        if current is not None and improve(current, best_val):
            best_val = current
            best_path = ckpt_dir / f"{exp_name}_best.pth"
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best model (val_loss={best_val:.4f}) saved: {best_path}")
            best_metric = best_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if es_enabled:
                logger.info(f"No improvement in val_loss for {epochs_no_improve}/{patience} epochs.")
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

        # Scheduler step
        if scheduler:
            scheduler.step()

        # Disable epoch checkpoint for saving memory
        # torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pth") 
        
        epoch_time = time.time() - epoch_start
        pruned_count = train_dataset.get_pruned_count() if use_infobatch else 0
        log_str = f"Epoch {epoch} done in {epoch_time/60:.1f}m - Train: {avg_train:.4f}, Val: {avg_val:.4f}"
        if use_infobatch:
            log_str += f" - InfoBatch total pruned so far: {pruned_count}"
        logger.info(log_str)

        # Write epoch row to CSV
        csv_writer.writerow({
            'epoch': epoch,
            'train_loss': round(avg_train, 6),
            'val_loss': round(avg_val, 6),
            'train_acc': round(train_acc, 6),
            'val_acc': round(val_acc, 6),
            'train_bce': round(avg_train_bce, 6),
            'train_dice': round(avg_train_dice, 6),
            'val_bce': round(avg_val_bce, 6),
            'val_dice': round(avg_val_dice, 6),
            'epoch_time_s': round(epoch_time, 2),
            'pruned_count': pruned_count,
        })
        csv_file.flush()  # write to disk immediately so it's readable mid-training

        
    # Close CSV and log total training time
    csv_file.close()
    total_train_time = time.time() - train_total_start
    logger.info(
        f"Total training time: {total_train_time/3600:.2f}h "
        f"({total_train_time/60:.1f}m) over {epoch} epoch(s). "
        f"Metrics CSV saved to: {csv_path}"
    )

    # plot loss curve
    try:
        epochs = list(range(1, epoch + 1))
        plt.figure()
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        loss_curve_path = results_dir / f"{exp_name}_loss_curve.png"
        plt.savefig(loss_curve_path)
        plt.close()
        logger.info(f"Loss curve saved to {loss_curve_path}")
    except Exception as e:
        logger.warning(f"Failed to plot loss curve: {e}")

    return {f"best_{monitor}": best_metric}

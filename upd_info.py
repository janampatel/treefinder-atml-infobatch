import math
import numpy as np
from typing import Iterator, Optional
import torch
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data import Dataset, _DatasetKind
from torch.utils.data.distributed import DistributedSampler
from operator import itemgetter
import torch.distributed as dist
import warnings

__all__ = ['InfoBatch']


def info_hack_indices(self):
    with torch.autograd.profiler.record_function(self._profile_name):
        if self._sampler_iter is None:
            # TODO(https://github.com/pytorch/pytorch/issues/76750)
            self._reset()  # type: ignore[call-arg]
        if isinstance(self._dataset, InfoBatch):
            indices, data = self._next_data()
        else:
            data = self._next_data()
        self._num_yielded += 1
        if self._dataset_kind == _DatasetKind.Iterable and \
                self._IterableDataset_len_called is not None and \
                self._num_yielded > self._IterableDataset_len_called:
            warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                        "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
                                                                self._num_yielded)
            if self._num_workers > 0:
                warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                                "IterableDataset replica at each worker. Please see "
                                "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
            warnings.warn(warn_msg)
        if isinstance(self._dataset, InfoBatch):
            self._dataset.set_active_indices(indices)
        return data


_BaseDataLoaderIter.__next__ = info_hack_indices


@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=dim)
    return output


class InfoBatch(Dataset):
    """
    InfoBatch aims to achieve lossless training speed up by randomly prunes a portion of less informative samples
    based on the loss distribution and rescales the gradients of the remaining samples to approximate the original
    gradient. See https://arxiv.org/pdf/2303.04947.pdf

    .. note::.
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for training.
        num_epochs (int): The number of epochs for pruning.
        prune_ratio (float, optional): The proportion of samples being pruned during training.
        delta (float, optional): The first delta * num_epochs the pruning process is conducted. It should be close to 1. Defaults to 0.875.
    """

    def __init__(self, dataset: Dataset, num_epochs: int,
                 prune_ratio: float = 0.5, delta: float = 0.875,
                 cls_labels=None, fg_pixel_fraction: float = 0.003722,
                 warmup_fraction: float = 0.1, ema_beta: float = 0.7):
        self.dataset = dataset
        self.keep_ratio = min(1.0, max(1e-1, 1.0 - prune_ratio))
        self.num_epochs = num_epochs
        self.delta = delta
        # self.scores stores the loss value H(z) of each sample. Initialised to
        # 1.0 so warmup epochs see a uniform distribution (no pruning fires).
        self.scores = torch.ones(len(self.dataset))
        self.weights = torch.ones(len(self.dataset))
        self.num_pruned_samples = 0
        self.cur_batch_index = None

        # CS-IB additions
        # cls_labels[i] = True  →  tile i contains at least one foreground pixel
        self.cls_labels = np.asarray(cls_labels, dtype=bool) if cls_labels is not None else None
        # Class-balanced score weights: w_fg upweights rare foreground signal
        self.w_fg = 1.0 / (2.0 * max(fg_pixel_fraction, 1e-6))
        self.w_bg = 1.0 / (2.0 * max(1.0 - fg_pixel_fraction, 1e-6))
        # Warmup: Phase 1 lasts warmup_epochs epochs with no pruning
        self.warmup_epochs = max(1, int(warmup_fraction * num_epochs))
        self.ema_beta = ema_beta

        print(f"[CS-IB] warmup_epochs={self.warmup_epochs}, stop_prune={int(self.stop_prune)}, "
              f"w_fg={self.w_fg:.2f}, w_bg={self.w_bg:.4f}, ema_beta={self.ema_beta}")

    def __getattr__(self, name):
        # Guard against recursion during pickle/unpickle in multiprocessing workers.
        # Before __init__ completes, self.dataset isn't in __dict__ yet, so accessing
        # it here would call __getattr__ again indefinitely.
        if name == 'dataset':
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def set_active_indices(self, cur_batch_indices: torch.Tensor):
        self.cur_batch_index = cur_batch_indices

    def update(self, values, scores=None):
        """Update sample scores and return the weighted mean loss for backprop.

        Args:
            values: Per-image loss tensor used for gradient rescaling (backprop).
            scores: Optional per-image score tensor used for pruning decisions
                    (H(z) in the paper). If None, ``values`` is used as the score.
                    Pass a combined loss (e.g. BCE + w_dice * dice) here while
                    keeping ``values`` as the BCE-only component so that each
                    loss term can still be rescaled independently.
        """
        assert isinstance(values, torch.Tensor)
        batch_size = values.shape[0]
        assert len(self.cur_batch_index) == batch_size, 'not enough index'
        device = values.device
        weights = self.weights[self.cur_batch_index].to(device)
        indices = self.cur_batch_index.to(device)
        score_val = (scores if scores is not None else values).detach().clone()
        self.cur_batch_index = []

        if dist.is_available() and dist.is_initialized():
            iv = torch.cat([indices.view(1, -1), score_val.view(1, -1)], dim=0)
            iv_whole_group = concat_all_gather(iv, 1)
            indices = iv_whole_group[0]
            score_val = iv_whole_group[1]
        # EMA smoothing: reduces noise from single-epoch scores
        idx = indices.cpu().long()
        self.scores[idx] = self.ema_beta * self.scores[idx] + (1.0 - self.ema_beta) * score_val.cpu()
        return (values * weights).mean()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # self.cur_batch_index.append(index)
        return index, self.dataset[index] # , index
        # return self.dataset[index], index, self.scores[index]

    def __getitems__(self, indices):
        # PyTorch >= 2.x's _MapDatasetFetcher checks for __getitems__ and calls it
        # directly if present (bypassing __getitem__). We must override it here to
        # ensure the (index, data) tuple format that info_hack_indices expects.
        return [self.__getitem__(idx) for idx in indices]

    def prune(self, effective_keep_ratio=None):
        """Class-stratified pruning with progressive keep ratio.

        Applies pruning independently within positive tiles (cls_labels=True)
        and negative tiles (cls_labels=False), so both classes contribute to
        the pruning pool and class imbalance does not bias the threshold.

        When cls_labels is None falls back to the original global-mean logic.
        """
        kr = effective_keep_ratio if effective_keep_ratio is not None else self.keep_ratio
        scores_np = self.scores.numpy()
        remained = []
        self.reset_weights()

        if self.cls_labels is not None:
            pruned_pos = 0
            pruned_neg = 0
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
                    pruned_this = len(easy_idx) - n_keep
                    if group_flag:
                        pruned_pos += pruned_this
                    else:
                        pruned_neg += pruned_this
            print(f"[CS-IB prune] kr={kr:.3f} | pruned pos={pruned_pos} neg={pruned_neg} "
                  f"| remained={len(remained)}/{len(self.dataset)}")
        else:
            # Fallback: original global-threshold logic (no cls_labels provided)
            threshold = scores_np.mean()
            easy_idx  = np.where(scores_np < threshold)[0]
            hard_idx  = np.where(scores_np >= threshold)[0]
            remained.extend(hard_idx.tolist())
            n_keep = int(kr * len(easy_idx))
            kept = np.random.choice(easy_idx, n_keep, replace=False)
            if len(kept) > 0:
                self.weights[kept] = 1.0 / kr
                remained.extend(kept.tolist())
            print(f"[InfoBatch prune] kr={kr:.3f} | remained={len(remained)}/{len(self.dataset)}")

        self.num_pruned_samples += len(self.dataset) - len(remained)
        np.random.shuffle(remained)
        return remained

    @property
    def sampler(self):
        sampler = IBSampler(self)
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedIBSampler(sampler)
        return sampler

    def no_prune(self):
        samples_indices = list(range(len(self)))
        np.random.shuffle(samples_indices)
        return samples_indices

    def mean_score(self):
        return self.scores.mean()

    def get_weights(self, indexes):
        return self.weights[indexes]

    def get_pruned_count(self):
        return self.num_pruned_samples

    @property
    def stop_prune(self):
        return self.num_epochs * self.delta

    def reset_weights(self):
        self.weights[:] = 1


class IBSampler(object):
    def __init__(self, dataset: InfoBatch):
        self.dataset = dataset
        self.stop_prune = dataset.stop_prune
        self.iterations = 0
        self.sample_indices = None
        self.iter_obj = None
        self.reset()

    def __getitem__(self, idx):
        return self.sample_indices[idx]

    def reset(self):
        np.random.seed(self.iterations)
        warmup_stop = self.dataset.warmup_epochs      # Phase 1 end (inclusive)
        prune_stop  = self.stop_prune                 # Phase 2 end = delta * num_epochs

        if self.iterations <= warmup_stop:
            # Phase 1: warmup — full dataset, no pruning, EMA scores accumulate
            print(f"[CS-IB] epoch={self.iterations} Phase 1 (warmup): full dataset, no pruning")
            self.sample_indices = self.dataset.no_prune()

        elif self.iterations <= prune_stop:
            # Phase 2: CS-pruning with progressive ratio ramp
            #   ρ(t) = ρ_max * (t - warmup_stop) / (prune_stop - warmup_stop)
            ramp = (self.iterations - warmup_stop) / max(1, prune_stop - warmup_stop)
            prune_ratio_max = 1.0 - self.dataset.keep_ratio   # ρ_max from config
            effective_prune = prune_ratio_max * ramp
            effective_keep  = max(self.dataset.keep_ratio, 1.0 - effective_prune)
            print(f"[CS-IB] epoch={self.iterations} Phase 2 (pruning): ramp={ramp:.3f} "
                  f"effective_keep={effective_keep:.3f}")
            self.sample_indices = self.dataset.prune(effective_keep_ratio=effective_keep)

        else:
            # Phase 3: annealing — full dataset, weights reset to 1.0
            if self.iterations == int(prune_stop) + 1:
                print(f"[CS-IB] epoch={self.iterations} Phase 3 start: resetting weights")
                self.dataset.reset_weights()
            self.sample_indices = self.dataset.no_prune()

        self.iter_obj = iter(self.sample_indices)
        self.iterations += 1

    def __next__(self):
        return next(self.iter_obj) # may raise StopIteration
        
    def __len__(self):
        return len(self.sample_indices)

    def __iter__(self):
        # Do NOT call reset() here. reset() (and hence prune()) is called
        # explicitly once per epoch from the training loop via
        # train_loader.sampler.reset(). Calling reset() here causes prune() to
        # fire on every spurious iter(sampler) that PyTorch triggers internally
        # (e.g. via the _sampler_iter-is-None guard in info_hack_indices).
        self.iter_obj = iter(self.sample_indices)
        return self


class DistributedIBSampler(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler can change size during training.
    """
    class DatasetFromSampler(Dataset):
        def __init__(self, sampler: IBSampler):
            self.dataset = sampler
            # self.indices = None
 
        def reset(self, ):
            self.indices = None
            self.dataset.reset()

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index: int):
            """Gets element of the dataset.
            Args:
                index: index of the element in the dataset
            Returns:
                Single element by index
            """
            # if self.indices is None:
            #    self.indices = list(self.dataset)
            return self.dataset[index]

    def __init__(self, dataset: IBSampler, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = True) -> None:
        sampler = self.DatasetFromSampler(dataset)
        super(DistributedIBSampler, self).__init__(
            sampler, num_replicas, rank, shuffle, seed, drop_last)
        self.sampler = sampler
        self.dataset = sampler.dataset.dataset # the real dataset.
        self.iter_obj = None

    def __iter__(self) -> Iterator[int]:
        """
        Notes self.dataset is actually an instance of IBSampler rather than InfoBatch.
        """
        self.sampler.reset()
        if self.drop_last and len(self.sampler) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.sampler) - self.num_replicas) /
                self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(
                len(self.sampler) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.sampler), generator=g).tolist()
        else:
            indices = list(range(len(self.sampler)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size /
                            len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # print('distribute iter is called')
        self.iter_obj = iter(itemgetter(*indices)(self.sampler))
        return self.iter_obj
   

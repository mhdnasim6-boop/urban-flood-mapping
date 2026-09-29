"""Per-channel normalisation statistics and class pixel counts from training chips."""
import numpy as np
import rasterio
import torch

from .bands import ALL_CHANNELS, CLASSES
from .data import ChipDataset
from .features import compute_all_channels


def channel_stats(items, event_stats, n_chips=400, px_per_chip=4000, seed=0):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(items), size=min(n_chips, len(items)), replace=False)
    ds = ChipDataset([items[i] for i in pick], event_stats)
    rows = []
    for k in range(len(ds)):
        b = ds[k]
        x = compute_all_channels(b["raw"][None], b["aux"][None], b["coh_offset"][None])[0]
        x = x.reshape(len(ALL_CHANNELS), -1)
        x = x[:, torch.isfinite(x).all(0)]
        if x.shape[1]:
            sel = rng.choice(x.shape[1], size=min(px_per_chip, x.shape[1]), replace=False)
            rows.append(x[:, sel].numpy())
    s = np.concatenate(rows, 1)
    return ({c: float(v) for c, v in zip(ALL_CHANNELS, s.mean(1))},
            {c: float(v) for c, v in zip(ALL_CHANNELS, s.std(1))})


def class_counts(items):
    counts = np.zeros(len(CLASSES), np.int64)
    for it in items:
        with rasterio.open(it.gt) as g:
            counts += np.bincount(g.read(1).ravel(), minlength=256)[: len(CLASSES)]
    return counts.tolist()


def compute(items, event_stats):
    mean, std = channel_stats(items, event_stats)
    return {"mean": mean, "std": std, "class_counts": class_counts(items)}

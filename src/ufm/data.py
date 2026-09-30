"""Datasets for the compact UrbanSARFloods chips.

Train/val: <root>/{01_NF,02_FO,03_FU}/{SAR,GT,AUX}/, split by the official
Train_dataset.txt / Valid_dataset.txt lists.
Test:      <root>/<event>/{SAR,AUX}/ chips + <root>/<event>/GT_full.tif
           (full-frame reference map, windowed per chip).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import from_bounds
from torch.utils.data import Dataset, WeightedRandomSampler

from .bands import IGNORE, decode_aux, decode_sar, event_of

CATEGORIES = ["01_NF", "02_FO", "03_FU"]


@dataclass
class Item:
    sar: Path
    aux: Path
    gt: Path | None
    event: str
    category: str  # 01_NF / 02_FO / 03_FU, or "test"

    @property
    def name(self):
        return self.sar.name.replace("_SAR.tif", "")


def _aux_for(sar):
    return sar.parent.parent / "AUX" / sar.name.replace("_SAR.tif", "_AUX.tif")


def _read_list(path):
    """Official split lists hold lines like '../03_FU/GT/<chip>_GT.tif'."""
    if not path.exists():
        return None
    return {Path(l.strip()).name.replace("_GT.tif", "") for l in path.read_text().splitlines() if l.strip()}


def trainval_items(root, split):
    """split: 'train' or 'val'. If Train_dataset.txt is absent, train = all - val."""
    root = Path(root)
    val = _read_list(root / "Valid_dataset.txt") or set()
    train = _read_list(root / "Train_dataset.txt")
    items = []
    for cat in CATEGORIES:
        for sar in sorted((root / cat / "SAR").glob("*_SAR.tif")):
            name = sar.name.replace("_SAR.tif", "")
            gt = root / cat / "GT" / f"{name}_GT.tif"
            if not gt.exists():
                continue
            in_val = name in val
            in_train = (name in train) if train is not None else not in_val
            if (split == "val" and in_val) or (split == "train" and in_train):
                items.append(Item(sar, _aux_for(sar), gt, event_of(sar.name), cat))
    return items


def test_items(root, events=None):
    root = Path(root)
    items = []
    for ev_dir in sorted(p for p in root.iterdir() if (p / "SAR").is_dir()):
        if events and ev_dir.name not in events:
            continue
        for sar in sorted((ev_dir / "SAR").glob("*_SAR.tif")):
            items.append(Item(sar, _aux_for(sar), ev_dir / "GT_full.tif", ev_dir.name, "test"))
    return items


def category_sampler(items, shares, num_samples=None, seed=0):
    """Weighted sampler giving each chip category a target share of each epoch."""
    counts = {c: sum(i.category == c for i in items) for c in shares}
    w = [shares.get(i.category, 0) / max(counts.get(i.category, 0), 1) for i in items]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(w, num_samples or len(items), replacement=True, generator=g)


class ChipDataset(Dataset):
    """Returns raw SAR (8,H,W, NaN = nodata), aux (3,H,W), label (H,W), coherence
    offset (2,) of the chip's event."""

    def __init__(self, items, event_stats, crop=None, train=False, focus_prob=0.0,
                 use_aux=True, seed=0):
        self.items, self.stats = items, event_stats or {}
        self.crop, self.train, self.focus_prob, self.use_aux = crop, train, focus_prob, use_aux
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def _label(self, it, src):
        if it.category != "test":
            with rasterio.open(it.gt) as g:
                return g.read(1)
        with rasterio.open(it.gt) as g:  # window of the full-event reference map
            w = from_bounds(*src.bounds, transform=g.transform).round_offsets().round_lengths()
            return g.read(1, window=w, boundless=True, fill_value=IGNORE)

    def __getitem__(self, i):
        it = self.items[i]
        with rasterio.open(it.sar) as src:
            raw = decode_sar(src.read())
            label = self._label(it, src)
        if self.use_aux and it.aux.exists():
            with rasterio.open(it.aux) as a:
                aux = decode_aux(a.read())
        else:
            aux = np.zeros((3,) + raw.shape[1:], np.float32)
        label = label.astype(np.int64)
        label[~np.isfinite(raw).all(0)] = IGNORE

        if self.crop and raw.shape[1] > self.crop:
            y, x = self._crop_origin(label)
            s = np.s_[y:y + self.crop, x:x + self.crop]
            raw, aux, label = raw[:, s[0], s[1]], aux[:, s[0], s[1]], label[s]
        if self.train:
            if self.rng.random() < 0.5:
                raw, aux, label = raw[:, :, ::-1], aux[:, :, ::-1], label[:, ::-1]
            if self.rng.random() < 0.5:
                raw, aux, label = raw[:, ::-1], aux[:, ::-1], label[::-1]

        off = np.array(self.stats.get(it.event, [0.0, 0.0]), np.float32)
        return {
            "raw": torch.from_numpy(np.ascontiguousarray(raw)),
            "aux": torch.from_numpy(np.ascontiguousarray(aux)),
            "label": torch.from_numpy(np.ascontiguousarray(label)),
            "coh_offset": torch.from_numpy(off),
            "index": i,
        }

    def _crop_origin(self, label):
        H, W = label.shape
        c = self.crop
        if self.train and self.rng.random() < self.focus_prob:
            # centre the crop on a random flood pixel, urban first (rarest class)
            for cls in (2, 1):
                ys, xs = np.nonzero(label == cls)
                if len(ys):
                    k = self.rng.integers(len(ys))
                    y = int(np.clip(ys[k] - c // 2 + self.rng.integers(-c // 4, c // 4 + 1), 0, H - c))
                    x = int(np.clip(xs[k] - c // 2 + self.rng.integers(-c // 4, c // 4 + 1), 0, W - c))
                    return y, x
        if not self.train:
            return (H - c) // 2, (W - c) // 2
        return int(self.rng.integers(0, H - c + 1)), int(self.rng.integers(0, W - c + 1))


def worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)

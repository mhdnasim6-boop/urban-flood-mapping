"""Shared test-set loop: iterate events, collect per-event confusion matrices and
write full-event map mosaics (GeoTIFF) on the reference-map grid."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

from .bands import IGNORE
from .metrics import Confusion


class EventMosaic:
    """uint8 rasters on the GT_full grid, filled chip by chip."""

    def __init__(self, gt_path, layers):
        with rasterio.open(gt_path) as g:
            self.profile, self.shape, self.transform = g.profile, g.shape, g.transform
        fill = {"pred": IGNORE}
        self.layers = {n: np.full(self.shape, fill.get(n, 255), np.uint8) for n in layers}

    def put(self, name, chip_bounds, arr):
        w = from_bounds(*chip_bounds, transform=self.transform).round_offsets().round_lengths()
        r0, c0 = int(w.row_off), int(w.col_off)
        r1, c1 = min(r0 + arr.shape[0], self.shape[0]), min(c0 + arr.shape[1], self.shape[1])
        ar0, ac0 = max(0, -r0), max(0, -c0)
        r0, c0 = max(r0, 0), max(c0, 0)
        if r1 > r0 and c1 > c0:
            self.layers[name][r0:r1, c0:c1] = arr[ar0:ar0 + r1 - r0, ac0:ac0 + c1 - c0]

    def save(self, folder, prefix):
        from .io import write_tif
        folder.mkdir(parents=True, exist_ok=True)
        for n, a in self.layers.items():
            write_tif(folder / f"{prefix}_{n}.tif", a, self.profile, "uint8", nodata=255)


def chip_bounds(items):
    out = []
    for it in items:
        with rasterio.open(it.sar) as s:
            out.append(tuple(s.bounds))
    return out


def save_metrics(per_event, out_dir, method):
    """per_event: {event: Confusion}. Adds a pooled 'ALL' row; writes CSV + JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pooled = Confusion()
    rows = []
    for ev, c in per_event.items():
        pooled += c
        rows.append({"method": method, "event": ev, **c.summary()})
    rows.append({"method": method, "event": "ALL", **pooled.summary()})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test_metrics.csv", index=False)
    (out_dir / "confusion.json").write_text(json.dumps(
        {ev: c.m.tolist() for ev, c in {**per_event, "ALL": pooled}.items()}, indent=1))
    cols = ["event", "flood_open_f1", "flood_urban_f1", "flood_mean_f1", "flood_any_f1", "kappa"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return df

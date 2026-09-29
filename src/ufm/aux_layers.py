"""Auxiliary layers (channels 15-17) resampled onto each SAR chip grid.

Sources (all public, no account needed):
  HAND   ASF GLO-30 HAND, 1 deg COG tiles (derived from Copernicus GLO-30 DEM)
  DEM    Copernicus GLO-30 DEM, 1 deg COG tiles (slope is computed from it)
  Water  JRC Global Surface Water v1.4 occurrence, 10 deg tiles
"""
import math
import urllib.error
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge

from .io import fetch_bytes


def _cop_name(lat, lon):
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00"


def hand_url(lat, lon):
    return f"https://glo-30-hand.s3.us-west-2.amazonaws.com/v1/2021/{_cop_name(lat, lon)}_HAND.tif"


def dem_url(lat, lon):
    n = _cop_name(lat, lon)
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{n}_DEM/{n}_DEM.tif"


def jrc_url(lat10, lon10):
    """JRC tiles are named by their top-left corner, e.g. occurrence_100W_30N."""
    ew = f"{-lon10}W" if lon10 < 0 else f"{lon10}E"
    ns = f"{lat10}N" if lat10 >= 0 else f"{-lat10}S"
    return ("https://storage.googleapis.com/global-surface-water/downloads2021/"
            f"occurrence/occurrence_{ew}_{ns}v1_4_2021.tif")


def tiles_for_bounds(b, pad=0.01):
    """1-deg (lat, lon) floors and 10-deg (top, left) JRC corners covering bounds."""
    w, s, e, n = b.left - pad, b.bottom - pad, b.right + pad, b.top + pad
    one = {(la, lo) for la in range(math.floor(s), math.floor(n) + 1)
           for lo in range(math.floor(w), math.floor(e) + 1)}
    ten = {(la, lo) for la in range(math.ceil(s / 10) * 10, math.ceil(n / 10) * 10 + 1, 10)
           for lo in range(math.floor(w / 10) * 10, math.floor(e / 10) * 10 + 1, 10)}
    return one, ten


class TileCache:
    """Downloads source tiles once into a local folder; missing tiles (ocean) are
    remembered and skipped."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def get(self, url):
        path = self.folder / url.rsplit("/", 1)[1]
        missing = path.with_suffix(".missing")
        if path.exists():
            return path
        if missing.exists():
            return None
        try:
            data = fetch_bytes(url, retries=4)
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                missing.touch()
                return None
            raise
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.rename(path)
        return path

    def clear(self):
        for p in self.folder.glob("*"):
            p.unlink()


def _mosaic(paths, bounds, res, pad_px=0):
    """Resample tiles onto the chip grid (EPSG:4326, same origin/resolution)."""
    left, bottom, right, top = bounds
    b = (left - pad_px * res[0], bottom - pad_px * res[1], right + pad_px * res[0], top + pad_px * res[1])
    width = round((right - left) / res[0]) + 2 * pad_px
    height = round((top - bottom) / res[1]) + 2 * pad_px
    if not paths:
        return np.full((height, width), np.nan, np.float32)
    srcs = [rasterio.open(p) for p in paths]
    try:
        nod = srcs[0].nodata
        arr, _ = merge(srcs, bounds=b, res=res, nodata=nod, dtype="float32",
                       resampling=Resampling.bilinear)
    finally:
        for s in srcs:
            s.close()
    a = np.full((height, width), np.nan, np.float32)  # guard against off-by-one rounding
    h, w = min(height, arr.shape[1]), min(width, arr.shape[2])
    a[:h, :w] = arr[0, :h, :w]
    if nod is not None:
        a[a == nod] = np.nan
    return a


def slope_deg(dem, res_deg, lat):
    """Slope in degrees from a DEM on a geographic grid."""
    dy = res_deg[1] * 110_574.0
    dx = res_deg[0] * 111_320.0 * math.cos(math.radians(lat))
    gy, gx = np.gradient(dem, dy, dx)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def aux_for_chip(cache, bounds, res):
    """Return (hand_m, slope_deg, occurrence_pct) arrays on the chip grid."""
    one, ten = tiles_for_bounds(bounds)
    hand_p = [p for p in (cache.get(hand_url(*t)) for t in sorted(one)) if p]
    dem_p = [p for p in (cache.get(dem_url(*t)) for t in sorted(one)) if p]
    jrc_p = [p for p in (cache.get(jrc_url(*t)) for t in sorted(ten)) if p]

    hand = _mosaic(hand_p, bounds, res)
    dem = _mosaic(dem_p, bounds, res, pad_px=1)
    lat = (bounds[1] + bounds[3]) / 2
    slope = slope_deg(dem, res, lat)[1:-1, 1:-1]
    occ = _mosaic(jrc_p, bounds, res)
    occ[(occ > 100)] = np.nan  # JRC: 255 = no data
    # no DEM tile = open sea: treat as water at drainage level
    if not hand_p:
        hand[:] = 0.0
    return hand, slope, occ

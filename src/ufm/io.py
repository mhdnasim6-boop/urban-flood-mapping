"""File/network helpers: resumable HTTP streaming, GeoTIFF read/write, HF listing."""
import io
import json
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np
import rasterio
from rasterio.io import MemoryFile

HF_REPO = "S1Floodbenchmark/UrbanSARFloods_v1"
HF_RESOLVE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/"
HF_TREE = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/"


def _request(url, extra_headers=None):
    """Request with an optional Hugging Face token (env HF_TOKEN) to avoid
    anonymous rate limits."""
    req = urllib.request.Request(url, headers=extra_headers or {})
    token = os.environ.get("HF_TOKEN")
    if token and url.startswith("https://huggingface.co/"):
        req.add_header("Authorization", f"Bearer {token}")
    return req


class ResumableHTTPStream(io.RawIOBase):
    """Read-only stream over HTTP that reconnects with a Range header on failure.

    Lets tarfile stream a 38 GB .tar.gz without storing it and without restarting
    from zero when the connection drops.
    """

    def __init__(self, url, retries=20, timeout=120):
        self.url, self.retries, self.timeout = url, retries, timeout
        self.pos = 0
        self._resp = None
        self._open()

    def _open(self):
        req = _request(self.url, {"Range": f"bytes={self.pos}-"} if self.pos else None)
        self._resp = urllib.request.urlopen(req, timeout=self.timeout)
        if self.pos and self._resp.status != 206:
            raise IOError("server ignored Range header; cannot resume")

    def readable(self):
        return True

    def readinto(self, b):
        for attempt in range(self.retries):
            try:
                n = self._resp.readinto(b)
                self.pos += n
                return n
            except Exception as e:  # network hiccup: reconnect at current offset
                wait = min(60, 2 ** attempt)
                print(f"[stream] {e!r} at byte {self.pos:,}; reconnecting in {wait}s")
                time.sleep(wait)
                try:
                    self._open()
                except Exception as e2:
                    print(f"[stream] reconnect failed: {e2!r}")
        raise IOError(f"giving up after {self.retries} retries at byte {self.pos:,}")


def hf_list(path):
    """Recursively list files under a folder of the UrbanSARFloods HF repo."""
    url = HF_TREE + path + "?recursive=true&limit=1000"
    out = []
    while url:
        r = urllib.request.urlopen(_request(url), timeout=120)
        out += [f for f in json.load(r) if f["type"] == "file"]
        m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("link") or "")
        url = m.group(1) if m else None
    return out


TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def fetch_bytes(url, retries=10, tiff=False):
    """GET url with retries. With tiff=True, also retry truncated/non-TIFF bodies
    (Hugging Face occasionally returns an empty body under parallel load)."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(_request(url), timeout=300) as r:
                data = r.read()
                expected = r.headers.get("content-length")
            if expected is not None and len(data) != int(expected):
                raise IOError(f"truncated: {len(data)} of {expected} bytes")
            if tiff and data[:4] not in TIFF_MAGIC:
                raise IOError(f"not a TIFF ({len(data)} bytes)")
            return data
        except urllib.error.HTTPError as e:
            if e.code in (403, 404) or attempt == retries - 1:
                raise
            err = e
        except Exception as e:
            if attempt == retries - 1:
                raise
            err = e
        print(f"[fetch] {err!r}; retrying {url.rsplit('/', 1)[-1]}")
        time.sleep(min(60, 2 ** attempt))  # up to ~6 min total: rides out rate limits


def read_tif_bytes(data):
    """Bytes of a GeoTIFF -> (array, profile)."""
    with MemoryFile(data) as m, m.open() as src:
        return src.read(), src.profile


def write_tif(path, array, like_profile, dtype, nodata=None):
    """Write a compressed, tiled GeoTIFF with the georeferencing of like_profile."""
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None]
    prof = {
        "driver": "GTiff",
        "width": array.shape[2],
        "height": array.shape[1],
        "count": array.shape[0],
        "dtype": dtype,
        "crs": like_profile["crs"],
        "transform": like_profile["transform"],
        "compress": "deflate",
        "predictor": 2,
        "nodata": nodata,
    }
    h, w = array.shape[1:]
    if h >= 16 and w >= 16:  # 256 px tiles (tile size, not image size, must be a multiple of 16)
        prof.update(tiled=True, blockxsize=min(256, w // 16 * 16), blockysize=min(256, h // 16 * 16))
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(array.astype(dtype))

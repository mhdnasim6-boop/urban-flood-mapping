"""Physics-guided features (channels 9-14) and input assembly.

Everything here runs batched on the GPU inside the model, so ablations only
change which channels are selected, never the stored data.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .bands import ALL_CHANNELS, RAW_BANDS

R = {n: i for i, n in enumerate(RAW_BANDS)}

# Pixels whose pre-event VV coherence is at least this are treated as stable
# scatterers (mostly buildings) when estimating the event-wide coherence drop.
STABLE_COH = 0.5
UFI_EPS = 0.05


def stable_drop_sample(raw, n=4000, rng=None):
    """Sample (drop_vv, drop_vh) over stable pixels of one chip, numpy (8,H,W)."""
    rng = rng or np.random.default_rng(0)
    pre_vv = raw[R["coh_pre_vv"]]
    ok = np.isfinite(raw).all(0) & (pre_vv >= STABLE_COH)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return np.empty((0, 2), np.float32)
    idx = rng.choice(idx, size=min(n, idx.size), replace=False)
    f = raw.reshape(8, -1)[:, idx]
    return np.stack([f[R["coh_pre_vv"]] - f[R["coh_co_vv"]],
                     f[R["coh_pre_vh"]] - f[R["coh_co_vh"]]], 1).astype(np.float32)


def event_offsets(samples_by_event):
    """{event: [arrays (n,2)]} -> {event: [median_drop_vv, median_drop_vh]}.

    The median coherence drop of stable pixels over a whole event captures
    decorrelation that is not due to flooding (rain, wet surfaces, baseline).
    Subtracting it is the normalisation recommended in Pulvirenti et al. (2019).
    """
    out = {}
    for ev, parts in samples_by_event.items():
        s = np.concatenate(parts) if parts else np.empty((0, 2))
        out[ev] = [float(np.median(s[:, 0])), float(np.median(s[:, 1]))] if len(s) else [0.0, 0.0]
    return out


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))


def load_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def compute_all_channels(raw, aux, coh_offset):
    """raw (B,8,H,W) dB/coherence with NaN, aux (B,3,H,W), coh_offset (B,2)
    -> (B,17,H,W) in ALL_CHANNELS order, unnormalised, NaN where raw is NaN."""
    g = lambda n: raw[:, R[n]]
    off = coh_offset[:, :, None, None]
    d_int_vv = g("int_post_vv") - g("int_pre_vv")
    d_int_vh = g("int_post_vh") - g("int_pre_vh")
    d_coh_vv = (g("coh_pre_vv") - g("coh_co_vv")) - off[:, 0]
    d_coh_vh = (g("coh_pre_vh") - g("coh_co_vh")) - off[:, 1]
    xpol = g("int_post_vv") - g("int_post_vh")
    # Zhang et al. (2021) urban flooding index in dB: backscatter ratio divided
    # by coherence ratio. High only when intensity rises AND coherence falls.
    coh_ratio = (g("coh_co_vv").clamp(min=0) + UFI_EPS) / (g("coh_pre_vv").clamp(min=0) + UFI_EPS)
    ufi = (d_int_vv - 10 * torch.log10(coh_ratio)).clamp(-20, 30)
    phys = torch.stack([d_int_vv, d_int_vh, d_coh_vv, d_coh_vh, xpol, ufi], 1)
    return torch.cat([raw, phys, aux], 1)


class InputBuilder(nn.Module):
    """Builds, normalises and selects input channels. Stats live in buffers so a
    checkpoint is self-contained."""

    def __init__(self, channels, mean, std):
        super().__init__()
        self.channels = list(channels)
        idx = [ALL_CHANNELS.index(c) for c in self.channels]
        self.register_buffer("idx", torch.tensor(idx, dtype=torch.long))
        self.register_buffer("mean", torch.tensor([mean[c] for c in self.channels]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([max(std[c], 1e-6) for c in self.channels]).view(1, -1, 1, 1))

    def forward(self, raw, aux, coh_offset):
        x = compute_all_channels(raw, aux, coh_offset).index_select(1, self.idx)
        x = (x - self.mean) / self.std
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)  # nodata -> channel mean


def valid_mask(raw):
    """(B,8,H,W) -> (B,H,W) bool, True where all SAR bands are present."""
    return torch.isfinite(raw).all(1)

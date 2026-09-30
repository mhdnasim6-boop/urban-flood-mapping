"""Band layout of UrbanSARFloods chips and the compact uint8/uint16 storage format.

Band order was verified against the Jubba test reference labels (see
README, "Band order"). The paper text lists intensity first, but the GeoTIFFs
store coherence first, and VH before VV inside each pair.
"""
import numpy as np

RAW_BANDS = [
    "coh_pre_vh", "coh_pre_vv",    # InSAR coherence, pre-event pair
    "coh_co_vh", "coh_co_vv",      # InSAR coherence, co-event pair
    "int_pre_vh", "int_pre_vv",    # backscatter intensity (dB), pre-event
    "int_post_vh", "int_post_vv",  # backscatter intensity (dB), post-event
]
PHYS_FEATURES = [
    "d_int_vv",       # post - pre intensity, VV (dB)
    "d_int_vh",       # post - pre intensity, VH (dB)
    "d_coh_vv_norm",  # coherence drop VV minus event-level drop of stable pixels
    "d_coh_vh_norm",  # same for VH
    "xpol_post",      # VV - VH post-event (dB): double-bounce vs volume scattering
    "ufi_vv",         # urban flood index (dB): intensity rise + coherence loss
]
AUX_LAYERS = [
    "hand",       # log1p(height above nearest drainage, m)
    "slope",      # terrain slope (degrees)
    "water_occ",  # JRC permanent water occurrence (0-1)
]
ALL_CHANNELS = RAW_BANDS + PHYS_FEATURES + AUX_LAYERS
GROUPS = {"@raw": RAW_BANDS, "@phys": PHYS_FEATURES, "@aux": AUX_LAYERS}

CLASSES = ["non_flood", "flood_open", "flood_urban"]
IGNORE = 255

# uint8 SAR encoding, 0 = nodata.
# Intensity: 0.2 dB steps over [-39.8, +11] dB, far below speckle (~1 dB).
# Coherence: 1/254 steps over [0, 1].
DB_MIN, DB_STEP = -40.0, 0.2
COH_SCALE = 254.0

# uint8 auxiliary encoding, 255 = nodata.
# HAND as log1p(m) in 1/32 steps (0.06 m resolution at 1 m, up to ~2800 m),
# slope in 0.25 deg steps (0-63.5 deg), water occurrence in % (0-100).
AUX_SCALE = np.array([32.0, 4.0, 1.0], dtype=np.float32)
AUX_NODATA = 255


def expand_channels(names):
    """Expand group shortcuts (@raw, @phys, @aux) and validate channel names."""
    out = []
    for n in names:
        for c in GROUPS.get(n, [n]):
            if c not in ALL_CHANNELS:
                raise ValueError(f"unknown channel {c!r}; choose from {ALL_CHANNELS}")
            if c not in out:
                out.append(c)
    return out


def encode_sar(a):
    """float32 (8,H,W) -> uint8 (8,H,W). NaN / non-finite -> 0."""
    a = np.asarray(a, dtype=np.float32)
    q = np.zeros(a.shape, dtype=np.uint8)
    finite = np.isfinite(a)
    coh = np.clip(np.rint(a[:4] * COH_SCALE) + 1, 1, 255)
    db = np.clip(np.rint((a[4:] - DB_MIN) / DB_STEP), 1, 255)
    q[:4] = np.where(finite[:4], coh, 0)
    q[4:] = np.where(finite[4:], db, 0)
    return q


def decode_sar(q):
    """uint8 (8,H,W) -> float32 with NaN for nodata."""
    q = np.asarray(q)
    a = np.empty(q.shape, dtype=np.float32)
    a[:4] = (q[:4].astype(np.float32) - 1) / COH_SCALE
    a[4:] = q[4:].astype(np.float32) * DB_STEP + DB_MIN
    a[q == 0] = np.nan
    return a


def encode_label(g):
    """GT float (H,W) with values 0/1/2 -> uint8, anything else -> IGNORE."""
    g = np.asarray(g)
    out = np.full(g.shape, IGNORE, dtype=np.uint8)
    for k in range(len(CLASSES)):
        out[g == k] = k
    return out


def encode_aux(hand_m, slope_deg, occ_pct):
    """Physical units -> uint8 (3,H,W); NaN -> AUX_NODATA."""
    a = np.stack([np.log1p(np.maximum(hand_m, 0)), slope_deg, occ_pct]).astype(np.float32)
    q = np.rint(a * AUX_SCALE[:, None, None])
    q = np.where(np.isfinite(q), np.clip(q, 0, AUX_NODATA - 1), AUX_NODATA)
    return q.astype(np.uint8)


def decode_aux(q):
    """uint8 (3,H,W) -> model-ready float32 (3,H,W): [log1p(HAND m), slope deg,
    occurrence 0-1]. Nodata is filled with 0.

    Missing values happen over sea or outside DEM tiles, where 0 (at drainage
    level, flat, no recorded water) is the neutral choice.
    """
    q = np.asarray(q)
    a = q.astype(np.float32) / AUX_SCALE[:, None, None]
    a[q == AUX_NODATA] = 0.0
    a[2] = a[2] / 100.0
    return a


def event_of(chip_name):
    """'20170830_Houston_ID_13_30_SAR.tif' -> '20170830_Houston'."""
    return chip_name.split("_ID_")[0]

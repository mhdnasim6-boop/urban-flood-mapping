"""Comparison methods B1-B4.

B1  Intensity change detection (open floods only): Otsu water threshold on
    post-event VV + a -3 dB decrease, the standard operational approach.
B2  B1 + fixed coherence-drop threshold (0.3) inside stable/urban pixels, the
    rule most urban-flood studies use (e.g. Chini et al. 2019; Bioresita et al.
    2021) and also how the dataset's semi-automatic urban labels were made.
B3  Rule-based decision tree combining intensity and event-normalised coherence
    (after Natsuaki & Hirose 2018; Pulvirenti et al. 2019).
B4  Random forest on the same 17 per-pixel inputs as the proposed model
    (XGBoost random-forest mode, so it can run on the GPU).
"""
import numpy as np
import torch
from skimage.filters import threshold_otsu

from .bands import ALL_CHANNELS
from .features import compute_all_channels

CH = {n: i for i, n in enumerate(ALL_CHANNELS)}

T = {
    "water_db_range": (-24.0, -15.0),  # clamp for the Otsu water threshold
    "decrease_db": -3.0,               # intensity decrease for open water
    "strong_decrease_db": -6.0,        # B3 rule 1
    "strong_increase_db": 6.0,         # B3 double-bounce rule
    "urban_coh": 0.5,                  # pre-event coherence of stable scatterers
    "coh_drop": 0.3,                   # B2 fixed coherence-drop threshold
    "coh_drop_norm": 0.2,              # B3 threshold on event-normalised drop
}


def features(raw, aux, off):
    """torch batch -> numpy (B,17,H,W)"""
    with torch.no_grad():
        return compute_all_channels(raw.float(), aux.float(), off.float()).numpy()


def water_threshold(post_vv_sample):
    s = post_vv_sample[np.isfinite(post_vv_sample)]
    t = threshold_otsu(s) if s.size > 100 else -18.0
    return float(np.clip(t, *T["water_db_range"]))


def _open_water(f, t_water):
    return (f[:, CH["int_post_vv"]] < t_water) & (f[:, CH["d_int_vv"]] < T["decrease_db"])


def b1(f, t_water):
    return np.where(_open_water(f, t_water), 1, 0).astype(np.uint8)


def b2(f, t_water):
    fo = _open_water(f, t_water)
    urban = f[:, CH["coh_pre_vv"]] >= T["urban_coh"]
    drop = f[:, CH["coh_pre_vv"]] - f[:, CH["coh_co_vv"]]
    fu = urban & (drop > T["coh_drop"]) & ~fo
    return np.select([fo, fu], [1, 2], 0).astype(np.uint8)


def b3(f, t_water):
    d = f[:, CH["d_int_vv"]]
    fo = _open_water(f, t_water) | (d < T["strong_decrease_db"])
    urban = f[:, CH["coh_pre_vv"]] >= T["urban_coh"]
    coh_loss = (f[:, CH["d_coh_vv_norm"]] > T["coh_drop_norm"]) & (d > T["decrease_db"])
    fu = urban & ~fo & (coh_loss | (d > T["strong_increase_db"]))
    return np.select([fo, fu], [1, 2], 0).astype(np.uint8)


RULES = {"B1": b1, "B2": b2, "B3": b3}


# ---------------------------------------------------------------- B4: random forest
def sample_pixels(dataset, per_class=(400, 2000, 2000), max_chips=3000, seed=0):
    """Class-stratified pixel sample from training chips -> X (n,17), y (n,)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(dataset))[:max_chips]
    X, Y = [], []
    for i in idx:
        b = dataset[i]
        f = features(b["raw"][None], b["aux"][None], b["coh_offset"][None])[0].reshape(len(ALL_CHANNELS), -1)
        y = b["label"].numpy().ravel()
        for k, n in enumerate(per_class):
            pix = np.flatnonzero(y == k)
            if pix.size:
                pick = rng.choice(pix, size=min(n, pix.size), replace=False)
                X.append(f[:, pick].T)
                Y.append(np.full(pick.size, k))
    return np.concatenate(X).astype(np.float32), np.concatenate(Y)


def train_rf(X, y, device="cpu", n_trees=150, depth=14, seed=0):
    import xgboost as xgb
    counts = np.bincount(y, minlength=3).astype(np.float64)
    w = (counts.sum() / (3 * np.maximum(counts, 1)))[y]  # balanced class weights
    rf = xgb.XGBRFClassifier(n_estimators=n_trees, max_depth=depth, subsample=0.8,
                             colsample_bynode=0.6, tree_method="hist", device=device,
                             random_state=seed, n_jobs=-1)
    rf.fit(X, y, sample_weight=w)
    return rf


def predict_rf(rf, f):
    """f: (B,17,H,W) -> (B,H,W) uint8"""
    B, C, H, W = f.shape
    X = f.transpose(0, 2, 3, 1).reshape(-1, C)
    return rf.predict(X).reshape(B, H, W).astype(np.uint8)

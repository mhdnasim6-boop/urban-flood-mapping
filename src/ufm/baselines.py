"""Comparison methods B1-B4.

B1  Intensity change detection (open floods only): Otsu water threshold on
    post-event VV + a -3 dB decrease, the standard operational approach.
B2  B1 + fixed coherence-drop threshold (0.3) inside stable/urban pixels, the
    rule most urban-flood studies use (e.g. Chini et al. 2019; Bioresita et al.
    2021) and also how the dataset's semi-automatic urban labels were made.
B3  Rule-based decision tree combining intensity and event-normalised coherence
    (after Natsuaki & Hirose 2018; Pulvirenti et al. 2019).
B4  Random forest on the same 17 per-pixel inputs as the proposed model
    (XGBoost random-forest mode, so it can run on the GPU). Trained on a
    class-stratified pixel sample; per-class decision weights are then tuned on
    the validation set (the same data the U-Nets use for model selection).

All four use SAR inputs smoothed with a 5x5 median filter, the usual speckle
suppression before pixel-wise classification.
"""
import numpy as np
import torch
from scipy.ndimage import median_filter
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


def despeckle(raw, size=5):
    """Median-filter every SAR band of a batch (B,8,H,W); NaN pixels stay NaN."""
    if not size:
        return raw
    a = raw.numpy().astype(np.float32, copy=True)
    for b in range(a.shape[0]):
        for c in range(a.shape[1]):
            x = a[b, c]
            ok = np.isfinite(x)
            if not ok.any():
                continue
            filled = np.where(ok, x, np.median(x[ok]))
            a[b, c] = np.where(ok, median_filter(filled, size=size, mode="reflect"), np.nan)
    return torch.from_numpy(a)


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
def _chip_features(dataset, i, speckle):
    b = dataset[i]
    raw = despeckle(b["raw"][None], speckle)
    f = features(raw, b["aux"][None], b["coh_offset"][None])[0].reshape(len(ALL_CHANNELS), -1)
    return f, b["label"].numpy().ravel()


def sample_pixels(dataset, per_class=(400, 2000, 2000), max_chips=3000, speckle=5, seed=0):
    """Class-stratified pixel sample from training chips -> X (n,17), y (n,)."""
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for i in rng.permutation(len(dataset))[:max_chips]:
        f, y = _chip_features(dataset, i, speckle)
        for k, n in enumerate(per_class):
            pix = np.flatnonzero(y == k)
            if pix.size:
                pick = rng.choice(pix, size=min(n, pix.size), replace=False)
                X.append(f[:, pick].T)
                Y.append(np.full(pick.size, k))
    return np.concatenate(X).astype(np.float32), np.concatenate(Y)


def sample_uniform(dataset, max_chips=600, px_per_chip=20000, speckle=5, seed=1):
    """Uniform pixel sample (keeps the true class proportions) for threshold tuning."""
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for i in rng.permutation(len(dataset))[:max_chips]:
        f, y = _chip_features(dataset, i, speckle)
        pix = np.flatnonzero(y != 255)
        pick = rng.choice(pix, size=min(px_per_chip, pix.size), replace=False)
        X.append(f[:, pick].T)
        Y.append(y[pick])
    return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.int64)


def train_rf(X, y, device="cpu", n_trees=150, depth=14, seed=0):
    import xgboost as xgb
    rf = xgb.XGBRFClassifier(n_estimators=n_trees, max_depth=depth, subsample=0.8,
                             colsample_bynode=0.6, tree_method="hist", device=device,
                             random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    return rf


def tune_class_weights(proba, y, grid=np.logspace(-2, 1, 16)):
    """Pick multipliers (w_FO, w_FU) for argmax(p * [1, w_FO, w_FU]) that maximise
    the mean F1 of the two flood classes on validation pixels."""
    from .metrics import Confusion
    best = (-1.0, 1.0, 1.0)
    for w1 in grid:
        for w2 in grid:
            c = Confusion()
            c.update(torch.from_numpy((proba * np.array([1.0, w1, w2])).argmax(1)), torch.from_numpy(y))
            score = c.summary()["flood_mean_f1"]
            if score > best[0]:
                best = (score, float(w1), float(w2))
    return {"val_flood_mean_f1": best[0], "weights": [1.0, best[1], best[2]]}


def predict_rf(rf, f, weights=(1.0, 1.0, 1.0)):
    """f: (B,17,H,W) -> (B,H,W) uint8"""
    B, C, H, W = f.shape
    p = rf.predict_proba(f.transpose(0, 2, 3, 1).reshape(-1, C))
    return (p * np.asarray(weights)).argmax(1).reshape(B, H, W).astype(np.uint8)

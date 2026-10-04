"""Run comparison methods B1-B4 on the test events.

    python scripts/run_baselines.py --test-root /kaggle/input/ufm-test \
        --trainval-root /kaggle/input/ufm-trainval --out /kaggle/working/runs/baselines

Each method gets <out>/<method>/test/test_metrics.csv and <event>_pred.tif maps.
B4 (random forest) needs --trainval-root to train and to tune its class weights
on the validation split. Settings actually used go to <out>/settings.json.
"""
import json
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

try:  # must load before torch: on macOS the two OpenMP runtimes otherwise segfault
    import xgboost  # noqa: F401
except ImportError:
    pass
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ufm import baselines as bl  # noqa: E402
from ufm.bands import IGNORE  # noqa: E402
from ufm.data import ChipDataset, test_items, trainval_items  # noqa: E402
from ufm.evaluation import EventMosaic, chip_bounds, save_metrics  # noqa: E402
from ufm.features import load_json  # noqa: E402
from ufm.metrics import Confusion  # noqa: E402

if sys.platform == "darwin":  # macOS: torch and xgboost ship separate OpenMP runtimes
    torch.set_num_threads(1)


def event_water_threshold(ds, speckle, n_chips=300, px=3000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for i in rng.permutation(len(ds))[:n_chips]:
        v = bl.despeckle(ds[i]["raw"][None], speckle, bands=[7])[0, 7].numpy().ravel()  # int_post_vv
        vals.append(rng.choice(v, size=min(px, v.size), replace=False))
    return bl.water_threshold(np.concatenate(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--trainval-root")
    ap.add_argument("--out", default="runs/baselines")
    ap.add_argument("--methods", nargs="*", default=["B1", "B2", "B3", "B4"])
    ap.add_argument("--rf-device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rf-max-chips", type=int, default=3000)
    ap.add_argument("--speckle", type=int, default=5, help="median filter size (0 = off)")
    ap.add_argument("--no-maps", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)

    settings = {"speckle_filter": args.speckle, "thresholds": bl.T, "water_threshold_db": {}}
    rf, rf_w = None, None
    if "B4" in args.methods:
        if not args.trainval_root:
            sys.exit("B4 needs --trainval-root")
        tv = Path(args.trainval_root)
        tv_stats = load_json(tv / "event_stats.json", {})
        model_path = out / "B4" / "rf.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if model_path.exists():
            rf = pickle.loads(model_path.read_bytes())
        else:
            ds_tr = ChipDataset(trainval_items(tv, "train"), tv_stats)
            X, y = bl.sample_pixels(ds_tr, max_chips=args.rf_max_chips, speckle=args.speckle)
            print(f"B4: training random forest on {len(y):,} pixels, class counts {np.bincount(y, minlength=3)}")
            rf = bl.train_rf(X, y, device=args.rf_device)
            model_path.write_bytes(pickle.dumps(rf))
        ds_va = ChipDataset(trainval_items(tv, "val"), tv_stats)
        Xv, yv = bl.sample_uniform(ds_va, max_chips=min(600, args.rf_max_chips), speckle=args.speckle)
        tuned = bl.tune_class_weights(rf.predict_proba(Xv), yv)
        rf_w = tuned["weights"]
        settings["B4_validation"] = {**tuned, "pixels": int(len(yv)), "class_counts": np.bincount(yv, minlength=3).tolist()}
        print(f"B4: class weights tuned on validation {rf_w}, val mean flood F1 {tuned['val_flood_mean_f1']:.3f}")

    root = Path(args.test_root)
    ev_stats = load_json(root / "event_stats.json", {})
    by_event = defaultdict(list)
    for it in test_items(root):
        by_event[it.event].append(it)

    conf = {m: {} for m in args.methods}
    for ev, items in by_event.items():
        ds = ChipDataset(items, ev_stats, use_aux="B4" in args.methods)
        t_water = event_water_threshold(ds, args.speckle)
        settings["water_threshold_db"][ev] = t_water
        print(f"{ev}: {len(items)} chips, Otsu water threshold {t_water:.1f} dB", flush=True)
        bounds = chip_bounds(items)
        mos = {} if args.no_maps else {m: EventMosaic(items[0].gt, ["pred"]) for m in args.methods}
        cm = {m: Confusion() for m in args.methods}
        for b in DataLoader(ds, batch_size=16, num_workers=args.workers):
            f = bl.features(bl.despeckle(b["raw"], args.speckle), b["aux"], b["coh_offset"])
            valid = np.isfinite(b["raw"].numpy()).all(1)
            y = b["label"].numpy().copy()
            y[~valid] = IGNORE
            for m in args.methods:
                pred = bl.predict_rf(rf, f, rf_w) if m == "B4" else bl.RULES[m](f, t_water)
                cm[m].update(torch.from_numpy(pred), torch.from_numpy(y))
                if m in mos:
                    pred[~valid] = IGNORE
                    for k, i in enumerate(b["index"].tolist()):
                        mos[m].put("pred", bounds[i], pred[k])
        for m in args.methods:
            conf[m][ev] = cm[m]
            if m in mos:
                mos[m].save(out / m / "test", ev)

    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps(settings, indent=2, default=float))
    for m in args.methods:
        print(f"\n=== {m}")
        save_metrics(conf[m], out / m / "test", m)


if __name__ == "__main__":
    main()

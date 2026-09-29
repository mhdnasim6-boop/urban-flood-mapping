"""Evaluate a trained run on the manually labelled test events.

    python scripts/evaluate.py --run runs/D_full --test-root /kaggle/input/ufm-test --mc 20

Writes into <run>/test[_<tag>]/: test_metrics.csv, confusion.json and per-event maps
<event>_pred.tif (0/1/2), <event>_pflood.tif (flood probability %),
<event>_uncert.tif (MC-dropout predictive entropy, % of maximum).
"""
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ufm.bands import AUX_LAYERS, IGNORE  # noqa: E402
from ufm.data import ChipDataset, test_items  # noqa: E402
from ufm.evaluation import EventMosaic, chip_bounds, save_metrics  # noqa: E402
from ufm.features import load_json, valid_mask  # noqa: E402
from ufm.metrics import Confusion  # noqa: E402
from ufm.model import build_model, predict  # noqa: E402
from ufm.utils import pick_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder containing best.pt")
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--mc", type=int, default=None, help="MC-dropout passes (0 = deterministic)")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--no-maps", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for output folder and method name, e.g. mc")
    ap.add_argument("--device")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    run = Path(args.run)
    out_dir = run / (f"test_{args.tag}" if args.tag else "test")
    ck = torch.load(run / args.ckpt, map_location="cpu", weights_only=False)
    cfg, st = ck["cfg"], ck["stats"]
    mc = cfg["eval"]["mc_samples"] if args.mc is None else args.mc
    dev = pick_device(args.device)
    model = build_model(cfg, st["mean"], st["std"], pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    amp = dev.type == "cuda"

    root = Path(args.test_root)
    ev_stats = load_json(root / "event_stats.json", {})
    items = test_items(root)
    use_aux = any(c in AUX_LAYERS for c in cfg["channels"])
    by_event = defaultdict(list)
    for it in items:
        by_event[it.event].append(it)
    print(f"{cfg['name']} (epoch {ck.get('epoch')}): {len(items)} test chips, MC passes = {mc}, device {dev}")

    per_event = {}
    for ev, ev_items in by_event.items():
        bounds = chip_bounds(ev_items)
        ds = ChipDataset(ev_items, ev_stats, use_aux=use_aux)
        dl = DataLoader(ds, batch_size=args.batch or cfg["eval"]["batch_size"], num_workers=args.workers)
        conf = Confusion()
        mos = None if args.no_maps else EventMosaic(ev_items[0].gt, ["pred", "pflood"] + (["uncert"] if mc else []))
        for b in dl:
            raw, aux = b["raw"].to(dev), b["aux"].to(dev)
            off, y = b["coh_offset"].to(dev), b["label"].to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                prob, unc = predict(model, raw, aux, off, mc_samples=mc)
            pred = prob.argmax(1)
            valid = valid_mask(raw)
            y[~valid] = IGNORE
            conf.update(pred, y)
            if mos:
                pred = pred.masked_fill(~valid, IGNORE).byte().cpu().numpy()
                pfl = (prob[:, 1:].sum(1) * 100).round().byte().cpu().numpy()
                un = (unc / math.log(prob.shape[1]) * 100).round().clamp(0, 100).byte().cpu().numpy() if unc is not None else None
                for k, i in enumerate(b["index"].tolist()):
                    mos.put("pred", bounds[i], pred[k])
                    mos.put("pflood", bounds[i], pfl[k])
                    if un is not None:
                        mos.put("uncert", bounds[i], un[k])
        per_event[ev] = conf
        if mos:
            mos.save(out_dir, ev)
        print(f"  {ev}: done", flush=True)

    save_metrics(per_event, out_dir, cfg["name"] + (f"_{args.tag}" if args.tag else ""))


if __name__ == "__main__":
    main()

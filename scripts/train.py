"""Train one experiment.

    python scripts/train.py --config configs/experiments/D_full.yaml \
        --data-root /kaggle/input/ufm-trainval --out /kaggle/working/runs

Writes <out>/<name>/: best.pt, last.pt, history.csv, config.yaml, val_best.json
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ufm import stats as ufm_stats  # noqa: E402
from ufm.bands import AUX_LAYERS  # noqa: E402
from ufm.data import ChipDataset, category_sampler, trainval_items, worker_init  # noqa: E402
from ufm.features import load_json, save_json, valid_mask  # noqa: E402
from ufm.losses import build_loss  # noqa: E402
from ufm.metrics import Confusion  # noqa: E402
from ufm.model import build_model  # noqa: E402
from ufm.utils import load_config, pick_device, set_seed  # noqa: E402


def to_dev(b, dev):
    return b["raw"].to(dev, non_blocking=True), b["aux"].to(dev, non_blocking=True), \
        b["coh_offset"].to(dev, non_blocking=True), b["label"].to(dev, non_blocking=True)


@torch.no_grad()
def validate(model, loader, dev, amp):
    model.eval()
    conf = Confusion()
    for b in loader:
        raw, aux, off, y = to_dev(b, dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            pred = model(raw, aux, off).argmax(1)
        y[~valid_mask(raw)] = 255
        conf.update(pred, y)
    return conf.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--device")
    ap.add_argument("--max-chips", type=int, default=0, help="subset for smoke tests")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("overrides", nargs="*", help="key.sub=value config overrides")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.data_root:
        cfg["data"]["trainval_root"] = args.data_root
    tc, dc = cfg["train"], cfg["data"]
    set_seed(tc["seed"])
    dev = pick_device(args.device)
    amp = bool(tc["amp"]) and dev.type == "cuda"
    run = Path(args.out) / cfg["name"]
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    root = Path(dc["trainval_root"])
    ev_stats = load_json(root / "event_stats.json", {})
    tr_items, va_items = trainval_items(root, "train"), trainval_items(root, "val")
    if args.max_chips:
        rng = np.random.default_rng(0)
        tr_items = [tr_items[i] for i in rng.permutation(len(tr_items))[: args.max_chips]]
        va_items = [va_items[i] for i in rng.permutation(len(va_items))[: max(4, args.max_chips // 4)]]
    print(f"{cfg['name']}: {len(cfg['channels'])} channels, {len(tr_items)} train / {len(va_items)} val chips on {dev}")

    st = load_json(root / "channel_stats.json") or load_json(run / "channel_stats.json")
    if st is None:
        print("computing channel statistics ...")
        st = ufm_stats.compute(tr_items, ev_stats)
        save_json(st, run / "channel_stats.json")

    use_aux = any(c in AUX_LAYERS for c in cfg["channels"])
    ds_tr = ChipDataset(tr_items, ev_stats, crop=dc["crop"], train=True,
                        focus_prob=dc["focus_prob"], use_aux=use_aux, seed=tc["seed"])
    ds_va = ChipDataset(va_items, ev_stats, use_aux=use_aux)
    shares = dc.get("category_shares")
    sampler = category_sampler(tr_items, shares, seed=tc["seed"]) if shares else None
    nw = dc["num_workers"]
    dl_tr = DataLoader(ds_tr, batch_size=tc["batch_size"], sampler=sampler, shuffle=sampler is None,
                       num_workers=nw, pin_memory=dev.type == "cuda", drop_last=True,
                       worker_init_fn=worker_init, persistent_workers=nw > 0)
    dl_va = DataLoader(ds_va, batch_size=max(1, tc["batch_size"] // 4), num_workers=nw,
                       pin_memory=dev.type == "cuda")

    model = build_model(cfg, st["mean"], st["std"], pretrained=not args.no_pretrained).to(dev)
    loss_fn = build_loss(cfg, st["class_counts"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total = tc["epochs"] * len(dl_tr)
    warm = tc.get("warmup_epochs", 1) * len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best, best_val, bad, hist = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(1, tc["epochs"] + 1):
        model.train()
        run_loss, n = torch.zeros((), device=dev), 0
        for b in dl_tr:
            raw, aux, off, y = to_dev(b, dev)
            y[~valid_mask(raw)] = 255
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(raw, aux, off), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.detach()  # no per-step GPU sync
            n += 1
        row = {"epoch": ep, "train_loss": run_loss.item() / max(n, 1), "lr": opt.param_groups[0]["lr"],
               "minutes": (time.time() - t0) / 60}
        if ep % tc["val_every"] == 0 or ep == tc["epochs"]:
            v = validate(model, dl_va, dev, amp)
            row.update({k: v[k] for k in ("flood_open_f1", "flood_urban_f1", "flood_mean_f1", "kappa")})
            score = v[tc["select_metric"]]
            if score > best:
                best, best_val, bad = score, v, 0
                torch.save({"model": model.state_dict(), "cfg": cfg, "stats": st, "epoch": ep, "val": v},
                           run / "best.pt")
            else:
                bad += 1
        hist.append(row)
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        fields = list(dict.fromkeys(k for r in hist for k in r))
        with open(run / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(hist)
        if bad >= tc["patience"]:
            print(f"early stop: no improvement in {bad} validation rounds")
            break

    torch.save({"model": model.state_dict(), "cfg": cfg, "stats": st, "epoch": ep}, run / "last.pt")
    (run / "val_best.json").write_text(json.dumps(best_val, indent=2, default=float))
    print(f"best {tc['select_metric']} = {best:.4f}; saved to {run}")


if __name__ == "__main__":
    main()

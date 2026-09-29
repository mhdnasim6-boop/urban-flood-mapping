"""End-to-end check on a tiny subset before spending Kaggle GPU time.

Downloads ~40 chips of one test event, builds aux layers, fakes a train/val
split from them, trains 2 epochs, evaluates with MC-dropout and runs B1-B4.
Metrics are meaningless (train = test); it only checks that everything runs.

    python scripts/smoke_test.py --work /tmp/ufm_smoke
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ufm.io import write_tif  # noqa: E402


def run(*cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([sys.executable, *map(str, cmd)], check=True, cwd=ROOT)


def fake_trainval(test_dir, event, out):
    """Copy test chips into the train/val layout, cutting labels from GT_full."""
    ev = test_dir / event
    with rasterio.open(ev / "GT_full.tif") as g:
        gt_t, gt = g.transform, g
        names = []
        for sar in sorted((ev / "SAR").glob("*_SAR.tif")):
            with rasterio.open(sar) as s:
                prof, b = s.profile, s.bounds
            w = from_bounds(*b, transform=gt_t).round_offsets().round_lengths()
            lab = gt.read(1, window=w, boundless=True, fill_value=255)
            cat = "03_FU" if (lab == 2).any() else "02_FO" if (lab == 1).any() else "01_NF"
            name = sar.name.replace("_SAR.tif", "")
            for sub in ("SAR", "GT", "AUX"):
                (out / cat / sub).mkdir(parents=True, exist_ok=True)
            shutil.copy(sar, out / cat / "SAR" / sar.name)
            aux = ev / "AUX" / f"{name}_AUX.tif"
            if aux.exists():
                shutil.copy(aux, out / cat / "AUX" / aux.name)
            write_tif(out / cat / "GT" / f"{name}_GT.tif", lab, prof, "uint8", nodata=255)
            names.append(f"../{cat}/GT/{name}_GT.tif")
    val = names[::4]
    (out / "Valid_dataset.txt").write_text("\n".join(val))
    (out / "Train_dataset.txt").write_text("\n".join(n for n in names if n not in val))
    shutil.copy(test_dir / "event_stats.json", out / "event_stats.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/ufm_smoke")
    ap.add_argument("--event", default="20231201_Jubba_1")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    w = Path(args.work)
    test, tv, runs = w / "test", w / "trainval", w / "runs"

    run("scripts/prepare_test.py", "--out", test, "--events", args.event, "--max-chips", 40, "--threads", 8)
    run("scripts/build_aux.py", "--root", test, "--cache", w / "tiles", "--workers", 2)
    if tv.exists():
        shutil.rmtree(tv)
    fake_trainval(test, args.event, tv)
    run("scripts/compute_stats.py", "--root", tv)
    for cfg in ("B5_authors_baseline", "D_full"):
        run("scripts/train.py", "--config", f"configs/experiments/{cfg}.yaml", "--data-root", tv,
            "--out", runs, "--device", args.device, "train.epochs=2", "train.val_every=1",
            "train.batch_size=4", "data.num_workers=0")
        run("scripts/evaluate.py", "--run", runs / cfg, "--test-root", test, "--mc", 0,
            "--device", args.device, "--workers", 0, "--batch", 8)
    run("scripts/evaluate.py", "--run", runs / "D_full", "--test-root", test, "--mc", 3, "--tag", "mc",
        "--device", args.device, "--workers", 0, "--batch", 8)
    run("scripts/run_baselines.py", "--test-root", test, "--trainval-root", tv, "--out", runs / "baselines",
        "--rf-device", "cpu", "--rf-max-chips", 30, "--workers", 0)
    run("scripts/summarize.py", "--runs", runs, "--test-root", test, "--out", w / "report")
    print("\nsmoke test passed")


if __name__ == "__main__":
    main()

"""Download the three manually labelled test events (four image-label pairs).

These events were never used for training and their labels were drawn by hand
from PlanetScope / UAV imagery, so they give an independent accuracy estimate.

    python scripts/prepare_test.py --out data/test [--max-chips 40 --events 20231201_Jubba_1]

Layout written:
    <out>/<event>/SAR/<chip>_SAR.tif   uint8 SAR chips (256x256)
    <out>/<event>/GT_full.tif          uint8 full-event reference map
    <out>/event_stats.json
"""
import argparse
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ufm.bands import decode_sar, encode_label, encode_sar  # noqa: E402
from ufm.features import event_offsets, load_json, save_json, stable_drop_sample  # noqa: E402
from ufm.io import HF_RESOLVE, fetch_bytes, hf_list, read_tif_bytes, write_tif  # noqa: E402

EVENTS = ["20210727_Weihui", "20230609_NovaKakhovka", "20231201_Jubba_1", "20231201_Jubba_2"]


def build_gt(event, ev_dir):
    out = ev_dir / "GT_full.tif"
    if out.exists():
        return out
    src = [f["path"] for f in hf_list(f"testing_case_orig/{event}") if f["path"].endswith("_GT.tif")][0]
    tmp = ev_dir / "_gt_download.tif"
    with urllib.request.urlopen(HF_RESOLVE + src, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=16 << 20)
    with rasterio.open(tmp) as g:
        prof, lab = g.profile, encode_label(g.read(1))
    write_tif(out, lab, prof, "uint8", nodata=255)
    tmp.unlink()
    u, c = np.unique(lab, return_counts=True)
    print(f"{event}: GT {lab.shape}, pixels per class {dict(zip(u.tolist(), c.tolist()))}")
    return out


def pick_chips(files, gt_path, n, seed=0):
    """For quick local tests: half the chips with most flood pixels, half random."""
    with rasterio.open(gt_path) as g:
        gt = g.read(1)
    score = []
    for f in files:
        r, c = map(int, f.split("_ID_")[1].split("_")[:2])
        t = gt[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256]
        score.append((t == 1).sum() + 20 * (t == 2).sum())
    order = np.argsort(score)[::-1]
    rng = np.random.default_rng(seed)
    top = list(order[: n // 2])
    rest = rng.choice(order[n // 2:], size=n - len(top), replace=False).tolist()
    return [files[i] for i in top + rest]


def convert_chip(path, sar_dir, rng_seed):
    dst = sar_dir / Path(path).name
    if dst.exists():
        with rasterio.open(dst) as s:
            q = s.read()
    else:
        arr, prof = read_tif_bytes(fetch_bytes(HF_RESOLVE + path, tiff=True))
        q = encode_sar(arr)
        write_tif(dst, q, prof, "uint8", nodata=0)
    return stable_drop_sample(decode_sar(q), n=1000, rng=np.random.default_rng(rng_seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--events", nargs="*", default=EVENTS)
    ap.add_argument("--max-chips", type=int, default=0, help="per event, for quick tests")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    stats = load_json(out / "event_stats.json", {}) or {}
    for ev in args.events:
        ev_dir = out / ev
        (ev_dir / "SAR").mkdir(parents=True, exist_ok=True)
        gt = build_gt(ev, ev_dir)
        listing = [f for f in hf_list(f"testing_case_256/{ev}") if f["path"].endswith("_SAR.tif")]
        files = sorted(f["path"] for f in listing if f["size"] > 0)
        if len(files) < len(listing):  # known defect: some published chips are 0-byte files
            print(f"{ev}: skipping {len(listing) - len(files)} empty chip files on the server")
        if args.max_chips:
            files = pick_chips(files, gt, args.max_chips)
        with ThreadPoolExecutor(args.threads) as ex:
            samples = list(ex.map(lambda a: convert_chip(a[1], ev_dir / "SAR", a[0]), enumerate(files)))
        stats.update(event_offsets({ev: samples}))
        save_json(stats, out / "event_stats.json")
        print(f"{ev}: {len(files)} chips; coherence-drop offset {stats[ev]}", flush=True)


if __name__ == "__main__":
    main()

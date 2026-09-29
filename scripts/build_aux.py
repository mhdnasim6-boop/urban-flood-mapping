"""Add HAND, slope and permanent-water layers for every SAR chip under --root.

Writes <...>/AUX/<chip>_AUX.tif (uint8, 3 bands) next to each <...>/SAR folder.
Tiles are downloaded per event into --cache and reused across chips.

    python scripts/build_aux.py --root data/trainval --cache /tmp/tiles
"""
import argparse
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ufm.aux_layers import TileCache, aux_for_chip, dem_url, hand_url, jrc_url, tiles_for_bounds  # noqa: E402
from ufm.bands import encode_aux, event_of  # noqa: E402
from ufm.io import write_tif  # noqa: E402


def aux_path(sar_path):
    return sar_path.parent.parent / "AUX" / sar_path.name.replace("_SAR.tif", "_AUX.tif")


def process_chip(sar_path, cache_dir):
    out = aux_path(sar_path)
    if out.exists():
        return 0
    with rasterio.open(sar_path) as src:
        prof, bounds, res = src.profile, src.bounds, src.res
    hand, slope, occ = aux_for_chip(TileCache(cache_dir), bounds, res)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_tif(out, encode_aux(hand, slope, occ), prof, "uint8", nodata=255)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", default="/tmp/ufm_tiles")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-cache-gb", type=float, default=10)
    args = ap.parse_args()

    chips = sorted(Path(args.root).rglob("SAR/*_SAR.tif"))
    by_event = defaultdict(list)
    for p in chips:
        by_event[event_of(p.name)].append(p)
    print(f"{len(chips)} chips in {len(by_event)} events")
    cache = TileCache(args.cache)
    t0 = time.time()
    for ev, paths in sorted(by_event.items()):
        todo = [p for p in paths if not aux_path(p).exists()]
        if not todo:
            continue
        one, ten = set(), set()
        for p in todo:
            with rasterio.open(p) as src:
                a, b = tiles_for_bounds(src.bounds)
            one |= a
            ten |= b
        urls = [hand_url(*t) for t in one] + [dem_url(*t) for t in one] + [jrc_url(*t) for t in ten]
        with ThreadPoolExecutor(8) as ex:  # download once, before workers read
            list(ex.map(cache.get, urls))
        with Pool(args.workers) as pool:
            n = sum(pool.map(partial(process_chip, cache_dir=args.cache), todo, chunksize=8))
        print(f"{ev}: {n} chips, {len(urls)} tiles | {time.time() - t0:.0f}s", flush=True)
        size_gb = sum(f.stat().st_size for f in Path(args.cache).glob("*")) / 1e9
        if size_gb > args.max_cache_gb:
            shutil.rmtree(args.cache)
            cache = TileCache(args.cache)


if __name__ == "__main__":
    main()

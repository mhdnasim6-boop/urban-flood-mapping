"""Stream the UrbanSARFloods train/val archive and store it in compact form.

The 38 GB .tar.gz is never written to disk: each chip is decoded, re-encoded
as uint8 (SAR) / uint8 (labels) with deflate compression, and written out.
Output (~13 GB) fits in Kaggle's 20 GB /kaggle/working.

    python scripts/prepare_train.py --out data/trainval [--max-members N]

Layout written:
    <out>/{01_NF,02_FO,03_FU}/SAR/<chip>_SAR.tif   uint8, 8 bands
    <out>/{01_NF,02_FO,03_FU}/GT/<chip>_GT.tif     uint8, 0/1/2, 255 = ignore
    <out>/Train_dataset.txt, Valid_dataset.txt, data_norm.txt (copied)
    <out>/event_stats.json   event-level coherence drop of stable pixels
"""
import argparse
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ufm.bands import decode_sar, encode_label, encode_sar, event_of  # noqa: E402
from ufm.features import event_offsets, load_json, save_json, stable_drop_sample  # noqa: E402
from ufm.io import HF_RESOLVE, ResumableHTTPStream, read_tif_bytes, write_tif  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default=HF_RESOLVE + "urban_sar_floods.tar.gz",
                    help="URL or local path of urban_sar_floods.tar.gz")
    ap.add_argument("--max-members", type=int, default=0, help="stop early (for testing)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = defaultdict(list)
    rng = np.random.default_rng(0)
    # keep samples from an interrupted earlier run so event stats stay complete
    partial = out / "_stable_samples.npz"
    if partial.exists():
        for k, v in np.load(partial).items():
            samples[k].append(v)

    fobj = open(args.source, "rb") if Path(args.source).exists() else ResumableHTTPStream(args.source)
    t0, n, n_sar, n_gt = time.time(), 0, 0, 0
    with tarfile.open(fileobj=fobj, mode="r|gz") as tf:
        for m in tf:
            n += 1
            if args.max_members and n > args.max_members:
                break
            if not m.isfile():
                continue
            rel = Path(m.name).relative_to("urban_sar_floods")
            dst = out / rel
            if rel.suffix == ".txt":
                dst.write_bytes(tf.extractfile(m).read())
                continue
            if rel.suffix != ".tif":
                continue
            data = tf.extractfile(m).read()
            if dst.exists():
                continue  # already converted in an earlier (interrupted) run
            dst.parent.mkdir(parents=True, exist_ok=True)
            arr, prof = read_tif_bytes(data)
            if rel.parent.name == "SAR":
                q = encode_sar(arr)
                write_tif(dst, q, prof, "uint8", nodata=0)
                samples[event_of(rel.name)].append(stable_drop_sample(decode_sar(q), n=1000, rng=rng))
                n_sar += 1
            elif rel.parent.name == "GT":
                write_tif(dst, encode_label(arr[0]), prof, "uint8", nodata=255)
                n_gt += 1
            if (n_sar + n_gt) % 250 == 0:
                gb = fobj.pos / 1e9 if hasattr(fobj, "pos") else float("nan")
                print(f"{n_sar} SAR, {n_gt} GT | {gb:.1f} GB streamed | {time.time() - t0:.0f}s", flush=True)
                np.savez(partial, **{k: np.concatenate(v) for k, v in samples.items()})

    stats = load_json(out / "event_stats.json", {}) or {}
    stats.update(event_offsets(samples))
    save_json(stats, out / "event_stats.json")
    np.savez(partial, **{k: np.concatenate(v) for k, v in samples.items()})
    print(f"done: {n_sar} SAR, {n_gt} GT chips in {time.time() - t0:.0f}s; events: {sorted(stats)}")


if __name__ == "__main__":
    main()

"""Compute channel normalisation statistics and class counts from training chips.

    python scripts/compute_stats.py --root data/trainval

Writes <root>/channel_stats.json (run once during data preparation, because
Kaggle input datasets are read-only).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ufm import stats  # noqa: E402
from ufm.data import trainval_items  # noqa: E402
from ufm.features import load_json, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    items = trainval_items(root, "train")
    st = stats.compute(items, load_json(root / "event_stats.json", {}))
    save_json(st, root / "channel_stats.json")
    print(f"{len(items)} training chips; class pixel counts {st['class_counts']}")
    for c in st["mean"]:
        print(f"  {c:15s} mean {st['mean'][c]:9.3f}  std {st['std'][c]:8.3f}")


if __name__ == "__main__":
    main()

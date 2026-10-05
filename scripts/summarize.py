"""Collect all test metrics into one comparison table and make figures.

    python scripts/summarize.py --runs /kaggle/working/runs --test-root /kaggle/input/ufm-test --out report

Writes: comparison.csv / comparison.md (all methods x events), f1_bars.png,
and maps_<event>.png (SAR, reference, every method, probability, uncertainty).

Pooled scores are recomputed from the per-event confusion matrices, leaving out
any --exclude-events. Runs named <method>_s<k> are extra random seeds of
<method>: the table reports their mean and standard deviation.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from rasterio.windows import Window, from_bounds  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ufm.bands import decode_sar  # noqa: E402
from ufm.metrics import Confusion  # noqa: E402

SEED_RE = re.compile(r"^(.*)_s\d+$")

ORDER = ["B1", "B2", "B3", "B4", "B5_authors_baseline", "A_raw", "B_raw_phys", "C_raw_aux",
         "D_full", "D_full_mc", "E_no_coherence", "F_no_vh"]
LABELS = {
    "B1": "B1 Intensity change (split Otsu)", "B2": "B2 Coherence threshold 0.3",
    "B3": "B3 Rule-based tree", "B4": "B4 Random forest (17 inputs)",
    "B5_authors_baseline": "B5 U-Net, 8 bands, WCE (dataset authors)",
    "A_raw": "A U-Net 8 bands + focal-Dice", "B_raw_phys": "B + physics features",
    "C_raw_aux": "C + HAND/slope/water", "D_full": "D Proposed (17 channels)",
    "D_full_mc": "D Proposed + MC-dropout (20 passes)",
    "E_no_coherence": "E Proposed without coherence", "F_no_vh": "F Proposed without VH",
}
CLASS_CMAP = ListedColormap(["#f2f2f2", "#1f77b4", "#d62728"])


def base_method(m):
    k = SEED_RE.match(m)
    return k.group(1) if k else m


def find_runs(runs):
    """{method: folder} for every folder holding test_metrics.csv + confusion.json."""
    files = list(Path(runs).glob("*/test*/test_metrics.csv")) + list(Path(runs).glob("baselines/*/test/test_metrics.csv"))
    return {pd.read_csv(f, nrows=1)["method"].iloc[0]: f.parent for f in files}


def metrics_table(run_dirs, exclude):
    rows = []
    for method, folder in run_dirs.items():
        pooled = Confusion()
        for ev, m in json.loads((folder / "confusion.json").read_text()).items():
            if ev == "ALL":
                continue
            c = Confusion()
            c.m = np.array(m, dtype=np.int64)
            rows.append({"method": method, "event": ev, **c.summary()})
            if ev not in exclude:
                pooled += c
        rows.append({"method": method, "event": "ALL", **pooled.summary()})
    df = pd.DataFrame(rows)
    df["base"] = df["method"].map(base_method)
    df["order"] = df["base"].map({m: i for i, m in enumerate(ORDER)}).fillna(99)
    return df.sort_values(["order", "method", "event"])


def comparison_table(df, out, exclude):
    cols = {"flood_open_f1": "FO F1", "flood_urban_f1": "FU F1", "flood_mean_f1": "Mean flood F1",
            "flood_any_f1": "Any-flood F1", "flood_urban_precision": "FU precision",
            "flood_urban_recall": "FU recall", "kappa": "Kappa", "overall_accuracy": "OA"}
    pooled = df[df.event == "ALL"]
    g = pooled.groupby("base")
    t = g[list(cols)].mean().rename(columns=cols)
    t.insert(0, "Runs", g.size())
    t.insert(2, "FO F1 sd", g["flood_open_f1"].std())
    t.insert(4, "FU F1 sd", g["flood_urban_f1"].std())
    ev = df[(df.event != "ALL") & ~df.event.isin(exclude)]
    per_ev = ev.pivot_table(index="base", columns="event", values=["flood_open_f1", "flood_urban_f1"], aggfunc="mean")
    per_ev.columns = [f"{'FO' if a == 'flood_open_f1' else 'FU'} F1 {e}" for a, e in per_ev.columns]
    t = t.join(per_ev)
    t = t.loc[[m for m in ORDER if m in t.index] + [m for m in t.index if m not in ORDER]]
    t.index = [LABELS.get(m, m) for m in t.index]
    t.to_csv(out / "comparison.csv", float_format="%.4f")
    fmt = t.copy().astype(object)
    for c in t.columns:
        fmt[c] = [str(int(v)) if c == "Runs" else ("" if pd.isna(v) else f"{v:.3f}") for v in t[c]]
    note = f"\n\nPooled over test events excluding: {', '.join(exclude) or 'none'}. " \
           "sd = standard deviation over random seeds (blank = single run).\n"
    (out / "comparison.md").write_text(fmt.to_markdown() + note)
    print(fmt.iloc[:, :8].to_string())
    return t


def f1_bars(t, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(t))
    kw = dict(capsize=3, error_kw={"elinewidth": 1})
    ax.bar(x - 0.2, t["FO F1"], 0.4, yerr=t["FO F1 sd"].fillna(0), label="Flooded open area", color="#1f77b4", **kw)
    ax.bar(x + 0.2, t["FU F1"], 0.4, yerr=t["FU F1 sd"].fillna(0), label="Flooded urban area", color="#d62728", **kw)
    ax.set_xticks(x, t.index, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("F1 (test events pooled)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "f1_bars.png", dpi=200)
    plt.close(fig)


def best_window(gt_path, size):
    with rasterio.open(gt_path) as g:
        gt = g.read(1, out_shape=(g.height // 8, g.width // 8), resampling=rasterio.enums.Resampling.nearest)
    score = (gt == 1).astype(np.float32) + 20 * (gt == 2)
    k = size // 8
    best, pos = -1, (0, 0)
    for r in range(0, max(1, score.shape[0] - k), k // 2):
        for c in range(0, max(1, score.shape[1] - k), k // 2):
            s = score[r:r + k, c:c + k].sum()
            if s > best:
                best, pos = s, (r * 8, c * 8)
    return Window(pos[1], pos[0], size, size)


def sar_window(event_dir, gt_path, win):
    """Assemble post-event VV (dB) for the window from the chips."""
    with rasterio.open(gt_path) as g:
        wt = rasterio.windows.transform(win, g.transform)
    img = np.full((int(win.height), int(win.width)), np.nan, np.float32)
    for sar in (event_dir / "SAR").glob("*_SAR.tif"):
        with rasterio.open(sar) as s:
            w = from_bounds(*s.bounds, transform=wt).round_offsets().round_lengths()
            r0, c0 = int(w.row_off), int(w.col_off)
            if r0 >= img.shape[0] or c0 >= img.shape[1] or r0 + s.height <= 0 or c0 + s.width <= 0:
                continue
            v = decode_sar(s.read())[7]
            rr, cc = max(0, -r0), max(0, -c0)
            r1, c1 = min(img.shape[0], r0 + s.height), min(img.shape[1], c0 + s.width)
            img[max(r0, 0):r1, max(c0, 0):c1] = v[rr:rr + r1 - max(r0, 0), cc:cc + c1 - max(c0, 0)]
    return img


def map_figure(event, test_root, run_dirs, out, size=1024):
    gt_path = Path(test_root) / event / "GT_full.tif"
    win = best_window(gt_path, size)
    panels = [("Post-event VV (dB)", sar_window(Path(test_root) / event, gt_path, win), "sar")]
    with rasterio.open(gt_path) as g:
        panels.append(("Reference", g.read(1, window=win, boundless=True, fill_value=255), "cls"))
    for m in [m for m in ORDER if m in run_dirs and m != "D_full_mc"]:
        p = run_dirs[m] / f"{event}_pred.tif"
        if p.exists():
            with rasterio.open(p) as s:
                panels.append((LABELS.get(m, m), s.read(1, window=win, boundless=True, fill_value=255), "cls"))
    for layer, title in (("pflood", "D flood probability"), ("uncert", "D uncertainty")):
        d = run_dirs.get("D_full_mc", run_dirs.get("D_full", Path("-")))
        p = d / f"{event}_{layer}.tif"
        if p.exists():
            with rasterio.open(p) as s:
                panels.append((title, s.read(1, window=win, boundless=True, fill_value=255), layer))
    n = len(panels)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for ax, (title, a, kind) in zip(np.ravel(axes), panels):
        if kind == "sar":
            ax.imshow(a, cmap="gray", vmin=-25, vmax=0)
        elif kind == "cls":
            ax.imshow(np.ma.masked_equal(a, 255), cmap=CLASS_CMAP, vmin=0, vmax=2, interpolation="nearest")
        else:
            im = ax.imshow(np.ma.masked_equal(a, 255), cmap="magma" if kind == "uncert" else "Blues", vmin=0, vmax=100)
            fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle(f"{event}  (blue = flooded open, red = flooded urban)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"maps_{event}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--out", default="report")
    ap.add_argument("--size", type=int, default=1024, help="map window size in pixels")
    ap.add_argument("--exclude-events", nargs="*", default=[],
                    help="events left out of pooled scores and per-event columns")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dirs = find_runs(args.runs)
    df = metrics_table(run_dirs, args.exclude_events)
    df.drop(columns="order").to_csv(out / "all_metrics.csv", index=False)
    t = comparison_table(df, out, args.exclude_events)
    f1_bars(t, out)
    for ev in sorted(df.event.unique()):
        if ev != "ALL" and ev not in args.exclude_events and (Path(args.test_root) / ev / "GT_full.tif").exists():
            map_figure(ev, args.test_root, run_dirs, out, args.size)
    print(f"report written to {out}")


if __name__ == "__main__":
    main()

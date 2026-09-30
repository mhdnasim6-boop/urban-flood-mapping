"""Collect all test metrics into one comparison table and make figures.

    python scripts/summarize.py --runs /kaggle/working/runs --test-root /kaggle/input/ufm-test --out report

Writes: comparison.csv / comparison.md (all methods x events), f1_bars.png,
and maps_<event>.png (SAR, reference, every method, probability, uncertainty).
"""
import argparse
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

ORDER = ["B1", "B2", "B3", "B4", "B5_authors_baseline", "A_raw", "B_raw_phys", "C_raw_aux",
         "D_full", "D_full_mc", "E_no_coherence", "F_no_vh"]
LABELS = {
    "B1": "B1 Intensity change (Otsu)", "B2": "B2 Coherence threshold 0.3",
    "B3": "B3 Rule-based tree", "B4": "B4 Random forest (17 inputs)",
    "B5_authors_baseline": "B5 U-Net, 8 bands, WCE (dataset authors)",
    "A_raw": "A U-Net 8 bands + focal-Dice", "B_raw_phys": "B + physics features",
    "C_raw_aux": "C + HAND/slope/water", "D_full": "D Proposed (17 channels)",
    "D_full_mc": "D Proposed + MC-dropout (20 passes)",
    "E_no_coherence": "E Proposed without coherence", "F_no_vh": "F Proposed without VH",
}
CLASS_CMAP = ListedColormap(["#f2f2f2", "#1f77b4", "#d62728"])


def find_metrics(runs):
    files = list(Path(runs).glob("*/test*/test_metrics.csv")) + list(Path(runs).glob("baselines/*/test/test_metrics.csv"))
    dfs = [pd.read_csv(f) for f in files]
    run_dirs = {d["method"].iloc[0]: f.parent for d, f in zip(dfs, files)}
    df = pd.concat(dfs, ignore_index=True)
    df["order"] = df["method"].map({m: i for i, m in enumerate(ORDER)}).fillna(99)
    return df.sort_values(["order", "event"]), run_dirs


def comparison_table(df, out):
    pooled = df[df.event == "ALL"].set_index("method")
    cols = {"flood_open_f1": "FO F1", "flood_urban_f1": "FU F1", "flood_mean_f1": "Mean flood F1",
            "flood_any_f1": "Any-flood F1", "flood_urban_precision": "FU precision",
            "flood_urban_recall": "FU recall", "kappa": "Kappa", "overall_accuracy": "OA"}
    t = pooled[list(cols)].rename(columns=cols)
    per_ev = df[df.event != "ALL"].pivot_table(index="method", columns="event",
                                               values=["flood_open_f1", "flood_urban_f1"])
    per_ev.columns = [f"{'FO' if a == 'flood_open_f1' else 'FU'} F1 {e}" for a, e in per_ev.columns]
    t = t.join(per_ev)
    t = t.loc[[m for m in ORDER if m in t.index] + [m for m in t.index if m not in ORDER]]
    t.index = [LABELS.get(m, m) for m in t.index]
    t.to_csv(out / "comparison.csv", float_format="%.4f")
    (out / "comparison.md").write_text(t.to_markdown(floatfmt=".3f"))
    print(t.iloc[:, :7].to_string(float_format=lambda v: f"{v:.3f}"))
    return t


def f1_bars(t, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t["FO F1"], 0.4, label="Flooded open area", color="#1f77b4")
    ax.bar(x + 0.2, t["FU F1"], 0.4, label="Flooded urban area", color="#d62728")
    ax.set_xticks(x, t.index, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("F1 (all test events pooled)")
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
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, run_dirs = find_metrics(args.runs)
    df.drop(columns="order").to_csv(out / "all_metrics.csv", index=False)
    t = comparison_table(df, out)
    f1_bars(t, out)
    for ev in sorted(df.event.unique()):
        if ev != "ALL" and (Path(args.test_root) / ev / "GT_full.tif").exists():
            map_figure(ev, args.test_root, run_dirs, out, args.size)
    print(f"report written to {out}")


if __name__ == "__main__":
    main()

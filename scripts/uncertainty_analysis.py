"""Does the MC-dropout uncertainty tell us where the map is wrong?

    python scripts/uncertainty_analysis.py --maps runs/D_full/test_mc --test-root data/test \
        --out report --exclude-events 20231201_Jubba_2

Uses <event>_pred.tif and <event>_uncert.tif from evaluate.py --mc N --tag mc.
Writes uncertainty.md / uncertainty.csv / uncertainty.png:
  * error rate per uncertainty bin
  * share of all errors found in the most uncertain k% of pixels
  * AUROC of uncertainty as an error detector
  * flood F1 after setting aside the most uncertain pixels for manual checking
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
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ufm.metrics import Confusion  # noqa: E402

REJECT = [0, 1, 2, 5, 10, 20]  # % of most uncertain pixels set aside


def load_event(maps, test_root, ev):
    with rasterio.open(Path(test_root) / ev / "GT_full.tif") as g:
        gt = g.read(1)
    with rasterio.open(maps / f"{ev}_pred.tif") as p:
        pred = p.read(1)
    with rasterio.open(maps / f"{ev}_uncert.tif") as u:
        unc = u.read(1)
    ok = (gt != 255) & (pred != 255) & (unc != 255)
    return gt[ok], pred[ok], unc[ok]


def analyse(gt, pred, unc, rng):
    wrong = gt != pred
    res = {"pixels": int(gt.size), "error_rate": float(wrong.mean())}
    # AUROC on a subsample (ranks only; 5 M pixels is plenty)
    idx = rng.choice(gt.size, size=min(5_000_000, gt.size), replace=False)
    res["auroc_error_detection"] = float(roc_auc_score(wrong[idx], unc[idx])) if wrong[idx].any() else float("nan")
    order = np.argsort(-unc.astype(np.int16), kind="stable")  # most uncertain first
    for k in REJECT:
        n = int(round(gt.size * k / 100))
        keep = np.ones(gt.size, bool)
        keep[order[:n]] = False
        c = Confusion()
        c.m = np.bincount(gt[keep].astype(np.int64) * 3 + pred[keep], minlength=9).reshape(3, 3)
        s = c.summary()
        res[f"errors_in_top{k}pct"] = float(wrong[order[:n]].sum() / max(wrong.sum(), 1)) if k else 0.0
        res[f"fo_f1_reject{k}pct"] = s["flood_open_f1"]
        res[f"fu_f1_reject{k}pct"] = s["flood_urban_f1"]
    bins = np.minimum(unc // 10, 9)
    n_bin = np.bincount(bins, minlength=10)
    e_bin = np.bincount(bins, weights=wrong, minlength=10)
    res["bins"] = [(int(b * 10), int(n_bin[b]), float(e_bin[b] / n_bin[b]) if n_bin[b] else float("nan")) for b in range(10)]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True, help="folder with <event>_pred.tif and <event>_uncert.tif")
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--out", default="report")
    ap.add_argument("--exclude-events", nargs="*", default=[])
    args = ap.parse_args()
    maps, out = Path(args.maps), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    events = sorted(p.name.replace("_uncert.tif", "") for p in maps.glob("*_uncert.tif"))
    events = [e for e in events if e not in args.exclude_events]
    results, pooled = {}, [[], [], []]
    for ev in events:
        gt, pred, unc = load_event(maps, args.test_root, ev)
        results[ev] = analyse(gt, pred, unc, rng)
        # subsample for the pooled analysis to bound memory
        idx = rng.choice(gt.size, size=min(20_000_000, gt.size), replace=False)
        for lst, a in zip(pooled, (gt, pred, unc)):
            lst.append(a[idx])
        print(f"{ev}: AUROC {results[ev]['auroc_error_detection']:.3f}, "
              f"errors in top 10% uncertain = {results[ev]['errors_in_top10pct']:.1%}", flush=True)
    results["ALL"] = analyse(*(np.concatenate(l) for l in pooled), rng)

    rows = [{"event": ev, **{k: v for k, v in r.items() if k != "bins"}} for ev, r in results.items()]
    df = pd.DataFrame(rows)
    df.to_csv(out / "uncertainty.csv", index=False)
    show = ["event", "error_rate", "auroc_error_detection", "errors_in_top5pct", "errors_in_top10pct",
            "fu_f1_reject0pct", "fu_f1_reject10pct", "fo_f1_reject0pct", "fo_f1_reject10pct"]
    (out / "uncertainty.md").write_text(df[show].to_markdown(index=False, floatfmt=".3f") +
                                        "\n\nerrors_in_topK = share of all misclassified pixels that fall in the K% most "
                                        "uncertain pixels; *_rejectK = F1 on the remaining pixels after setting those aside.\n")
    print(df[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    b = results["ALL"]["bins"]
    axes[0].bar([x[0] + 5 for x in b], [x[2] for x in b], width=8, color="#6a3d9a")
    ax2 = axes[0].twinx()
    ax2.plot([x[0] + 5 for x in b], [x[1] / results["ALL"]["pixels"] for x in b], "k.--", label="share of pixels")
    ax2.set_yscale("log")
    ax2.set_ylabel("share of pixels (log)")
    axes[0].set_xlabel("MC-dropout uncertainty (% of maximum entropy)")
    axes[0].set_ylabel("error rate")
    axes[0].set_title("Error rate by uncertainty level (all events)")
    for ev, r in results.items():
        if ev != "ALL":
            axes[1].plot(REJECT, [r[f"fu_f1_reject{k}pct"] for k in REJECT], "o-", lw=0.8, alpha=0.6,
                         label=f"{ev} urban")
    r = results["ALL"]
    axes[1].plot(REJECT, [r[f"fu_f1_reject{k}pct"] for k in REJECT], "o-", lw=2.5, color="#d62728", label="ALL urban")
    axes[1].plot(REJECT, [r[f"fo_f1_reject{k}pct"] for k in REJECT], "s--", lw=2.5, color="#1f77b4", label="ALL open")
    axes[1].set_xlabel("most uncertain pixels set aside (%)")
    axes[1].set_ylabel("F1 on remaining pixels")
    axes[1].set_title("Accuracy after deferring uncertain pixels")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "uncertainty.png", dpi=200)
    plt.close(fig)
    print(f"written to {out}")


if __name__ == "__main__":
    main()

"""Confusion-matrix metrics: per-class precision/recall/F1/IoU, OA, Cohen's kappa."""
import numpy as np
import torch

from .bands import CLASSES, IGNORE


class Confusion:
    def __init__(self, n=len(CLASSES)):
        self.n = n
        self.m = np.zeros((n, n), dtype=np.int64)  # rows = reference, cols = prediction

    def update(self, pred, target):
        pred, target = torch.as_tensor(pred).flatten(), torch.as_tensor(target).flatten()
        keep = target != IGNORE
        idx = target[keep].long() * self.n + pred[keep].long()
        self.m += torch.bincount(idx.cpu(), minlength=self.n ** 2).reshape(self.n, self.n).numpy()

    def __iadd__(self, other):
        self.m += other.m
        return self

    def summary(self):
        m = self.m.astype(np.float64)
        tp = np.diag(m)
        ref, pred, total = m.sum(1), m.sum(0), m.sum()
        out = {"pixels": int(total)}
        for k, name in enumerate(CLASSES):
            p = tp[k] / pred[k] if pred[k] else 0.0
            r = tp[k] / ref[k] if ref[k] else float("nan")
            out[f"{name}_precision"] = p
            out[f"{name}_recall"] = r
            out[f"{name}_f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            out[f"{name}_iou"] = tp[k] / (ref[k] + pred[k] - tp[k]) if (ref[k] + pred[k] - tp[k]) else float("nan")
            out[f"{name}_ref_pixels"] = int(ref[k])
        oa = tp.sum() / total if total else float("nan")
        pe = (ref * pred).sum() / total ** 2 if total else float("nan")
        out["overall_accuracy"] = oa
        out["kappa"] = (oa - pe) / (1 - pe) if total and pe < 1 else float("nan")
        # any-flood vs non-flood
        ftp = m[1:, 1:].sum()
        fp_, fr_ = m[:, 1:].sum(), m[1:, :].sum()
        bp = ftp / fp_ if fp_ else 0.0
        br = ftp / fr_ if fr_ else float("nan")
        out["flood_any_f1"] = 2 * bp * br / (bp + br) if (bp + br) > 0 else 0.0
        out["flood_mean_f1"] = np.nanmean([out["flood_open_f1"], out["flood_urban_f1"]])
        return out

"""Evaluation metrics implemented with standard library."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple


def _calc_confusion(y_true: Iterable[bool], y_pred: Iterable[bool]) -> Tuple[int, int, int, int]:
    tp = fp = fn = tn = 0
    for truth, pred in zip(y_true, y_pred):
        if truth and pred:
            tp += 1
        elif not truth and pred:
            fp += 1
        elif truth and not pred:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def precision_recall_f1(y_true: Iterable[bool], y_pred: Iterable[bool]) -> Tuple[float, float, float]:
    tp, fp, fn, _ = _calc_confusion(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def cohen_kappa(y_true: Iterable[bool], y_pred: Iterable[bool]) -> float:
    tp, fp, fn, tn = _calc_confusion(y_true, y_pred)
    total = tp + fp + fn + tn
    if total == 0:
        return 0.0
    po = (tp + tn) / total
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (total * total)
    if pe == 1:
        return 0.0
    return (po - pe) / (1 - pe)


def road_pr_f1(gt_status: Dict[str, str], pred_status: Dict[str, str]) -> Dict[str, float]:
    segments = list(set(gt_status) | set(pred_status))
    y_true = [gt_status.get(seg, "open") != "open" for seg in segments]
    y_pred = [pred_status.get(seg, "open") != "open" for seg in segments]
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    return {"precision": precision, "recall": recall, "f1": f1}

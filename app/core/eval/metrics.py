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


def context_precision_recall(relevant_ids: Iterable[str], retrieved_ids: Iterable[str]) -> Tuple[float, float]:
    """
    Calculate Context Precision and Recall based on document IDs.
    
    Precision = (Relevant ∩ Retrieved) / Retrieved
    Recall = (Relevant ∩ Retrieved) / Relevant
    """
    rel_set = set(relevant_ids)
    ret_set = set(retrieved_ids)
    if not rel_set:
        return 0.0, 0.0
        
    intersection = len(rel_set & ret_set)
    precision = intersection / len(ret_set) if ret_set else 0.0
    recall = intersection / len(rel_set) if rel_set else 0.0
    
    return precision, recall


def simple_rouge_l(output: str, reference: str) -> float:
    """
    Calculate a simplified ROUGE-L score (Longest Common Subsequence) based on token overlap.
    This is a lightweight proxy for generation quality.
    """
    if not output or not reference:
        return 0.0
        
    out_tokens = output.lower().split()
    ref_tokens = reference.lower().split()
    
    # Dynamic programming to find LCS length
    m, n = len(out_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if out_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                
    lcs_len = dp[m][n]
    
    if not lcs_len:
        return 0.0
        
    # ROUGE-L F-measure
    prec = lcs_len / m
    rec = lcs_len / n
    return (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

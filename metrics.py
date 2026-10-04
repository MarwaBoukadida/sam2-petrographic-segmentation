# -*- coding: utf-8 -*-
"""Mask-level metrics and summary statistics."""

import numpy as np
from collections import Counter
from typing import Tuple, Dict, Optional, List

from sklearn.metrics import rand_score, adjusted_rand_score
from config import METRIC_WEIGHTS

class MetricsCalculator:
    """Compute mask metrics. IoU, Dice, purity, GCE, precision, recall and F1 are percentages; weighted_score uses a 0-1 scale."""

    @staticmethod
    def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Return intersection over union as a fraction."""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        return float(intersection / union) if union > 0 else 0.0

    @staticmethod
    def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Return the Dice coefficient as a fraction."""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        total = pred_mask.sum() + gt_mask.sum()
        return float(2 * intersection / total) if total > 0 else 0.0

    @staticmethod
    def compute_purity(phase_image: np.ndarray, pred_mask: np.ndarray) -> Tuple[float, Tuple, int, int]:
        """Return purity in percent, dominant RGB colour, dominant count and mask area. Ties follow colour encounter order."""
        masked_pixels = phase_image[pred_mask == 1]
        if masked_pixels.size == 0:
            return 0.0, (0, 0, 0), 0, 0

        # Ties follow RGB colour encounter order.
        masked_colors: List[Tuple[int, int, int]] = [tuple(px) for px in masked_pixels]
        color_counts = Counter(masked_colors)
        dominant_color, dominant_count = color_counts.most_common(1)[0]
        total = int(pred_mask.sum())

        purity = (dominant_count / total * 100.0) if total > 0 else 0.0
        return float(purity), dominant_color, int(dominant_count), total

    @staticmethod
    def compute_gce(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Return foreground-overlap consistency on a 0-1 scale; smaller values indicate less discrepancy."""
        s = pred_mask.astype(bool)
        g = gt_mask.astype(bool)

        def local_error(a, b):
            inter = np.logical_and(a, b).sum()
            return 1.0 - (inter / a.sum()) if a.sum() > 0 else 1.0

        return float(min(local_error(s, g), local_error(g, s)))

    @staticmethod
    def compute_precision_recall(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
        """Return precision and recall as fractions."""
        tp = np.logical_and(pred_mask, gt_mask).sum()
        pp = pred_mask.sum()
        gp = gt_mask.sum()

        precision = (tp / pp) if pp > 0 else 0.0
        recall = (tp / gp) if gp > 0 else 0.0
        return float(precision), float(recall)

    @staticmethod
    def compute_f1_score(precision: float, recall: float) -> float:
        """Return F1 on a 0-1 scale."""
        return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    @staticmethod
    def compute_pr_npr(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
        """Return Rand and adjusted Rand indices for flattened binary masks."""
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        return float(rand_score(gt_flat, pred_flat)), float(adjusted_rand_score(gt_flat, pred_flat))

    @staticmethod
    def compute_weighted_score(metrics: Dict[str, float],
                               weights: Optional[Dict[str, float]] = None) -> float:
        """Divide percentage scores by 100, clip to [0, 1] and combine them. Skip invalid scores and renormalize available weights."""

        if weights is None:
            weights = METRIC_WEIGHTS.copy()

        score = 0.0
        total_w = 0.0

        for m, w in weights.items():
            val = metrics.get(m)
            if val is None:
                continue
            try:
                v = float(val) / 100.0
                v = max(0.0, min(v, 1.0))
                score += w * v
                total_w += w
            except (TypeError, ValueError):
                continue

        return score / total_w if total_w > 0 else 0.0

    @classmethod
    def compute_all_metrics(cls,
                            pred_mask: np.ndarray,
                            gt_mask: np.ndarray,
                            phase_image: np.ndarray) -> Dict[str, float]:
        """Return mask metrics and weighted score; return zero scores if calculation fails."""
        out: Dict[str, float] = {}

        try:

            out['iou'] = cls.compute_iou(pred_mask, gt_mask) * 100.0
            out['dice'] = cls.compute_dice(pred_mask, gt_mask) * 100.0

            purity, dom_color, dom_cnt, total = cls.compute_purity(phase_image, pred_mask)
            out['purity'] = purity

            out['gce'] = cls.compute_gce(pred_mask, gt_mask) * 100.0

            precision, recall = cls.compute_precision_recall(pred_mask, gt_mask)
            out['precision'] = precision * 100.0
            out['recall'] = recall * 100.0
            out['f1_score'] = cls.compute_f1_score(precision, recall) * 100.0

            out['weighted_score'] = cls.compute_weighted_score(out)

        except Exception as e:

            out.update({
                'iou': 0.0, 'dice': 0.0, 'purity': 0.0, 'gce': 0.0,
                'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'weighted_score': 0.0
            })
            print(f"[MetricsCalculator] Erreur compute_all_metrics: {e}")

        return out

    @staticmethod
    def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Summarize mask rows with equal weight: mean, population standard deviation (ddof=0), median, minimum, maximum and count."""
        if not metrics_list:
            return {}

        aggregated: Dict[str, float] = {}
        keys = ['iou', 'dice', 'purity', 'gce', 'precision', 'recall', 'f1_score', 'weighted_score']

        for k in keys:
            vals = [m.get(k) for m in metrics_list if m.get(k) is not None]
            if len(vals) == 0:
                continue
            arr = np.asarray(vals, dtype=float)
            aggregated[f'{k}_mean'] = float(np.mean(arr))
            aggregated[f'{k}_std'] = float(np.std(arr))
            aggregated[f'{k}_median'] = float(np.median(arr))
            aggregated[f'{k}_min'] = float(np.min(arr))
            aggregated[f'{k}_max'] = float(np.max(arr))
            aggregated[f'{k}_count'] = int(arr.size)

        return aggregated

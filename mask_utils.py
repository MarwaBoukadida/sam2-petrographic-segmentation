# -*- coding: utf-8 -*-
"""Mask generation, visualization and result files."""

import os
import gc
import json
import logging
from typing import List, Dict, Optional, Any, Iterable, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    RESULTS_DIR, TRIALS_DETAILS_DIR,
    VISUALIZATION_DPI,
    SAVE_TOP_BOTTOM_MASKS, N_BEST_MASKS_PER_PATCH, N_WORST_MASKS_PER_PATCH,
    SAVE_INDIVIDUAL_VISUALIZATIONS, SAVE_2X2_VISUALIZATIONS,
    MAX_VISUALIZATIONS_PER_TRIAL
)

logger = logging.getLogger(__name__)

def safe_mask_generation(mask_generator, image: np.ndarray, max_retries: int = 3,
                         device: Optional[torch.device] = None) -> List:
    """Generate masks, retrying CUDA out-of-memory errors after cleanup. Return an empty list if generation fails."""
    for attempt in range(max_retries):
        try:
            return mask_generator.generate(image)
        except RuntimeError as e:
            msg = str(e)
            if "CUDA out of memory" in msg and device is not None and device.type == "cuda":
                logger.warning(f"[safe_mask_generation] CUDA OOM ({attempt+1}/{max_retries}) — empty_cache()")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            logger.error(f"[safe_mask_generation] RuntimeError: {e}")
            break
        except Exception as e:
            logger.error(f"[safe_mask_generation] Erreur inattendue: {e}")
            break
    return []

def _viz_budget_path(trial_dir: str) -> str:
    """Return the visualization-count file path."""
    return os.path.join(trial_dir, "_viz_budget.json")

def _load_viz_count(trial_dir: str) -> int:
    """Read the visualization count; return zero on failure."""
    try:
        with open(_viz_budget_path(trial_dir), "r", encoding="utf-8") as f:
            d = json.load(f)
        return int(d.get("count", 0))
    except Exception:
        return 0

def _save_viz_count(trial_dir: str, count: int) -> None:
    """Save the visualization count."""
    try:
        with open(_viz_budget_path(trial_dir), "w", encoding="utf-8") as f:
            json.dump({"count": int(count)}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.debug(f"[viz_budget] write fail: {e}")

def _can_consume_viz_budget(trial_dir: str, n: int = 1) -> bool:
    """Check and update the optional visualization limit."""
    if MAX_VISUALIZATIONS_PER_TRIAL is None or MAX_VISUALIZATIONS_PER_TRIAL <= 0:
        return True
    cnt = _load_viz_count(trial_dir)
    if cnt + n <= MAX_VISUALIZATIONS_PER_TRIAL:
        _save_viz_count(trial_dir, cnt + n)
        return True
    return False

def save_individual_metrics(metrics: Dict, vis_dir: str, mask_id: int):
    """Save mask metrics as text and JSON."""
    os.makedirs(vis_dir, exist_ok=True)
    try:
        with open(os.path.join(vis_dir, "metrics.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Métriques masque {mask_id}\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Score Pondéré: {metrics.get('weighted_score', 0):.4f}\n")
            f.write(f"IoU: {metrics.get('iou', 0):.2f}%\n")
            f.write(f"Dice: {metrics.get('dice', 0):.2f}%\n")
            f.write(f"Pureté: {metrics.get('purity', 0):.2f}%\n")
            f.write(f"Précision: {metrics.get('precision', 0):.2f}%\n")
            f.write(f"Rappel: {metrics.get('recall', 0):.2f}%\n")
            f.write(f"F1: {metrics.get('f1_score', 0):.2f}%\n")
    except Exception as e:
        logger.debug(f"metrics.txt failed: {e}")

    try:
        json_metrics = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                        for k, v in metrics.items()}
        json_metrics['mask_id'] = int(mask_id)
        json_metrics['saved_at'] = __import__('datetime').datetime.now().isoformat()
        with open(os.path.join(vis_dir, "metrics.json"), 'w', encoding='utf-8') as f:
            json.dump(json_metrics, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.debug(f"metrics.json failed: {e}")

def save_individual_visualizations(pred_mask: np.ndarray, gt_mask: np.ndarray,
                                   patch_image: np.ndarray, patch_gt: np.ndarray,
                                   mask_dir: str):
    """Save RGB, reference, prediction and overlap images when enabled."""
    if not SAVE_INDIVIDUAL_VISUALIZATIONS:
        return
    trial_dir = os.path.dirname(os.path.dirname(mask_dir))
    if not _can_consume_viz_budget(trial_dir, n=4):
        return

    os.makedirs(mask_dir, exist_ok=True)

    try:
        plt.figure(figsize=(6, 6))
        plt.imshow(patch_image)
        plt.title('Image (optical)')
        plt.axis('off')
        plt.savefig(os.path.join(mask_dir, "original.png"), dpi=VISUALIZATION_DPI, bbox_inches='tight')
    finally:
        plt.close()

    try:
        plt.figure(figsize=(6, 6))
        plt.imshow(patch_gt)
        plt.title('Ground Truth (phases)')
        plt.axis('off')
        plt.savefig(os.path.join(mask_dir, "gt.png"), dpi=VISUALIZATION_DPI, bbox_inches='tight')
    finally:
        plt.close()

    try:
        plt.figure(figsize=(6, 6))
        plt.imshow(pred_mask, cmap='gray')
        plt.title('Masque prédit')
        plt.axis('off')
        plt.savefig(os.path.join(mask_dir, "pred.png"), dpi=VISUALIZATION_DPI, bbox_inches='tight')
    finally:
        plt.close()

    try:
        overlay = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        overlay[(gt_mask == 1) & (pred_mask == 1)] = [0, 255, 0]
        overlay[(gt_mask == 1) & (pred_mask == 0)] = [255, 0, 0]
        overlay[(gt_mask == 0) & (pred_mask == 1)] = [0, 0, 255]
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.title('TP/FP/FN (fond neutre)')
        plt.axis('off')
        plt.savefig(os.path.join(mask_dir, "overlay.png"), dpi=VISUALIZATION_DPI, bbox_inches='tight')
    finally:
        plt.close()

def save_mask_visualization_2x3(pred_mask: np.ndarray, gt_mask: np.ndarray,
                                patch_image: np.ndarray, patch_gt: np.ndarray,
                                patch_dir: str, mask_id: int,
                                metrics: Dict, patch_id: str):
    """Save a six-panel visualization when SAVE_2X2_VISUALIZATIONS is enabled."""
    if not SAVE_2X2_VISUALIZATIONS:
        return
    trial_dir = os.path.dirname(patch_dir)
    if not _can_consume_viz_budget(trial_dir, n=1):
        return

    mask_dir = os.path.join(patch_dir, f"mask_{mask_id}")
    os.makedirs(mask_dir, exist_ok=True)

    tpfnfp = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    tpfnfp[(gt_mask == 1) & (pred_mask == 1)] = [0, 255, 0]
    tpfnfp[(gt_mask == 1) & (pred_mask == 0)] = [255, 0, 0]
    tpfnfp[(gt_mask == 0) & (pred_mask == 1)] = [0, 0, 255]

    w = float(metrics.get('weighted_score', 0))
    title = (f'Patch {patch_id} - Masque {mask_id} | '
             f'IoU {metrics.get("iou", 0):.1f}%  Dice {metrics.get("dice", 0):.1f}%  '
             f'Purity {metrics.get("purity", 0):.1f}%  Score {w:.3f}')

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    fig.suptitle(title, fontsize=11, fontweight='bold')

    axes[0, 0].imshow(patch_image); axes[0, 0].set_title('Image'); axes[0, 0].axis('off')

    axes[0, 1].imshow(patch_gt); axes[0, 1].set_title('GT (phases)'); axes[0, 1].axis('off')

    axes[0, 2].imshow(pred_mask, cmap='gray'); axes[0, 2].set_title('Pred (binaire)'); axes[0, 2].axis('off')

    axes[1, 0].imshow(tpfnfp)
    axes[1, 0].set_title('TP/FP/FN');
    axes[1, 0].axis('off')

    axes[1, 1].imshow(patch_image)
    axes[1, 1].imshow(pred_mask, cmap='jet', alpha=0.35)
    axes[1, 1].set_title('Pred sur Image');
    axes[1, 1].axis('off')

    axes[1, 2].imshow(patch_image)
    axes[1, 2].imshow(tpfnfp, alpha=0.35)
    axes[1, 2].set_title('TP/FP/FN sur Image');
    axes[1, 2].axis('off')

    plt.tight_layout()
    out = os.path.join(mask_dir, f"mask_{mask_id:03d}_visualization_2x3.png")
    try:
        plt.savefig(out, dpi=VISUALIZATION_DPI, bbox_inches='tight', facecolor='white')
    finally:
        plt.close()

def save_mask_visualization_original_style(pred_mask: np.ndarray, gt_mask: np.ndarray,
                                           patch_image: np.ndarray, patch_gt: np.ndarray,
                                           patch_dir: str, mask_id: int,
                                           metrics: Dict, patch_id: str):
    """Call the six-panel visualization helper."""
    return save_mask_visualization_2x3(
        pred_mask=pred_mask,
        gt_mask=gt_mask,
        patch_image=patch_image,
        patch_gt=patch_gt,
        patch_dir=patch_dir,
        mask_id=mask_id,
        metrics=metrics,
        patch_id=patch_id
    )

def select_top_bottom_indices(patch_metrics: List[Dict]) -> Tuple[List[int], List[int]]:
    """Select mask positions by weighted score. High- and low-score selections can overlap."""
    if not patch_metrics:
        return [], []
    sorted_idx = sorted(range(len(patch_metrics)),
                        key=lambda k: patch_metrics[k].get('weighted_score', 0.0))
    worst_k = max(0, int(N_WORST_MASKS_PER_PATCH))
    best_k = max(0, int(N_BEST_MASKS_PER_PATCH))
    worst_idx = sorted_idx[:worst_k]
    best_idx = sorted_idx[-best_k:] if best_k > 0 else []
    return best_idx, worst_idx

def save_patch_top_bottom_summary(patch_metrics: List[Dict], patch_id: str, trial_dir: str) -> None:
    """Save the selected mask summary when enabled."""
    if not patch_metrics or (not SAVE_TOP_BOTTOM_MASKS and not SAVE_INDIVIDUAL_VISUALIZATIONS and not SAVE_2X2_VISUALIZATIONS):
        return
    try:
        best_idx, worst_idx = select_top_bottom_indices(patch_metrics)
        payload = {
            "patch_id": patch_id,
            "total_masks": len(patch_metrics),
            "best_indices": best_idx,
            "worst_indices": worst_idx,
            "scores": [float(m.get("weighted_score", 0.0)) for m in patch_metrics]
        }
        patch_dir = os.path.join(trial_dir, f"patch_{patch_id}")
        os.makedirs(patch_dir, exist_ok=True)
        with open(os.path.join(patch_dir, "top_bottom_masks.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.debug(f"save_patch_top_bottom_summary failed: {e}")

def save_trial_details(trial_number: int, summary: Dict[str, Any]) -> None:
    """Save a trial summary if the output file does not exist."""
    trial_dir = os.path.join(RESULTS_DIR, "trials_details", f"trial_{trial_number}")
    os.makedirs(trial_dir, exist_ok=True)
    path = os.path.join(trial_dir, "trial_summary.json")
    try:
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.debug(f"[save_trial_details] Échec écriture: {e}")

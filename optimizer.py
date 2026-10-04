# -*- coding: utf-8 -*-
"""SAM2 mask generation and scoring for petrographic patches."""
import gc
import os
import time
import json
import logging
from collections import Counter
from typing import List, Dict, Optional, Tuple

import csv

import numpy as np
import optuna
import torch
from scipy.ndimage import label
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from patchs_loader import PatchData, PatchDataset
from metrics import MetricsCalculator
from mask_utils import (
    safe_mask_generation,
    save_individual_visualizations,
    save_mask_visualization_original_style,
    save_individual_metrics,
    save_patch_top_bottom_summary,
    select_top_bottom_indices
)

logger = logging.getLogger(__name__)

# Fréquence de nettoyage mémoire (patchs)
try:
    from config import MEMORY_CLEANUP_FREQUENCY
except Exception:
    MEMORY_CLEANUP_FREQUENCY = 50


class OptimizedSAM2Optimizer:
    """Generate and score SAM2 masks for petrographic patches."""

    # Patch summary CSV.
    def _patch_summary_path(self, trial_dir: str) -> str:
        return os.path.join(trial_dir, "patch_summaries.csv")

    def _append_patch_summary(self, trial_dir: str, row: dict):
        path = self._patch_summary_path(trial_dir)
        header = ["trial_tag", "image_id", "patch_id", "y", "x",
                  "n_masks_total", "n_masks_kept", "n_selected_for_png", "processing_time_s"]
        write_header = not os.path.exists(path)
        os.makedirs(trial_dir, exist_ok=True)
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=header)
                if write_header: w.writeheader()
                w.writerow({k: row.get(k) for k in header})
        except Exception as e:
            logger.debug(f"[patch_summaries] write fail: {e}")


    def __init__(self, sam2_checkpoint: str, model_cfg: str, dataset: PatchDataset, device):
        logger.info("Initialisation OptimizedSAM2Optimizer")
        self.device = device
        self.dataset = dataset
        self.calculator = MetricsCalculator()

        try:
            logger.info(f"Chargement modèle SAM2: {sam2_checkpoint}")
            self.sam2 = build_sam2(model_cfg, sam2_checkpoint, device=self.device, apply_postprocessing=False)
            logger.info("Modèle SAM2 chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur chargement SAM2: {e}")
            raise

        self.session_stats = {
            'total_patches_processed': 0,
            'total_masks_generated': 0,   # nb de lignes métriques (tous masques)
            'total_processing_time': 0.0,
            'memory_cleanups': 0,
            'successful_batches': 0,
            'failed_batches': 0
        }

    
    # Espace de recherche (Optuna)
    
    def get_optimized_search_space(self, trial: optuna.Trial) -> Dict:
        points_per_side = trial.suggest_int("points_per_side", 16, 48, step=8)
        pred_iou_thresh = trial.suggest_float("pred_iou_thresh", 0.30, 0.80, step=0.05)
        stability_score_thresh = trial.suggest_float("stability_score_thresh", 0.80, 0.95, step=0.02)
        mask_threshold = trial.suggest_float("mask_threshold", 0.00, 0.50, step=0.05)

        crop_n_layers = trial.suggest_int("crop_n_layers", 0, 2)
        crop_n_points_downscale_factor = 1
        box_nms_thresh = trial.suggest_float("box_nms_thresh", 0.30, 0.80, step=0.05)
        crop_overlap_ratio = trial.suggest_float("crop_overlap_ratio", 0.10, 0.70, step=0.05)

        min_area_float = trial.suggest_float("min_mask_region_area", 32, 4096, log=True)
        min_mask_region_area = int(max(0, round(min_area_float / 16)) * 16) or 16

        use_m2m = trial.suggest_categorical("use_m2m", [True, False])
        stability_score_offset = 0.95

        assert 0.0 <= mask_threshold <= 1.0
        assert 0.0 <= pred_iou_thresh <= 1.0
        assert 0.0 <= stability_score_thresh <= 1.0

        return {
            "points_per_side": points_per_side,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh,
            "mask_threshold": mask_threshold,
            "min_mask_region_area": min_mask_region_area,
            "crop_n_layers": crop_n_layers,
            "crop_n_points_downscale_factor": crop_n_points_downscale_factor,
            "box_nms_thresh": box_nms_thresh,
            "crop_overlap_ratio": crop_overlap_ratio,
            "use_m2m": use_m2m,
            "stability_score_offset": stability_score_offset,
        }

    
    # API publique
    
    def compute_weighted_score(self, metrics: Dict, weights: Optional[Dict] = None) -> float:
        return self.calculator.compute_weighted_score(metrics, weights)

    def evaluate_patches_batch(self, patch_indices: List[int], params: Optional[Dict],
                               trial_dir: str, trial_number) -> List[Dict]:
        logger.info(f"Début évaluation batch - {len(patch_indices)} patches")
        batch_start_time = time.time()

        try:
            if params is None:
                mask_generator = SAM2AutomaticMaskGenerator(
                    self.sam2,
                    output_mode="binary_mask"
                )
            else:
                mask_generator = SAM2AutomaticMaskGenerator(
                    self.sam2,
                    output_mode="binary_mask",
                    **params
                )
            logger.info("Générateur de masques créé")
        except Exception as e:
            logger.error(f"Erreur création générateur: {e}")
            self.session_stats["failed_batches"] += 1
            return []

        all_metrics: List[Dict] = []
        patches_processed = 0

        logger.info("Début traitement des patches")
        for i, patch_idx in enumerate(tqdm(patch_indices, desc=f"Trial {trial_number}")):
            t0_patch = time.time()
            try:
                patch = self.dataset.load_patch(patch_idx)
                if patch is None:
                    logger.warning(f"Échec chargement patch {patch_idx}")
                    continue

                # Generate masks.
                with torch.inference_mode():
                    masks = safe_mask_generation(mask_generator, patch.patch_image, device=self.device)
                if not masks:
                    logger.debug(f"Aucun masque généré pour patch {patch_idx}")
                    # Save a patch summary even if no masks were generated.
                    self._append_patch_summary(trial_dir, {
                        "trial_tag": str(trial_number),
                        "image_id": patch.image_id,
                        "patch_id": f"{patch.image_id}_{patch.position[0]}_{patch.position[1]}",
                        "y": int(patch.position[0]), "x": int(patch.position[1]),
                        "n_masks_total": 0,
                        "n_masks_kept": 0,
                        "n_selected_for_png": 0,
                        "processing_time_s": float(time.time() - t0_patch)
                    })
                    continue

                # Évalue le patch (peut renvoyer liste ou (liste, info))
                res = self._evaluate_single_patch(patch, masks, trial_dir)
                patch_info = {}
                if isinstance(res, tuple):
                    patch_metrics, patch_info = res
                else:
                    patch_metrics = res

                # Résumé par patch
                processing_time_s = time.time() - t0_patch
                self._append_patch_summary(trial_dir, {
                    "trial_tag": str(trial_number),
                    "image_id": patch.image_id,
                    "patch_id": f"{patch.image_id}_{patch.position[0]}_{patch.position[1]}",
                    "y": int(patch.position[0]), "x": int(patch.position[1]),
                    "n_masks_total": int(patch_info.get("n_masks_total", len(masks))),
                    "n_masks_kept": int(patch_info.get("n_masks_kept", len(patch_metrics or []))),
                    "n_selected_for_png": int(patch_info.get("n_selected_for_png", 0)),
                    "processing_time_s": float(processing_time_s)
                })

                if patch_metrics:
                    all_metrics.extend(patch_metrics)
                    patches_processed += 1

                if (i + 1) % max(1, int(MEMORY_CLEANUP_FREQUENCY)) == 0:
                    self._cleanup_memory()

            except Exception as patch_error:
                logger.error(f"Erreur patch {patch_idx}: {patch_error}")
                continue

        self._cleanup_memory()
        batch_time = time.time() - batch_start_time

        self.session_stats['total_patches_processed'] += patches_processed
        self.session_stats['total_masks_generated'] += len(all_metrics)
        self.session_stats['total_processing_time'] += batch_time
        if len(all_metrics) > 0:
            self.session_stats['successful_batches'] += 1
        else:
            self.session_stats['failed_batches'] += 1

        logger.info(f"Batch terminé: {patches_processed}/{len(patch_indices)} patches traités")
        logger.info(f"Masques évalués (lignes metrics): {len(all_metrics)}")
        logger.info(f"Temps total: {batch_time:.1f}s")
        if len(all_metrics) > 0:
            avg_weighted = float(np.mean([m.get('weighted_score', 0.0) for m in all_metrics]))
            logger.info(f"Score pondéré moyen: {avg_weighted:.4f}")

        return all_metrics

    
    # Évaluation d’un patch (tous masques) + I/O seulement Top/Bottom
    
    def _evaluate_single_patch(self, patch: PatchData, masks: List, trial_dir: str):
        """Score all masks, save configured visualizations for selected high- and low-score masks, and return metrics with patch counts."""
        y, x = patch.position
        patch_id = f"{patch.image_id}_{y}_{x}"
        patch_dir = os.path.join(trial_dir, f"patch_{patch_id}")
        os.makedirs(patch_dir, exist_ok=True)

        metrics_list: List[Dict] = []
        mask_buffers: List[Tuple[int, np.ndarray, np.ndarray, Dict]] = []  # (i, pred, gt, metrics)

        # Score all masks.
        n_masks_total = len(masks)  # Number of generated masks.
        for i, mask_data in enumerate(masks):
            try:
                if 'segmentation' not in mask_data:
                    continue
                pred_mask = mask_data['segmentation'].astype(np.uint8)
                if pred_mask.sum() == 0:
                    continue

                # couleur dominante sous prédiction → régions GT associées
                masked_pixels = patch.patch_gt[pred_mask == 1]
                if masked_pixels.size == 0:
                    continue
                masked_colors = [tuple(px) for px in masked_pixels]
                dominant_color, _ = Counter(masked_colors).most_common(1)[0]

                full_mineral_mask = np.all(patch.patch_gt == dominant_color, axis=-1).astype(np.uint8)
                labeled_regions, _ = label(full_mineral_mask)
                region_ids = np.unique(labeled_regions[pred_mask == 1])
                region_ids = region_ids[region_ids != 0]
                gt_mask = np.isin(labeled_regions, region_ids).astype(np.uint8)
                if gt_mask.sum() == 0:
                    continue

                m = self.calculator.compute_all_metrics(pred_mask, gt_mask, patch.patch_gt)
                row = {
                    **m,
                    "patch_id": patch_id,
                    "mask_id": int(i),
                    "y": int(y), "x": int(x),
                    "image_id": patch.image_id,
                    "mask_area": int(pred_mask.sum()),
                    "gt_area": int(gt_mask.sum())
                }
                metrics_list.append(row)
                mask_buffers.append((i, pred_mask, gt_mask, row))

            except Exception as e:
                logger.debug(f"Patch {patch_id} masque {i} — skip ({e})")
                continue

        if not metrics_list:
            # Return patch counts with the metrics.
            return [], {"n_masks_total": n_masks_total, "n_masks_kept": 0, "n_selected_for_png": 0}

        # Number of scored masks.
        n_masks_kept = len(metrics_list)

        # Select high- and low-score masks.
        best_idx, worst_idx = select_top_bottom_indices(metrics_list)

        # Save selected mask outputs.
        rgb_u8 = patch.patch_image if patch.patch_image.dtype == np.uint8 else np.clip(patch.patch_image, 0,
                                                                                       255).astype(np.uint8)
        n_selected_for_png = 0  # Number of saved visualizations.
        for i, pred_mask, gt_mask, row in mask_buffers:
            if (i in best_idx) or (i in worst_idx):
                n_selected_for_png += 1
                mask_dir = os.path.join(patch_dir, f"mask_{i}")
                # PNG séparés
                save_individual_visualizations(pred_mask, gt_mask, rgb_u8, patch.patch_gt, mask_dir)
                # Combined 2-by-3 panel.
                save_mask_visualization_original_style(pred_mask, gt_mask, rgb_u8, patch.patch_gt,
                                                       patch_dir, i, row, patch_id)
                # TXT + JSON
                save_individual_metrics(row, mask_dir, i)

        # Save the selection summary.
        save_patch_top_bottom_summary(metrics_list, patch_id, trial_dir)

        # Return counts for patch_summaries.csv.
        patch_info = {
            "n_masks_total": n_masks_total,
            "n_masks_kept": n_masks_kept,
            "n_selected_for_png": n_selected_for_png
        }
        return metrics_list, patch_info

    
    # Nettoyage mémoire
    
    def _cleanup_memory(self):
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    
    # Évaluation finale optionnelle
    
    def evaluate_best_params(self, params: Dict, n_patches: int = 100) -> Dict:
        logger.info(f"Évaluation finale (aléatoire) - {n_patches} patches")
        valid_indices = self.dataset.get_filtered_indices()
        if len(valid_indices) == 0:
            logger.error("Aucun patch valide pour l'évaluation finale.")
            return {}

        import numpy as _np
        test_indices = _np.random.choice(valid_indices, min(n_patches, len(valid_indices)), replace=False)

        final_dir = "final_evaluation"
        os.makedirs(final_dir, exist_ok=True)

        all_metrics = self.evaluate_patches_batch(test_indices.tolist(), params, final_dir, trial_number="FINAL")
        if not all_metrics:
            logger.error("Aucune métrique obtenue lors de l'évaluation finale")
            return {}

        stats = self.calculator.aggregate_metrics(all_metrics)
        final_results = {
            'evaluation_params': params,
            'n_patches_evaluated': len(test_indices),
            'n_masks_generated': len(all_metrics),
            'statistics': stats,
            'session_stats': self.session_stats,
            'timestamp': time.time()
        }
        with open(f"{final_dir}/final_complete_results.json", "w", encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        import pandas as pd
        pd.DataFrame(all_metrics).to_csv(f"{final_dir}/final_detailed_metrics.csv", index=False)
        logger.info("Évaluation finale (aléatoire) terminée")
        return stats

# -*- coding: utf-8 -*-
"""SAM2 parameter optimization with per-slide validation folds."""

import os
import json
import random
import logging
import time
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import optuna

from optimizer import OptimizedSAM2Optimizer
from mask_utils import save_trial_details
from config import SAMPLING_RATIO

logger = logging.getLogger(__name__)

def create_objective_with_cv(optimizer: OptimizedSAM2Optimizer,
                             cv_folds: List[Tuple[List[int], List[int]]]):
    """Create a LOSO objective with fixed model weights. Sample validation patches using SAMPLING_RATIO with a minimum of 20. Average mask scores within each fold, then average all fold scores, including zeros."""

    def objective(trial: optuna.Trial) -> float:
        logger.info(f"[Trial {trial.number}] Début")
        trial_start_time = time.time()

        # Includes the fixed settings and the effective mask-area conversion.
        params = optimizer.get_optimized_search_space(trial)
        logger.info(f"[Trial {trial.number}] Paramètres:")
        for k, v in params.items():
            logger.info(f"   - {k}: {v}")

        trial_dir = os.path.join("optuna_results", f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)

        # Store the effective settings before evaluating any validation fold.
        with open(os.path.join(trial_dir, "params.json"), "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)

        fold_scores: List[float] = []
        fold_details: List[Dict] = []
        all_fold_metrics: List[Dict] = []
        successful_folds = 0

        logger.info(f"[Trial {trial.number}] Évaluation sur {len(cv_folds)} folds (LOSO).")

        for fold_idx, (train_indices, val_indices) in enumerate(cv_folds):
            fold_t0 = time.time()
            logger.info(f"[Trial {trial.number}] Fold {fold_idx + 1}/{len(cv_folds)}")

            try:
                if not val_indices:
                    logger.warning(f"[Trial {trial.number}] Fold {fold_idx}: val_indices vide.")
                    fold_scores.append(0.0)
                    fold_details.append({
                        'fold_idx': fold_idx,
                        'status': 'no_val_data',
                        'fold_weighted_score': 0.0,
                        'processing_time_seconds': time.time() - fold_t0
                    })

                    # Report an empty fold and continue.
                    trial.report(0.0, step=fold_idx)
                    continue

                val_img_id = optimizer.dataset.metadata[val_indices[0]]["image_id"]
                n_val_total = len(val_indices)

                # If the requested sample is not smaller, use all fold patches.
                val_sample_size = max(20, int(n_val_total * SAMPLING_RATIO))
                if val_sample_size < n_val_total:
                    val_sample = random.sample(val_indices, val_sample_size)
                else:
                    val_sample = list(val_indices)

                logger.info(f"[Trial {trial.number}] Fold {fold_idx}: lame tenue-out='{val_img_id}', "
                            f"val_patches={n_val_total}, échantillon={len(val_sample)}")

                fold_dir = os.path.join(trial_dir, f"fold_{fold_idx}")
                os.makedirs(fold_dir, exist_ok=True)

                fold_metrics = optimizer.evaluate_patches_batch(
                    patch_indices=val_sample,
                    params=params,
                    trial_dir=fold_dir,
                    trial_number=f"{trial.number}_fold{fold_idx}"
                )

                if fold_metrics:

                    fold_weighted_scores = [m.get('weighted_score', 0.0) for m in fold_metrics]
                    fold_score = float(np.mean(fold_weighted_scores))
                    successful_folds += 1

                    df_fold = pd.DataFrame(fold_metrics)
                    df_fold.to_csv(os.path.join(fold_dir, "fold_metrics.csv"), index=False)
                    all_fold_metrics.extend(fold_metrics)

                else:
                    fold_score = 0.0

                fold_time = time.time() - fold_t0
                fold_scores.append(fold_score)

                fold_stats = {
                    'fold_idx': fold_idx,
                    'val_image_id': val_img_id,
                    'n_val_patches_total': n_val_total,
                    'n_val_patches_sampled': len(val_sample),
                    'n_masks_evaluated': len(fold_metrics) if fold_metrics else 0,
                    'fold_weighted_score': fold_score,
                    'processing_time_seconds': fold_time,
                    'status': 'success' if fold_metrics else 'no_metrics'
                }
                fold_details.append(fold_stats)

                logger.info(f"[Trial {trial.number}] Fold {fold_idx} terminé — score={fold_score:.4f}")

                # Report the fold score for pruning.
                trial.report(fold_score, step=fold_idx)
                if trial.should_prune():
                    logger.info(f"[Trial {trial.number}] PRUNE au fold {fold_idx} (score provisoire={fold_score:.4f})")
                    raise optuna.TrialPruned(f"Pruned at fold {fold_idx}")

            except optuna.TrialPruned as pr:

                fold_time = time.time() - fold_t0
                fold_details.append({
                    'fold_idx': fold_idx,
                    'val_image_id': optimizer.dataset.metadata[val_indices[0]]["image_id"] if val_indices else "unknown",
                    'fold_weighted_score': fold_scores[-1] if fold_scores else 0.0,
                    'processing_time_seconds': fold_time,
                    'status': 'pruned',
                    'message': str(pr)
                })

                _save_trial_summary_safely(trial_dir, trial.number, params, fold_scores, fold_details, all_fold_metrics,
                                           trial_start_time)
                raise

            except Exception as fold_err:
                fold_time = time.time() - fold_t0
                logger.error(f"[Trial {trial.number}] Erreur Fold {fold_idx}: {fold_err}")

                fold_scores.append(0.0)
                fold_details.append({
                    'fold_idx': fold_idx,
                    'val_image_id': optimizer.dataset.metadata[val_indices[0]]["image_id"] if val_indices else "unknown",
                    'fold_weighted_score': 0.0,
                    'processing_time_seconds': fold_time,
                    'status': 'failed',
                    'error': str(fold_err)
                })

                trial.report(0.0, step=fold_idx)

        valid_scores = [s for s in fold_scores if s is not None]
        final_score = float(np.mean(valid_scores)) if valid_scores else 0.0

        total_time = time.time() - trial_start_time
        logger.info(f"[Trial {trial.number}] Scores folds: {[f'{s:.4f}' for s in valid_scores]}")
        logger.info(f"[Trial {trial.number}] Folds réussis: {successful_folds}/{len(cv_folds)}")
        logger.info(f"[Trial {trial.number}] Score FINAL: {final_score:.4f}")

        _save_trial_summary_safely(trial_dir, trial.number, params, fold_scores, fold_details,
                                   all_fold_metrics, trial_start_time)

        return final_score

    return objective

def _save_trial_summary_safely(trial_dir: str,
                               trial_number: int,
                               params: Dict,
                               fold_scores: List[float],
                               fold_details: List[Dict],
                               all_fold_metrics: List[Dict],
                               trial_start_time: float) -> None:
    """Save trial summaries and mask-level metrics as CSV and JSON."""
    try:
        total_time = time.time() - trial_start_time

        summary = {
            "trial_number": trial_number,
            "params": params,
            "final_score": float(np.mean([s for s in fold_scores if s is not None])) if fold_scores else 0.0,
            "fold_scores": fold_scores,
            "fold_details": fold_details,
            "total_patches_evaluated": len(all_fold_metrics),
            "total_processing_time_seconds": total_time,
            "successful_folds": int(sum(1 for d in fold_details if d.get('status') == 'success')),
            "n_folds_total": len(fold_scores),
            "status": "completed" if any(fold_scores) else "no_metrics",
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }

        with open(os.path.join(trial_dir, "trial_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        if all_fold_metrics:
            df_all = pd.DataFrame(all_fold_metrics)
            df_all.to_csv(os.path.join(trial_dir, "all_metrics.csv"), index=False)

            from metrics import MetricsCalculator
            global_stats = MetricsCalculator.aggregate_metrics(all_fold_metrics)
            with open(os.path.join(trial_dir, "global_stats.json"), "w", encoding="utf-8") as f:
                json.dump(global_stats, f, indent=2, ensure_ascii=False)

        save_trial_details(trial_number, summary)

        try:
            import analysis
            analysis.generate_trial_figures(trial_dir)
        except Exception as e:
            logger.debug(f"[Trial {trial_number}] analyse locale échouée: {e}")

    except Exception as e:
        logger.warning(f"[Trial {trial_number}] Échec sauvegarde résumé: {e}")

# -*- coding: utf-8 -*-
"""Run SAM2 parameter optimization and export the best trial."""

import gc
import logging
import optuna
import json
import os
import time
from datetime import datetime
from config import RUN_POSTHOC_ANALYSIS, ANALYSIS_DIR

from patchs_loader import PatchDataset
from optimizer import OptimizedSAM2Optimizer
from objective import create_objective_with_cv
from config import (
    PATCH_DIR, CHECKPOINT_PATH, MODEL_CFG_PATH, N_TRIALS,
    ENABLE_FULL_LOSO, LOG_LEVEL, SAVE_LOGS_TO_FILE, LOG_FILE,
    OPTUNA_STUDY_NAME, OPTUNA_DB_URL, OPTUNA_PRUNER_TYPE,
    OPTUNA_PRUNER_STARTUP_TRIALS, OPTUNA_PRUNER_WARMUP_STEPS,
    RESULTS_DIR,
    EXCLUDE_BORDERS, BORDER_MARGIN, BORDER_KEEP_RATIO
)
from utils import get_device, set_seed



def setup_logging():
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]
    if SAVE_LOGS_TO_FILE:
        file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
        handlers.append(file_handler)
    logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=log_format, handlers=handlers)
    return logging.getLogger(__name__)


def create_optuna_study():
    if OPTUNA_PRUNER_TYPE.lower() == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=OPTUNA_PRUNER_STARTUP_TRIALS,
            n_warmup_steps=OPTUNA_PRUNER_WARMUP_STEPS
        )
    elif OPTUNA_PRUNER_TYPE.lower() == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        direction="maximize",
        study_name=OPTUNA_STUDY_NAME,
        storage=OPTUNA_DB_URL,
        load_if_exists=True,
        pruner=pruner
    )
    return study


def _selected_effective_parameters(optimizer, selected_trial):
    """Build the selected generator parameters without running an objective or creating a trial."""
    fixed_trial = optuna.trial.FixedTrial(
        dict(selected_trial.params), number=selected_trial.number
    )
    return optimizer.get_optimized_search_space(fixed_trial)


def generate_final_report(study, optimizer, execution_time_sec, effective_params=None):
    if effective_params is None:
        effective_params = _selected_effective_parameters(optimizer, study.best_trial)
    report_dir = "final_report"
    os.makedirs(report_dir, exist_ok=True)

    report_data = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time_hours": execution_time_sec / 3600,
            "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_trials_requested": N_TRIALS,
            "dataset_patches": len(optimizer.dataset.metadata),
            "validation_method": "LOSO_complete" if ENABLE_FULL_LOSO else "single_fold",
            "results_dir": RESULTS_DIR,
            "exclude_borders": EXCLUDE_BORDERS,
            "border_margin": BORDER_MARGIN
        },
        "best_configuration": {
            "weighted_score": study.best_value,
            "parameters": study.best_params,
            "effective_parameters": effective_params,
            "trial_number": study.best_trial.number
        },
        "session_statistics": optimizer.session_stats
    }

    with open(os.path.join(report_dir, "final_report.json"), "w", encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(report_dir, "experiment_summary.txt"), "w", encoding='utf-8') as f:
        f.write("RAPPORT OPTIMISATION SAM2 - LAMES MINCES GEOLOGIQUES\n")
        f.write("=" * 60 + "\n\n")
        f.write("CONFIGURATION:\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Durée totale: {execution_time_sec / 3600:.2f} heures\n")
        f.write(f"Patches dataset: {len(optimizer.dataset.metadata)}\n")
        f.write(f"Validation: {'LOSO complet' if ENABLE_FULL_LOSO else 'Fold unique'}\n")
        f.write(f"Filtrage bordures: {EXCLUDE_BORDERS} (margin={BORDER_MARGIN})\n\n")
        f.write("RESULTATS:\n")
        f.write(f"Meilleur score pondéré: {study.best_value:.4f}\n")
        f.write(
            f"Trials complétés: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}/{N_TRIALS}\n\n")
        f.write("PARAMÈTRES OPTIMISÉS:\n")
        for param, value in study.best_params.items():
            f.write(f" - {param}: {value}\n")
        f.write("\nPARAMÈTRES EFFECTIFS DU GÉNÉRATEUR:\n")
        for param, value in effective_params.items():
            f.write(f" - {param}: {value}\n")


def main():
    start_time = time.time()

    device = get_device()
    set_seed()
    log = setup_logging()

    log.info("Démarrage optimisation SAM2 pour lames minces géologiques")
    log.info(f"Config: {N_TRIALS} trials | Device: {device} | Results: {RESULTS_DIR}")

    try:
        # Dataset avec EXCLUSION DES BORDURES dès le chargement
        log.info("Initialisation du dataset…")
        dataset = PatchDataset(
            PATCH_DIR,
            exclude_borders=EXCLUDE_BORDERS,
            border_margin=BORDER_MARGIN,
            border_keep_ratio=BORDER_KEEP_RATIO
        )
        log.info(f"Dataset chargé: {len(dataset.metadata)} patches "
                 f"({len(dataset.get_filtered_indices())} non-bordures)")

        # Folds LOSO déjà filtrés
        if ENABLE_FULL_LOSO:
            log.info("Création des folds LOSO (indices filtrés)…")
            cv_folds = dataset.loso_split(use_filtered=True)
        else:
            log.info("Validation simple (1 fold LOSO pris au hasard, filtré)")
            all_folds = dataset.loso_split(use_filtered=True)
            cv_folds = [all_folds[0]]

        # Optimiseur SAM2
        log.info("Initialisation de l'optimiseur SAM2…")
        optimizer = OptimizedSAM2Optimizer(
            sam2_checkpoint=CHECKPOINT_PATH,
            model_cfg=MODEL_CFG_PATH,
            dataset=dataset,
            device=device
        )

        # Étude Optuna
        log.info("Configuration d’Optuna…")
        study = create_optuna_study()

        # Fonction objectif
        objective_func = create_objective_with_cv(optimizer, cv_folds)

        # Optimisation
        log.info(f"Lancement optimisation ({N_TRIALS} trials)…")
        optimization_start = time.time()
        study.optimize(objective_func, n_trials=N_TRIALS, catch=(Exception,))
        optimization_time = time.time() - optimization_start
        log.info(f"Optimisation terminée en {optimization_time / 3600:.2f} heures")

        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            log.error("Aucun trial complété avec succès.")
            return

        # Résultats meilleurs paramètres
        best = study.best_trial
        effective_params = _selected_effective_parameters(optimizer, best)
        log.info("Meilleure configuration :")
        log.info(f"  Score pondéré : {best.value:.4f}")
        for k, v in best.params.items():
            log.info(f"    {k}: {v}")


        # Sauvegarde meilleurs paramètres et résumé Optuna
        best_params_complete = {
            "optimized_parameters": best.params,
            "effective_parameters": effective_params,
            "selected_trial_number": best.number,
            "final_weighted_score": best.value,
            "optimization_info": {
                "total_trials": len(study.trials),
                "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                "optimization_time_hours": optimization_time / 3600,
                "validation_method": "LOSO_complete" if ENABLE_FULL_LOSO else "single_fold",
                "exclude_borders": EXCLUDE_BORDERS,
                "border_margin": BORDER_MARGIN
            }
        }
        with open("best_sam2_parameters.json", "w", encoding='utf-8') as f:
            json.dump(best_params_complete, f, indent=2, ensure_ascii=False)
        log.info("Meilleurs paramètres sauvegardés: best_sam2_parameters.json")

        try:
            df_trials = study.trials_dataframe()
            df_trials.to_csv("optuna_complete_summary.csv", index=False)
            if RUN_POSTHOC_ANALYSIS:
                try:
                    import analysis
                    analysis.run_posthoc_analysis(RESULTS_DIR, ANALYSIS_DIR)
                except Exception as e:
                    log.warning(f"Post-hoc analysis failed: {e}")

            log.info("Résumé des trials sauvegardé: optuna_complete_summary.csv")
        except Exception as e:
            log.warning(f"Impossible de sauvegarder le résumé des trials: {e}")

        # Visualisations Optuna (si plotly dispo)
        try:
            import optuna.visualization as vis
            log.info("Génération des visualisations Optuna (HTML)…")
            vis.plot_optimization_history(study).write_html("optimization_history.html")
            vis.plot_param_importances(study).write_html("param_importances.html")
            if len(study.trials) >= 10:
                vis.plot_parallel_coordinate(study).write_html("parallel_coordinate.html")
            log.info("Visualisations générées.")
        except ImportError:
            log.warning("optuna.visualization nécessite plotly — visualisations sautées.")
        except Exception as e:
            log.warning(f"Erreur génération visualisations: {e}")

        # Rapport final
        total_time = time.time() - start_time
        generate_final_report(study, optimizer, total_time, effective_params=effective_params)

        # Résumé console
        log.info("\n" + "=" * 60)
        log.info("OPTIMISATION SAM2 — TERMINÉE")
        log.info("=" * 60)
        log.info(f"Durée totale: {total_time / 3600:.2f} heures")
        log.info(f"Meilleur score pondéré: {best.value:.4f}")
        log.info(f"Trials complétés: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}/{N_TRIALS}")
        log.info(f"Patches traités: {optimizer.session_stats['total_patches_processed']}")
        log.info(f"Masques générés: {optimizer.session_stats['total_masks_generated']}")

    except Exception as e:
        log.error(f"ERREUR CRITIQUE: {e}")
        raise

    finally:
        if device.type == "cuda":
            gc.collect()
            import torch
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

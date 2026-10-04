#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import sys
from datetime import datetime
from typing import List, Set

from patchs_loader import PatchDataset
from optimizer import OptimizedSAM2Optimizer
from utils import get_device, set_seed
from config import (
    PATCH_DIR as _TRAIN_VAL_DIR,   
    CHECKPOINT_PATH, MODEL_CFG_PATH
)


def _discover_image_ids(patch_dir: str) -> List[str]:
    ids: Set[str] = set()
    for root, _, files in os.walk(patch_dir):
        for f in files:
            if f.endswith(".npz"):
                try:
                    import numpy as np
                    with np.load(os.path.join(root, f), allow_pickle=True) as d:
                        if "image_id" in d:
                            ids.add(str(d["image_id"]).strip())
                except Exception:
                    continue
    return sorted(list(ids))


def _complete_search_parameters(params: dict, sampled: bool) -> dict:
    """Construct effective generator parameters, including area conversion and fixed settings. Trial params.json files already store effective values."""
    if not isinstance(params, dict):
        raise ValueError("Les paramètres SAM2 doivent être un objet JSON.")
    params = params.copy()
    required = {
        "points_per_side", "pred_iou_thresh", "stability_score_thresh",
        "mask_threshold", "min_mask_region_area", "crop_n_layers",
        "box_nms_thresh", "crop_overlap_ratio", "use_m2m",
    }
    missing = required.difference(params)
    if missing:
        raise ValueError(f"Configuration sélectionnée incomplète : {sorted(missing)}")

    if sampled:
        # Quantize the sampled area to a multiple of 16.
        area = params["min_mask_region_area"]
        params["min_mask_region_area"] = int(max(0, round(area / 16)) * 16) or 16
    elif type(params["min_mask_region_area"]) is not int:
        raise ValueError(
            "Une configuration effective doit contenir min_mask_region_area "
            "comme entier. Pour des valeurs Optuna brutes, fournir l'export "
            "contenant optimized_parameters."
        )

    fixed = {
        "crop_n_points_downscale_factor": 1,
        "stability_score_offset": 0.95,
    }
    for key, value in fixed.items():
        if key in params and params[key] != value:
            raise ValueError(
                f"{key}={params[key]} est incompatible avec la recherche "
                f"originale, qui utilisait {value}. Vérifie le fichier sélectionné."
            )
        params[key] = value
    return params


def _read_selected_params(params_path: str) -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    if isinstance(blob, dict) and "effective_parameters" in blob:
        params = _complete_search_parameters(blob["effective_parameters"], sampled=False)
    elif isinstance(blob, dict) and "optimized_parameters" in blob:
        params = _complete_search_parameters(blob["optimized_parameters"], sampled=True)
    else:
        # params.json saved by objective.py is already an effective configuration.
        params = _complete_search_parameters(blob, sampled=False)
    print(f"[INFO] Paramètres sélectionnés chargés depuis {params_path}")
    return params


def _load_best_params(trial_number: int | None = None,
                      params_file: str | None = None) -> dict:
    if params_file is not None:
        return _read_selected_params(params_file)
    if trial_number is not None:
        params_path = os.path.join("optuna_results", f"trial_{trial_number}", "params.json")
        return _read_selected_params(params_path)
    if os.path.exists("best_sam2_parameters.json"):
        return _read_selected_params("best_sam2_parameters.json")
    # The last completed/created trial is not necessarily the selected trial.
    raise FileNotFoundError(
        "Impossible de trouver la configuration sélectionnée : "
        "génère best_sam2_parameters.json avec main.py, ou spécifie "
        "--trial N ou --params-file CHEMIN. Aucun trial n'est choisi automatiquement."
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Évaluation finale sur lames de TEST jamais vues.")
    ap.add_argument("--patch-dir", type=str, default="patches_data_test",
                    help="Répertoire contenant UNIQUEMENT les lames de TEST (par défaut: patches_data_test).")
    ap.add_argument("--images", nargs="*", default=None,
                    help="Liste d'image_id à tester. Si omis, détection auto depuis --patch-dir.")
    ap.add_argument("--out", type=str, default="final_test_evaluation",
                    help="Répertoire racine des sorties (default: final_test_evaluation).")
    ap.add_argument("--trial", type=int, default=None,
                    help="Numéro de trial à utiliser pour charger params.json (optionnel).")
    ap.add_argument("--params-file", "--tuned-params", dest="params_file", default=None,
                    help="JSON de paramètres sélectionnés (optionnel). Sans cet argument "
                         "ni --trial, utilise best_sam2_parameters.json créé par main.py.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Graine pour reproductibilité (default: 42).")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    patch_dir = args.patch_dir.rstrip("/")
    if not os.path.isdir(patch_dir):
        print(f"[ERROR] --patch-dir introuvable: {patch_dir}")
        sys.exit(1)

    if args.images and len(args.images) > 0:
        test_image_ids = args.images
        print(f"[INFO] TEST image_ids (CLI) = {test_image_ids}")
    else:
        test_image_ids = _discover_image_ids(patch_dir)
        print(f"[INFO] TEST image_ids (auto) = {test_image_ids}")

    if not test_image_ids:
        print("[ERROR] Aucune image_id détectée dans le dossier test. "
              "Vérifie que tes .npz contiennent bien un champ 'image_id'.")
        sys.exit(2)

    params = _load_best_params(trial_number=args.trial, params_file=args.params_file)

    print(f"[INFO] Chargement dataset TEST depuis : {os.path.abspath(patch_dir)}")
    dataset = PatchDataset(patch_dir)

    device = get_device()
    print(f"[INFO] Device: {device} | (Train/Val étaient sur: {os.path.abspath(_TRAIN_VAL_DIR)})")

    optimizer = OptimizedSAM2Optimizer(
        sam2_checkpoint=CHECKPOINT_PATH,
        model_cfg=MODEL_CFG_PATH,
        dataset=dataset,
        device=device
    )

    root_out = args.out.rstrip("/")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root_out, f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Démarrage TEST FINAL → out_dir='{out_dir}'")

    # Record the complete explicit settings passed to the tuned generator.
    # Other generator arguments retain the installed SAM2 library defaults.
    with open(os.path.join(out_dir, "applied_sam2_parameters.json"),
              "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    patch_indices = dataset.get_filtered_indices()
    allmetrics = optimizer.evaluate_patches_batch(patch_indices, params, out_dir, "FINAL_TEST_BEST")

    from metrics import MetricsCalculator
    stats = MetricsCalculator.aggregate_metrics(allmetrics)

    import pandas as pd
    stats_path = os.path.join(out_dir, "test_final_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    csv_path = os.path.join(out_dir, "test_final_detailed.csv")
    pd.DataFrame(allmetrics).to_csv(csv_path, index=False)

    if not stats:
        print("[ERROR] TEST FINAL — aucune statistique calculée.")
        sys.exit(3)

    keys = ['weighted_score_mean', 'iou_mean', 'dice_mean', 'purity_mean']
    print("\n=== TEST FINAL — Moyennes clés ===")
    for k in keys:
        if k in stats:
            std_key = k.replace('_mean', '_std')
            print(f"  {k.replace('_mean','').upper():>12s} : {stats[k]:6.2f} ± {stats.get(std_key, 0):.2f}")

    print("\n[OK] TEST FINAL terminé.")
    print(f"    - Statistiques : {out_dir}/test_final_stats.json")
    print(f"    - Détails CSV  : {out_dir}/test_final_detailed.csv")
    # === RUN DEFAULT (paramètres SAM2 d'origine) ===
    print(f"\n[INFO] Démarrage TEST FINAL DEFAULT → out_dir='{out_dir}'")

    default_dir = os.path.join(out_dir, "default")
    os.makedirs(default_dir, exist_ok=True)

    default_patch_indices = dataset.get_filtered_indices()
    default_metrics = optimizer.evaluate_patches_batch(
        default_patch_indices, None, default_dir, "FINAL_TEST_DEFAULT"
    )

    from metrics import MetricsCalculator as MCDefault
    default_stats = MCDefault.aggregate_metrics(default_metrics)

    import json as json_default, pandas as pd_default
    default_stats_path = os.path.join(default_dir, "test_final_stats_default.json")
    with open(default_stats_path, "w", encoding="utf-8") as f:
        json_default.dump(default_stats, f, indent=2, ensure_ascii=False)
    default_csv_path = os.path.join(default_dir, "test_final_detailed_default.csv")
    pd_default.DataFrame(default_metrics).to_csv(default_csv_path, index=False)

    if not default_stats:
        print("[ERROR] TEST FINAL DEFAULT — aucune statistique calculée.")
    else:
        keys = ['weighted_score_mean', 'iou_mean', 'dice_mean', 'purity_mean']
        print("\n=== TEST FINAL DEFAULT — Moyennes clés ===")
        for k in keys:
            if k in default_stats:
                std_key = k.replace('_mean', '_std')
                print(f"  {k.replace('_mean','').upper():>12s} : {default_stats[k]:6.2f} ± {default_stats.get(std_key, 0):.2f}")

        print("\n[OK] TEST FINAL DEFAULT terminé.")
        print(f"    - Statistiques : {default_stats_path}")
        print(f"    - Détails CSV  : {default_csv_path}")

    # === COMPARAISON BEST vs DEFAULT (tableau console) ===
    try:
        import pandas as pd

        with open(stats_path, "r", encoding="utf-8") as f:
            best_stats = json.load(f)
        with open(default_stats_path, "r", encoding="utf-8") as f:
            default_stats_cmp = json.load(f)

        keys = ['weighted_score_mean', 'iou_mean', 'dice_mean', 'purity_mean']
        rows = []
        for k in keys:
            row = {
                "metric": k.replace("_mean", "").upper(),
                "BEST_mean": best_stats.get(k, float("nan")),
                "BEST_std": best_stats.get(k.replace("_mean", "_std"), float("nan")),
                "DEFAULT_mean": default_stats_cmp.get(k, float("nan")),
                "DEFAULT_std": default_stats_cmp.get(k.replace("_mean", "_std"), float("nan")),
            }
            rows.append(row)

        df_cmp = pd.DataFrame(rows)
        cmp_path = os.path.join(out_dir, "best_vs_default_summary.csv")
        df_cmp.to_csv(cmp_path, index=False)

        print("\n=== COMPARAISON BEST vs DEFAULT (moyennes) ===")
        print(df_cmp.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
        print(f"\n[INFO] Tableau comparatif sauvegardé dans : {cmp_path}")
    except Exception as e:
        print(f"[WARN] Impossible de générer le tableau comparatif BEST vs DEFAULT : {e}")

    # === GRAPHIQUES DE COMPARAISON (BEST vs DEFAULT) ===
    try:
        import matplotlib.pyplot as plt

        metrics_order = ['weighted_score_mean', 'iou_mean', 'dice_mean', 'purity_mean']
        labels = [m.replace("_mean", "").upper() for m in metrics_order]
        best_vals = [best_stats.get(m, 0.0) for m in metrics_order]
        default_vals = [default_stats_cmp.get(m, 0.0) for m in metrics_order]

        x = range(len(labels))
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar([i - width/2 for i in x], best_vals, width, label="BEST")
        plt.bar([i + width/2 for i in x], default_vals, width, label="DEFAULT")
        plt.xticks(list(x), labels, rotation=15)
        plt.ylabel("Valeur moyenne")
        plt.title("BEST vs DEFAULT — Moyennes des métriques")
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(out_dir, "best_vs_default_barplot.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()

        print(f"[INFO] Graphique BEST vs DEFAULT sauvegardé dans : {fig_path}")
    except Exception as e:
        print(f"[WARN] Impossible de générer le graphique de comparaison : {e}")



if __name__ == "__main__":
    main()

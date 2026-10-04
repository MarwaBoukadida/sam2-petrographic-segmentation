# -*- coding: utf-8 -*-
"""Plots for mask scores and optimization trials."""

import os
import json
import math
import logging
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

from config import (
    ANALYSIS_FIGURES,
    METRICS_FOR_ANALYSIS,
    DEFAULT_SCORE,
    TOP_K_TRIALS,
    WORST_K_PERCENT,
    PDP_BINS,
)

logger = logging.getLogger(__name__)

def _safe_read_json(path: str) -> dict:
    """Read JSON; return an empty dictionary on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _safe_read_csv(path: str) -> pd.DataFrame:
    """Read CSV; return an empty DataFrame on failure."""
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _ensure_dir(d: str):
    """Create the output directory if needed."""
    os.makedirs(d, exist_ok=True)

def generate_trial_figures(trial_dir: str) -> None:
    """Plot mask-score distributions from a trial all_metrics.csv file."""
    csv_path = os.path.join(trial_dir, "all_metrics.csv")
    df = _safe_read_csv(csv_path)
    if df.empty:
        logger.debug(f"[analysis] Pas de all_metrics.csv pour {trial_dir}")
        return

    cols = {c.lower(): c for c in df.columns}
    wcol = "weighted_score" if "weighted_score" in df.columns else cols.get("weighted", None)
    pcol = "purity" if "purity" in df.columns else cols.get("purity", None)
    icol = "image_id" if "image_id" in df.columns else cols.get("image_id", None)

    if wcol:
        try:
            plt.figure(figsize=(6, 4))
            df[wcol].dropna().plot(kind="hist", bins=20)
            plt.title("Histogramme — Weighted score")
            plt.xlabel("weighted_score"); plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(os.path.join(trial_dir, "hist_weighted_score.png"), dpi=150)
            plt.close()
        except Exception as e:
            logger.debug(f"hist weighted fail: {e}")

    if pcol:
        try:
            plt.figure(figsize=(6, 4))
            df[pcol].dropna().plot(kind="hist", bins=20)
            plt.title("Histogramme — Purity")
            plt.xlabel("purity"); plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(os.path.join(trial_dir, "hist_purity.png"), dpi=150)
            plt.close()
        except Exception as e:
            logger.debug(f"hist purity fail: {e}")

    if wcol and icol and df[icol].nunique() > 1:
        try:
            plt.figure(figsize=(8, 4))
            df.boxplot(column=wcol, by=icol, rot=45)
            plt.suptitle("")
            plt.title("Box — weighted_score par image")
            plt.xlabel("image_id"); plt.ylabel("weighted_score")
            plt.tight_layout()
            plt.savefig(os.path.join(trial_dir, "box_by_image_weighted_score.png"), dpi=150)
            plt.close()
        except Exception as e:
            logger.debug(f"box_by_image fail: {e}")

    if WORST_K_PERCENT and wcol:
        try:
            k = max(1, int(len(df) * (WORST_K_PERCENT / 100.0)))
            worst = df.sort_values(wcol).head(k)
            plt.figure(figsize=(6, 4))
            plt.plot(np.arange(len(worst)), worst[wcol].values, marker="o")
            plt.title(f"Worst {WORST_K_PERCENT}% — weighted_score")
            plt.xlabel("samples (asc)"); plt.ylabel("weighted_score")
            plt.tight_layout()
            plt.savefig(os.path.join(trial_dir, "worstk_weighted_score.png"), dpi=150)
            plt.close()
        except Exception as e:
            logger.debug(f"worstK fail: {e}")

def run_posthoc_analysis(results_dir: str, analysis_dir: str) -> None:
    """Read trial metrics and parameters from results_dir/trials_details/trial_*. Parameter plots use median trial scores; PDP plots show quantile-bin means. The cost proxy is the number of metric rows."""
    trials_root = os.path.join(results_dir, "trials_details")
    if not os.path.isdir(trials_root):
        logger.warning(f"[analysis] Dossier introuvable: {trials_root}")
        return

    _ensure_dir(analysis_dir)

    rows = []
    params_rows = []
    trial_dirs = sorted(glob(os.path.join(trials_root, "trial_*")))
    if not trial_dirs:
        logger.warning("[analysis] Aucun trial trouvé.")
        return

    for tdir in trial_dirs:
        tnum = os.path.basename(tdir).split("_")[-1]

        try:
            generate_trial_figures(tdir)
        except Exception:
            pass

        df = _safe_read_csv(os.path.join(tdir, "all_metrics.csv"))
        if not df.empty:
            df["trial"] = int(tnum)
            rows.append(df)

        p = _safe_read_json(os.path.join(tdir, "params.json"))
        if p:
            flat = {"trial": int(tnum)}
            for k, v in p.items():

                if isinstance(v, (int, float, str, bool, np.integer, np.floating)):
                    flat[k] = float(v) if isinstance(v, (np.integer, np.floating)) else v
            params_rows.append(flat)

    if not rows:
        logger.warning("[analysis] Pas de all_metrics.csv au niveau global.")
        return

    df_all = pd.concat(rows, ignore_index=True)
    df_params = pd.DataFrame(params_rows).set_index("trial") if params_rows else pd.DataFrame()

    score_col = DEFAULT_SCORE if DEFAULT_SCORE in df_all.columns else (
        "weighted_score" if "weighted_score" in df_all.columns else df_all.columns[0]
    )

    if "box_by_trial" in ANALYSIS_FIGURES and score_col in df_all.columns:
        try:
            plt.figure(figsize=(8, 4))
            df_all.boxplot(column=score_col, by="trial", rot=0)
            plt.suptitle("")
            plt.title(f"Box — {score_col} par trial")
            plt.xlabel("trial"); plt.ylabel(score_col)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f"box_by_trial_{score_col}.png"), dpi=150)
            plt.close()
        except Exception as e:
            logger.debug(f"box_by_trial fail: {e}")

    if "parallel_coords" in ANALYSIS_FIGURES and not df_params.empty and score_col in df_all.columns:
        try:

            g = df_all.groupby("trial")[score_col].median().rename("score_med")
            pc = df_params.copy()
            pc = pc.join(g, how="inner")

            num_cols = [c for c in pc.columns if c != "score_med" and pd.api.types.is_numeric_dtype(pc[c])]
            for c in num_cols:
                v = pc[c].astype(float)
                rng = v.max() - v.min()
                pc[c] = (v - v.min()) / (rng if rng != 0 else 1.0)

            thr = pc["score_med"].median()
            pc["class"] = np.where(pc["score_med"] >= thr, "High", "Low")
            plot_df = pc[num_cols + ["class"]].copy()
            if len(num_cols) >= 2:
                plt.figure(figsize=(max(8, len(num_cols)*1.4), 4))
                parallel_coordinates(plot_df, "class", alpha=0.6)
                plt.title("Paramètres (normalisés) — coords parallèles")
                plt.tight_layout()
                plt.savefig(os.path.join(analysis_dir, "parallel_coords.png"), dpi=150)
                plt.close()
        except Exception as e:
            logger.debug(f"parallel_coords fail: {e}")

    if "importance" in ANALYSIS_FIGURES and not df_params.empty and score_col in df_all.columns:
        try:
            g = df_all.groupby("trial")[score_col].median().rename("score_med")
            im = df_params.join(g, how="inner").copy()
            corrs = []
            for c in im.columns:
                if c in ("score_med",) or not pd.api.types.is_numeric_dtype(im[c]):
                    continue
                s = im[c].astype(float)
                corr = s.corr(im["score_med"], method="spearman")
                if not np.isnan(corr):
                    corrs.append((c, float(corr)))
            if corrs:
                corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
                names, vals = zip(*corrs)
                plt.figure(figsize=(8, max(3, len(names)*0.35)))
                plt.barh(range(len(names)), vals)
                plt.yticks(range(len(names)), names)
                plt.axvline(0, color="k", lw=0.5)
                plt.title("Importance (Spearman) vs score médian par trial")
                plt.tight_layout()
                plt.savefig(os.path.join(analysis_dir, "param_importance_spearman.png"), dpi=150)
                plt.close()
        except Exception as e:
            logger.debug(f"importance fail: {e}")

    if "pdp" in ANALYSIS_FIGURES and not df_params.empty and score_col in df_all.columns:
        try:
            g = df_all.groupby("trial")[score_col].median().rename("score_med")
            im = df_params.join(g, how="inner").copy()

            num_cols = [c for c in im.columns if c not in ("score_med",) and pd.api.types.is_numeric_dtype(im[c])]
            for c in num_cols:
                s = im[[c, "score_med"]].dropna().astype(float)
                if len(s) < max(6, PDP_BINS):
                    continue

                b = PDP_BINS
                s["bin"] = pd.qcut(s[c], q=b, duplicates="drop")
                mu = s.groupby("bin")["score_med"].mean()
                xs = [iv.mid for iv in mu.index.categories] if hasattr(mu.index, "categories") else range(len(mu))
                plt.figure(figsize=(6, 4))
                plt.plot(range(len(mu)), mu.values, marker="o")
                plt.title(f"PDP — {c} → score_med")
                plt.xlabel(f"{c} (bins quantiles)"); plt.ylabel("score_med")
                plt.tight_layout()
                plt.savefig(os.path.join(analysis_dir, f"pdp_{c}.png"), dpi=150)
                plt.close()
        except Exception as e:
            logger.debug(f"pdp fail: {e}")

    if "heatmap2d" in ANALYSIS_FIGURES and not df_params.empty and score_col in df_all.columns:
        try:

            g = df_all.groupby("trial")[score_col].median().rename("score_med")
            im = df_params.join(g, how="inner").copy()
            corrs = []
            for c in im.columns:
                if c in ("score_med",) or not pd.api.types.is_numeric_dtype(im[c]):
                    continue
                corr = im[c].astype(float).corr(im["score_med"], method="spearman")
                if not np.isnan(corr):
                    corrs.append((c, float(abs(corr))))
            if len(corrs) >= 2:
                corrs = sorted(corrs, key=lambda x: x[1], reverse=True)
                p1, p2 = corrs[0][0], corrs[1][0]
                sub = im[[p1, p2, "score_med"]].dropna()
                if len(sub) >= 6:

                    xbin = min(6, max(3, len(sub)//3))
                    ybin = xbin
                    xv = pd.qcut(sub[p1], q=xbin, labels=False, duplicates="drop")
                    yv = pd.qcut(sub[p2], q=ybin, labels=False, duplicates="drop")
                    M = np.full((yv.max()+1, xv.max()+1), np.nan)
                    for xi in range(M.shape[1]):
                        for yi in range(M.shape[0]):
                            sel = sub[(xv==xi) & (yv==yi)]["score_med"]
                            if len(sel):
                                M[yi, xi] = sel.mean()
                    plt.figure(figsize=(6, 5))
                    imshow = plt.imshow(M, origin="lower", aspect="auto")
                    plt.colorbar(imshow, label="score_med")
                    plt.title(f"Heatmap 2D — {p1} vs {p2}")
                    plt.xlabel(p1); plt.ylabel(p2)
                    plt.tight_layout()
                    plt.savefig(os.path.join(analysis_dir, f"heatmap2d_{p1}_vs_{p2}.png"), dpi=150)
                    plt.close()
        except Exception as e:
            logger.debug(f"heatmap2d fail: {e}")

    if "pareto" in ANALYSIS_FIGURES and score_col in df_all.columns:
        try:

            agg = df_all.groupby("trial")[score_col].median().to_frame("score_med")
            agg["count"] = df_all.groupby("trial")[score_col].size()
            plt.figure(figsize=(6, 4))
            plt.scatter(agg["count"], agg["score_med"])
            plt.xlabel("#lignes métriques (≈masques)")
            plt.ylabel("score_med")
            plt.title("Pareto — score vs coût (proxy)")
            for t, r in agg.iterrows():
                plt.annotate(str(t), (r["count"], r["score_med"]), fontsize=8, xytext=(2,2), textcoords="offset points")
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, "pareto_score_vs_cost.png"), dpi=150)
            plt.close()
        except Exception as e:
            logger.debug(f"pareto fail: {e}")

    logger.info(f"[analysis] Post-hoc terminé → {analysis_dir}")

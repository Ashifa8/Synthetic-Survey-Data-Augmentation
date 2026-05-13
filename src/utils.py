"""
src/utils.py
Evaluation utilities: quality metrics, classification evaluation,
feature importance concordance, novelty experiments, and plotting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, pearsonr, spearmanr
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)


# ------------------------------------------------------------------ #
#  Synthetic Data Quality                                              #
# ------------------------------------------------------------------ #

def column_shape_score(real: pd.DataFrame, synth: pd.DataFrame, cols: list) -> float:
    """KS-based column shape score (mean over all feature columns)."""
    scores = []
    for col in cols:
        ks_stat, _ = ks_2samp(real[col].dropna(), synth[col].dropna())
        scores.append(1 - ks_stat)
    return float(np.mean(scores))


def column_pair_trend_score(real: pd.DataFrame, synth: pd.DataFrame, cols: list) -> float:
    """Pearson-correlation-based pairwise trend score."""
    scores = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r_corr = real[cols[i]].corr(real[cols[j]])
            s_corr = synth[cols[i]].corr(synth[cols[j]])
            scores.append(1 - (s_corr - r_corr) ** 2)
    return float(np.mean(scores))


def evaluate_quality(original_df, datasets: dict, feature_cols: list) -> dict:
    """Compute CSS, CPTS, OQS for the three pure-synthetic datasets."""
    quality = {}
    for name in ["Synthesized (CTGAN)", "Synthesized (TVAE)", "Synthesized (CopulaGAN)"]:
        synth = datasets[name]
        cs  = column_shape_score(original_df, synth, feature_cols)
        cpt = column_pair_trend_score(original_df, synth, feature_cols)
        oqs = (cs + cpt) / 2
        quality[name] = {"Column Shape": cs, "Column Pair Trend": cpt, "Overall Quality": oqs}
        print(f"  {name}: CS={cs:.4f}, CPT={cpt:.4f}, OQS={oqs:.4f}")
    return quality


# ------------------------------------------------------------------ #
#  ML Classification                                                   #
# ------------------------------------------------------------------ #

def evaluate_dataset(df: pd.DataFrame, clf, feature_cols: list,
                     target: str, random_state: int = 42):
    """
    Train clf on 64% of df, evaluate on held-out 20%.
    Returns (metrics_dict, feature_importances_or_None).
    """
    X = df[feature_cols].values
    y = df[target].values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    X_train, _, y_train, _ = train_test_split(
        X_temp, y_temp, test_size=0.20, random_state=random_state
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1":        f1_score(y_test, y_pred, zero_division=0),
        "AUC":       roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
    }
    feat_imp = clf.feature_importances_ if hasattr(clf, "feature_importances_") else None
    return metrics, feat_imp


def run_classification(datasets: dict, classifiers: dict,
                       feature_cols: list, target: str,
                       random_state: int = 42):
    """
    Evaluate all classifiers on every dataset configuration.
    Returns (results, feature_importances) dicts.
    """
    results, feature_importances = {}, {}

    for ds_name, df in datasets.items():
        ds_metrics, ds_fi = [], []
        for clf_name, clf in classifiers.items():
            try:
                m, fi = evaluate_dataset(df, clf, feature_cols, target, random_state)
                ds_metrics.append(m)
                if fi is not None:
                    ds_fi.append(fi)
            except Exception as e:
                print(f"  [WARN] {ds_name}/{clf_name}: {e}")

        avg = {k: np.mean([m[k] for m in ds_metrics]) for k in ds_metrics[0]}
        std = {k: np.std([m[k]  for m in ds_metrics]) for k in ds_metrics[0]}
        results[ds_name] = {"avg": avg, "std": std}
        feature_importances[ds_name] = np.mean(ds_fi, axis=0) if ds_fi else None
        print(f"  {ds_name}: Acc={avg['Accuracy']:.4f}, F1={avg['F1']:.4f}, AUC={avg['AUC']:.4f}")

    return results, feature_importances


# ------------------------------------------------------------------ #
#  Feature Importance Concordance                                      #
# ------------------------------------------------------------------ #

def concordance_analysis(feature_importances: dict) -> dict:
    """Pearson r and Spearman rho vs Real (Original) benchmark."""
    orig_fi = feature_importances["Real (Original)"]
    concordance = {}
    for ds_name, fi in feature_importances.items():
        if fi is None or ds_name == "Real (Original)":
            continue
        pr, _ = pearsonr(orig_fi, fi)
        sr, _ = spearmanr(orig_fi, fi)
        concordance[ds_name] = {"Pearson r": pr, "Spearman rho": sr}
        print(f"  {ds_name}: Pearson r={pr:.4f}, Spearman rho={sr:.4f}")
    return concordance


# ------------------------------------------------------------------ #
#  Novelty Experiments                                                 #
# ------------------------------------------------------------------ #

def novelty_ratio_sweep(seed_df, copulagan_synth, classifiers: dict,
                        feature_cols: list, target: str, cfg: dict) -> dict:
    """
    NOVELTY 1: Sweep synthetic-to-seed augmentation ratios
    using CopulaGAN and measure downstream F1 / AUC.
    """
    ratios = cfg["novelty"]["ratio_sweep"]
    ratio_results = {}

    for ratio in ratios:
        n_synth = int(len(seed_df) * ratio)
        syn_ratio = copulagan_synth.sample(num_rows=n_synth)
        df_aug = pd.concat([seed_df, syn_ratio], ignore_index=True)

        metrics_list = []
        for clf_name, clf in classifiers.items():
            try:
                m, _ = evaluate_dataset(df_aug, clf, feature_cols, target)
                metrics_list.append(m)
            except Exception:
                pass

        avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
        ratio_results[ratio] = avg
        print(f"  Ratio {ratio}x: F1={avg['F1']:.4f}, AUC={avg['AUC']:.4f}")

    return ratio_results


def novelty_ensemble_dataset(seed_df, syn_ctgan, syn_tvae, syn_copulagan,
                              classifiers: dict, feature_cols: list,
                              target: str, cfg: dict) -> dict:
    """
    NOVELTY 2: 12th dataset config — equal mix of all three generators.
    """
    n_each = cfg["novelty"]["ensemble_n_each"]
    syn_ensemble = pd.concat([
        syn_ctgan.sample(n=n_each, random_state=42),
        syn_tvae.sample(n=n_each, random_state=42),
        syn_copulagan.sample(n=n_each, random_state=42),
    ], ignore_index=True)

    df_seed_ensemble = pd.concat([seed_df, syn_ensemble], ignore_index=True)

    ens_metrics = []
    for clf_name, clf in classifiers.items():
        m, fi = evaluate_dataset(df_seed_ensemble, clf, feature_cols, target)
        ens_metrics.append(m)

    avg_ens = {k: np.mean([m[k] for m in ens_metrics]) for k in ens_metrics[0]}
    print(f"Seed + Ensemble: F1={avg_ens['F1']:.4f}, AUC={avg_ens['AUC']:.4f}")
    return {"Seed + Ensemble (All3)": avg_ens}


def novelty_cross_validation(datasets: dict, classifiers: dict,
                              feature_cols: list, target: str,
                              n_splits: int = 5) -> dict:
    """
    NOVELTY 4: 5-Fold Stratified Cross-Validation on all datasets.
    """
    cv_results = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for ds_name, df in datasets.items():
        X = df[feature_cols].values
        y = df[target].values
        fold_f1, fold_auc = [], []

        for clf_name, clf in classifiers.items():
            for train_idx, test_idx in skf.split(X, y):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)
                y_prob = clf.predict_proba(X_te)[:, 1]
                fold_f1.append(f1_score(y_te, y_pred, zero_division=0))
                fold_auc.append(roc_auc_score(y_te, y_prob))

        cv_results[ds_name] = {
            "F1_mean":  float(np.mean(fold_f1)),
            "F1_std":   float(np.std(fold_f1)),
            "AUC_mean": float(np.mean(fold_auc)),
            "AUC_std":  float(np.std(fold_auc)),
        }
        print(f"  {ds_name}: F1={cv_results[ds_name]['F1_mean']:.4f} "
              f"±{cv_results[ds_name]['F1_std']:.4f}")

    return cv_results


# ------------------------------------------------------------------ #
#  Plotting                                                            #
# ------------------------------------------------------------------ #

def plot_2d_maps(results: dict, feature_importances: dict, out_dir: str):
    """Figure 9 reproduction — 2D performance/stability and concordance maps."""
    orig_fi = feature_importances.get("Real (Original)")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    ax.set_title("Fig 9A: Predictive Performance vs Stability", fontsize=13, fontweight="bold")
    for ds_name, res in results.items():
        avg = np.mean(list(res["avg"].values()))
        std = np.mean(list(res["std"].values()))
        ax.scatter(avg, std, s=80, zorder=5)
        ax.annotate(ds_name, (avg, std), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Average Evaluation Metric (higher = better)", fontsize=11)
    ax.set_ylabel("Standard Deviation (lower = more stable)", fontsize=11)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.set_title("Fig 9B: Predictive Performance vs Concordance", fontsize=13, fontweight="bold")
    for ds_name, res in results.items():
        avg = np.mean(list(res["avg"].values()))
        fi  = feature_importances.get(ds_name)
        if fi is not None and ds_name != "Real (Original)" and orig_fi is not None:
            sr, _ = spearmanr(orig_fi, fi)
            ax.scatter(avg, sr, s=80, zorder=5)
            ax.annotate(ds_name, (avg, sr), fontsize=7, ha="left", va="bottom")
        elif ds_name == "Real (Original)":
            ax.scatter(avg, 1.0, s=120, marker="*", color="gold", zorder=6)
            ax.annotate("Original (Benchmark)", (avg, 1.0), fontsize=7)
    ax.set_xlabel("Average Evaluation Metric (higher = better)", fontsize=11)
    ax.set_ylabel("Spearman's rho with Original (higher = better)", fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure9_2D_maps.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved figure9_2D_maps.png → {out_dir}")


def plot_ratio_sweep(ratio_results: dict, out_dir: str):
    """Plot F1 vs synthetic-to-seed ratio (Novelty 1)."""
    ratios = list(ratio_results.keys())
    f1s    = [ratio_results[r]["F1"] for r in ratios]

    plt.figure(figsize=(8, 5))
    plt.plot(ratios, f1s, "o-", color="steelblue", linewidth=2, markersize=8)
    plt.xlabel("Synthetic-to-Seed Ratio", fontsize=12)
    plt.ylabel("Average F1 Score", fontsize=12)
    plt.title("Optimal Augmentation Ratio (CopulaGAN)", fontsize=13, fontweight="bold")
    plt.xticks(ratios, [f"{r}x" for r in ratios])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "novelty1_ratio_sweep.png"), dpi=150)
    plt.show()
    print(f"Saved novelty1_ratio_sweep.png → {out_dir}")

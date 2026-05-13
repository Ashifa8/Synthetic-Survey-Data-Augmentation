"""
train.py
End-to-end training pipeline for DGNN-based synthetic data augmentation.

Usage (Kaggle / Colab):
    python train.py                           # runs all steps
    python train.py --skip-synthesis          # reuse saved checkpoints
"""

import argparse
import json
import os
import yaml
import pandas as pd
import numpy as np

from src.dataset import load_data, preprocess, build_sdv_metadata, build_datasets
from src.model   import train_synthesizers, load_synthesizers, get_classifiers
from src.utils   import (
    evaluate_quality, run_classification, concordance_analysis,
    novelty_ratio_sweep, novelty_ensemble_dataset, novelty_cross_validation,
    plot_2d_maps, plot_ratio_sweep,
)


# ------------------------------------------------------------------ #
#  Helper: JSON serialisation (numpy types → native Python)           #
# ------------------------------------------------------------------ #
class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NpEncoder)
    print(f"Saved → {path}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main(args):
    # 1. Load config
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    target      = cfg["data"]["target_column"]
    results_dir = cfg["paths"]["results_dir"]
    fig_dir     = cfg["paths"]["figures_dir"]
    rs          = cfg["training"]["random_state"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir,     exist_ok=True)

    # 2. Load & preprocess data
    print("\n=== STEP 1-3: Data Loading & Preprocessing ===")
    original_df, seed_df = load_data(cfg)
    original_df, seed_df, feature_cols, le = preprocess(original_df, seed_df, cfg)

    # 3. Build SDV metadata
    print("\n=== STEP 4: Build SDV Metadata ===")
    metadata = build_sdv_metadata(seed_df, feature_cols, target)

    # 4. Train or load synthesizers
    if args.skip_synthesis:
        print("\n=== STEP 5: Loading Saved Synthesizers ===")
        synthesizers = load_synthesizers(cfg)
    else:
        print("\n=== STEP 5: Training DGNN Synthesizers ===")
        synthesizers = train_synthesizers(seed_df, metadata, cfg)

    syn_ctgan     = synthesizers["ctgan"].sample(num_rows=len(seed_df))
    syn_tvae      = synthesizers["tvae"].sample(num_rows=len(seed_df))
    syn_copulagan = synthesizers["copulagan"].sample(num_rows=len(seed_df))

    # 5. Build 11 dataset configurations
    print("\n=== STEP 6: Building 11 Dataset Configurations ===")
    datasets = build_datasets(original_df, seed_df, syn_ctgan, syn_tvae, syn_copulagan)

    # 6. Synthetic data quality
    print("\n=== STEP 7: Synthetic Data Quality Evaluation ===")
    quality_results = evaluate_quality(original_df, datasets, feature_cols)
    save_json(quality_results, os.path.join(results_dir, "quality_scores.json"))

    # 7. ML classification
    print("\n=== STEP 8: ML Classification (4 classifiers × 11 datasets) ===")
    classifiers = get_classifiers(random_state=rs)
    results, feature_importances = run_classification(
        datasets, classifiers, feature_cols, target, rs
    )

    # Save baseline (Real Original) vs improved (Seed + CopulaGAN)
    save_json(results["Real (Original)"]["avg"],       os.path.join(results_dir, "baseline_metrics.json"))
    save_json(results["Seed + CopulaGAN"]["avg"],      os.path.join(results_dir, "improved_metrics.json"))

    # Save full results
    save_json({k: {"avg": v["avg"], "std": v["std"]} for k, v in results.items()},
              os.path.join(results_dir, "all_classification_results.json"))

    # training log CSV
    log_rows = []
    for ds_name, res in results.items():
        row = {"dataset": ds_name}
        row.update({f"avg_{k}": v for k, v in res["avg"].items()})
        row.update({f"std_{k}": v for k, v in res["std"].items()})
        log_rows.append(row)
    pd.DataFrame(log_rows).to_csv(os.path.join(results_dir, "training_log.csv"), index=False)
    print(f"Saved training_log.csv → {results_dir}")

    # 8. Feature importance concordance
    print("\n=== STEP 9: Feature Importance Concordance ===")
    concordance = concordance_analysis(feature_importances)
    save_json(concordance, os.path.join(results_dir, "concordance.json"))

    # 9. Plots
    print("\n=== STEP 10: Generating Plots ===")
    plot_2d_maps(results, feature_importances, fig_dir)

    # 10. Novelty experiments
    print("\n=== NOVELTY 1: Augmentation Ratio Sweep ===")
    ratio_results = novelty_ratio_sweep(
        seed_df, synthesizers["copulagan"], classifiers, feature_cols, target, cfg
    )
    save_json(ratio_results, os.path.join(results_dir, "novelty1_ratio_sweep.json"))
    plot_ratio_sweep(ratio_results, fig_dir)

    print("\n=== NOVELTY 2: Ensemble Synthetic Dataset ===")
    ens_results = novelty_ensemble_dataset(
        seed_df, syn_ctgan, syn_tvae, syn_copulagan,
        classifiers, feature_cols, target, cfg
    )
    save_json(ens_results, os.path.join(results_dir, "novelty2_ensemble.json"))

    print("\n=== NOVELTY 4: 5-Fold Cross-Validation ===")
    cv_results = novelty_cross_validation(
        datasets, classifiers, feature_cols, target,
        n_splits=cfg["training"]["cv_folds"]
    )
    save_json(cv_results, os.path.join(results_dir, "novelty4_cv_results.json"))

    print("\n✅ TRAINING PIPELINE COMPLETE!")
    print(f"   All results saved in → {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGNN Augmentation Training Pipeline")
    parser.add_argument(
        "--skip-synthesis", action="store_true",
        help="Skip synthesizer training; load from checkpoints instead."
    )
    args = parser.parse_args()
    main(args)

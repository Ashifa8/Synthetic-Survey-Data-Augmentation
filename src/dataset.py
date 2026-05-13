"""
src/dataset.py
Data loading, preprocessing, and dataset construction utilities.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sdv.metadata import SingleTableMetadata


def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load original and seed DataFrames from Excel."""
    xl_path = cfg["data"]["excel_path"]
    original_df = pd.read_excel(xl_path, sheet_name=cfg["data"]["sheet_original"])
    seed_df     = pd.read_excel(xl_path, sheet_name=cfg["data"]["sheet_seed"])
    print(f"Loaded — Original: {original_df.shape}, Seed: {seed_df.shape}")
    return original_df, seed_df


def preprocess(original_df: pd.DataFrame, seed_df: pd.DataFrame, cfg: dict):
    """
    Drop admin columns, label-encode target consistently,
    and return (original_df, seed_df, feature_cols, label_encoder).
    """
    target  = cfg["data"]["target_column"]
    drops   = cfg["data"]["drop_columns"]

    original_df = original_df.drop(columns=drops, errors="ignore").copy()
    seed_df     = seed_df.drop(columns=drops, errors="ignore").copy()

    # Fit on original so seed shares the same encoding
    le = LabelEncoder()
    original_df[target] = le.fit_transform(original_df[target])
    seed_df[target]     = le.transform(seed_df[target])

    feature_cols = [c for c in original_df.columns if c != target]
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    return original_df, seed_df, feature_cols, le


def build_sdv_metadata(df: pd.DataFrame, feature_cols: list, target: str) -> SingleTableMetadata:
    """
    Build SDV SingleTableMetadata with all feature columns forced to numerical
    and target forced to categorical (prevents silent type mismatches).
    """
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    for col in feature_cols:
        meta.update_column(col, sdtype="numerical")
    meta.update_column(target, sdtype="categorical")
    return meta


def build_datasets(original_df, seed_df,
                   syn_ctgan, syn_tvae, syn_copulagan) -> dict:
    """Construct all 11 standard dataset configurations."""
    datasets = {
        "Real (Original)":          original_df,
        "Real (Seed)":              seed_df,
        "Synthesized (CTGAN)":      syn_ctgan,
        "Synthesized (TVAE)":       syn_tvae,
        "Synthesized (CopulaGAN)":  syn_copulagan,
        "Original + CTGAN":         pd.concat([original_df, syn_ctgan],     ignore_index=True),
        "Original + TVAE":          pd.concat([original_df, syn_tvae],      ignore_index=True),
        "Original + CopulaGAN":     pd.concat([original_df, syn_copulagan], ignore_index=True),
        "Seed + CTGAN":             pd.concat([seed_df, syn_ctgan],         ignore_index=True),
        "Seed + TVAE":              pd.concat([seed_df, syn_tvae],          ignore_index=True),
        "Seed + CopulaGAN":         pd.concat([seed_df, syn_copulagan],     ignore_index=True),
    }
    return datasets

"""
inference.py
Load trained synthesizers and generate synthetic samples on demand.
Optionally evaluate a new CSV against the saved classifiers.

Usage:
    python inference.py --model copulagan --n-rows 200
    python inference.py --model all --n-rows 100 --out-dir results/generated
"""

import argparse
import os
import yaml
import pandas as pd

from src.model import load_synthesizers


def main(args):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    out_dir = args.out_dir or os.path.join(cfg["paths"]["results_dir"], "generated")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading synthesizers from checkpoints...")
    synthesizers = load_synthesizers(cfg)

    models_to_run = (
        list(synthesizers.keys()) if args.model == "all" else [args.model]
    )

    for model_name in models_to_run:
        if model_name not in synthesizers:
            print(f"[WARN] Unknown model '{model_name}'. Skipping.")
            continue

        print(f"Generating {args.n_rows} rows with {model_name}...")
        synth = synthesizers[model_name].sample(num_rows=args.n_rows)

        out_path = os.path.join(out_dir, f"synthetic_{model_name}_{args.n_rows}rows.csv")
        synth.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")
        print(f"  Shape : {synth.shape}")
        print(f"  Preview:\n{synth.head(3)}\n")

    print("✅ Inference complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGNN Inference — generate synthetic rows")
    parser.add_argument(
        "--model", type=str, default="copulagan",
        choices=["ctgan", "tvae", "copulagan", "all"],
        help="Which synthesizer to use (default: copulagan)."
    )
    parser.add_argument(
        "--n-rows", type=int, default=200,
        help="Number of synthetic rows to generate (default: 200)."
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (default: results/generated/)."
    )
    args = parser.parse_args()
    main(args)

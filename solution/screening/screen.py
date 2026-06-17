"""
Dataset screening for training-data reconstruction.

Screens any molecule dataset through the task models and ranks by
raw logit certainty — the key signal for membership inference in
overfit models.

Usage:
    pixi run python solution/screening/screen.py \
        --data data/hERG_data.csv \
        --smiles-col SMILES \
        --output results/best/

Experiment 3 (certainty): |logit_class0 - logit_class1| — molecules
the model is most decisive about are most likely training members.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from solution.screening.utils import ROOT, load_model, featurize


def screen(
    smiles_list: list[str],
    model,
    n_classes: int,
    top_n: int = 1000,
    batch_size: int = 2048,
) -> pd.DataFrame:
    fps, valid = featurize(smiles_list)
    if not fps:
        raise ValueError("No valid SMILES found.")

    all_logits = []
    for i in range(0, len(fps), batch_size):
        X = torch.from_numpy(np.stack(fps[i : i + batch_size]))
        with torch.no_grad():
            all_logits.append(model(X).numpy())
    logits = np.concatenate(all_logits)

    print(f"  {len(valid):,} valid molecules | logit range: {logits.min():.2f} – {logits.max():.2f}")

    if n_classes == 2:
        # Experiment 3: certainty = |logit_class0 - logit_class1|
        certainty = np.abs(logits[:, 0] - logits[:, 1])
        top = np.argsort(certainty)[::-1][:top_n]
        print(f"  Certainty (top-1000): min={certainty[top[-1]]:.2f}  max={certainty[top[0]]:.2f}")
        return pd.DataFrame({"smiles": [valid[i] for i in top]})
    else:
        # Per-class: top-(top_n // n_classes) by prediction margin
        per_class = top_n // n_classes
        predicted = np.argmax(logits, axis=1)
        sorted_logits = np.sort(logits, axis=1)[:, ::-1]
        margin = sorted_logits[:, 0] - sorted_logits[:, 1]

        results = []
        for c in range(n_classes):
            class_margin = np.where(predicted == c, margin, -np.inf)
            top_c = np.argsort(class_margin)[::-1][:per_class]
            mols = [valid[i] for i in top_c if predicted[i] == c][:per_class]
            results.extend(mols)
            print(f"  class_{c}: {(predicted == c).sum():,} predicted | top margin={margin[top_c[0]]:.2f} | {len(mols)} collected")
        return pd.DataFrame({"smiles": results})


def main():
    parser = argparse.ArgumentParser(description="Screen a molecule dataset through Task 1 and Task 2 models.")
    parser.add_argument("--data", required=True, help="Path to CSV/TSV file with SMILES")
    parser.add_argument("--smiles-col", default="SMILES", help="Column name for SMILES (default: SMILES)")
    parser.add_argument("--output", default="results/best/", help="Output directory")
    parser.add_argument("--sep", default=None, help="Separator (auto-detected from extension)")
    args = parser.parse_args()

    data_path = Path(args.data)
    sep = args.sep or ("\t" if data_path.suffix == ".tsv" else ",")
    df = pd.read_csv(data_path, sep=sep)
    smiles = df[args.smiles_col].dropna().tolist()
    print(f"Loaded {len(smiles):,} molecules from {data_path.name}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = data_path.stem

    print("\n── Task 1 (binary classifier) ──")
    model1 = load_model(1)
    results1 = screen(smiles, model1, n_classes=2)
    out1 = out_dir / f"task1_{stem}.csv"
    results1.to_csv(out1, index=False)
    print(f"  Saved {len(results1)} → {out1}")

    print("\n── Task 2 (4-class classifier) ──")
    model2 = load_model(2)
    results2 = screen(smiles, model2, n_classes=4)
    out2 = out_dir / f"task2_{stem}.csv"
    results2.to_csv(out2, index=False)
    print(f"  Saved {len(results2)} → {out2}")


if __name__ == "__main__":
    main()

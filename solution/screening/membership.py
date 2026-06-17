"""
Membership inference via cross-entropy gradient norm.

For an overfit model, training molecules sit at loss minima.
By definition, the gradient of the loss at a minimum is zero.
So: small gradient norm → model is at a loss minimum → likely a training member.

Pipeline:
  1. Load CARBIDE data_splits.tsv (molecules used in the train/test CV splits)
  2. Keep only training folds (Fold != 0 — fold 0 assumed held-out with seed=0)
  3. Featurize each molecule to ECFP4 (2048 bits)
  4. Backpropagate cross-entropy loss through the task model to the input
  5. Rank by gradient norm ascending — smallest norm = most memorized
  6. Task 1: export top-1000
     Task 2: assign each molecule to the class whose gradient norm is smallest,
             export top-250 per class

Usage:
    PYTHONPATH=. pixi run python solution/screening/membership.py

Output:
    results/best/task1_submission.csv
    results/best/task2_submission.csv
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path

from solution.screening.utils import ROOT, load_model, featurize

DATA_SPLITS = ROOT / "data" / "carbide" / "data_splits.tsv"
OUT_DIR = ROOT / "results" / "best"
TEST_FOLD = 0       # fold held out for evaluation (seed=0 convention)
TOP_TASK1 = 1000
TOP_PER_CLASS = 250
N_CLASSES = 4


def gradient_norms(
    model: torch.nn.Module,
    fps: list[np.ndarray],
    class_idx: int | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Compute gradient of cross-entropy loss w.r.t. each input fingerprint.

    Args:
        model:      task model in eval mode
        fps:        list of ECFP4 numpy arrays
        class_idx:  if given, use this as the target class for all molecules;
                    otherwise use each molecule's own predicted class
        batch_size: number of molecules per forward/backward pass

    Returns:
        1-D array of gradient norms, one per molecule
    """
    norms = []
    for i in range(0, len(fps), batch_size):
        X = torch.from_numpy(np.stack(fps[i : i + batch_size])).requires_grad_(True)
        logits = model(X)
        target = (
            torch.full((len(logits),), class_idx, dtype=torch.long)
            if class_idx is not None
            else logits.argmax(dim=1)
        )
        loss = F.cross_entropy(logits, target, reduction="sum")
        loss.backward()
        norms.append(X.grad.norm(dim=1).detach().numpy())
    return np.concatenate(norms)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load training molecules ──────────────────────────────────────────────
    splits = pd.read_csv(DATA_SPLITS, sep="\t")
    train_smiles = splits[splits["Fold"] != TEST_FOLD]["SMILES"].dropna().tolist()
    print(f"Training folds ({[f for f in range(5) if f != TEST_FOLD]}): {len(train_smiles)} molecules")

    fps, valid = featurize(train_smiles)
    print(f"Valid SMILES: {len(valid)}")

    model1 = load_model(1)
    model2 = load_model(2)

    # ── Task 1: binary classifier ────────────────────────────────────────────
    print("\nTask 1 — gradient norm (predicted class as target)...")
    norms1 = gradient_norms(model1, fps)
    top = np.argsort(norms1)[:TOP_TASK1]
    task1 = [valid[i] for i in top]

    out1 = OUT_DIR / "task1_submission.csv"
    pd.DataFrame({"smiles": task1}).to_csv(out1, index=False)
    print(f"  norm range: {norms1[top[0]]:.4f} – {norms1[top[-1]]:.4f}")
    print(f"  saved {len(task1)} → {out1.relative_to(ROOT)}")

    # ── Task 2: 4-class classifier ───────────────────────────────────────────
    print("\nTask 2 — per-class gradient norm...")
    per_class_norms = np.zeros((len(fps), N_CLASSES))
    for c in range(N_CLASSES):
        per_class_norms[:, c] = gradient_norms(model2, fps, class_idx=c)
        print(f"  class_{c}: min norm = {per_class_norms[:, c].min():.4f}")

    # Assign each molecule to the class for which its gradient is smallest
    assigned = np.argmin(per_class_norms, axis=1)
    task2 = []
    for c in range(N_CLASSES):
        mask = assigned == c
        class_norms = np.where(mask, per_class_norms[:, c], np.inf)
        top_c = np.argsort(class_norms)[:TOP_PER_CLASS]
        mols = [valid[i] for i in top_c if mask[i]][:TOP_PER_CLASS]
        task2.extend(mols)
        print(f"  class_{c}: {mask.sum()} assigned → {len(mols)} selected")

    out2 = OUT_DIR / "task2_submission.csv"
    pd.DataFrame({"smiles": task2}).to_csv(out2, index=False)
    print(f"  saved {len(task2)} → {out2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

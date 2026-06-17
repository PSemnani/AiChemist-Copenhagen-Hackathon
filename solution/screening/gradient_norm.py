"""
Gradient-norm based membership inference for training data reconstruction.

Two submissions:

  Submission 1 — Fold-based + gradient norm (cross-entropy)
    Uses data/carbide/data_splits.tsv Fold column.
    Assumes fold 0 = held-out test set (seed=0 convention).
    Training molecules = folds 1-4 (~1588 molecules).
    Ranks by smallest gradient norm → model is at a loss minimum there.

  Submission 2 — CARBIDE + hERG combined, logit gradient norm
    Avoids softmax saturation by differentiating the raw logit instead
    of the cross-entropy loss. Works even when all logits are very large.

Why gradient norm:
    At a memorized training point the loss (or logit) is at a local
    minimum → gradient ≈ 0. Non-training molecules are not at any
    minimum, so their gradient is larger.

Usage:
    pixi run python solution/screening/gradient_norm.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).parent.parent.parent
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def load_model(task_num: int):
    spec = importlib.util.spec_from_file_location(
        f"task{task_num}_model",
        ROOT / "tasks" / f"task-{task_num}" / "model" / "model.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.load_model(
        str(ROOT / "tasks" / f"task-{task_num}" / "model" / "model.safetensors")
    )
    model.eval()
    return model


def featurize(smiles_list: list[str]) -> tuple[list[np.ndarray], list[str]]:
    fps, valid = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            fps.append(_gen.GetFingerprintAsNumPy(mol).astype("float32"))
            valid.append(smi)
    return fps, valid


def loss_grad_norms(model, fps: list[np.ndarray], batch_size: int = 256) -> np.ndarray:
    """
    Gradient of cross-entropy loss w.r.t. input fingerprint.
    Works well when logits are moderate (not saturated).
    Small norm → model is at a loss minimum → likely memorized.
    """
    norms = []
    for i in range(0, len(fps), batch_size):
        X = torch.from_numpy(np.stack(fps[i : i + batch_size])).requires_grad_(True)
        logits = model(X)
        pred = logits.argmax(dim=1)
        loss = F.cross_entropy(logits, pred, reduction="sum")
        loss.backward()
        norms.append(X.grad.norm(dim=1).detach().numpy())
    return np.concatenate(norms)


def logit_grad_norms(
    model, fps: list[np.ndarray], class_idx: int = None, batch_size: int = 512
) -> np.ndarray:
    """
    Gradient of the raw logit w.r.t. input fingerprint — no softmax involved.
    Use this when logits are very large (softmax-based gradients collapse to 0).
    Small norm → model is at a logit peak → likely memorized.
    """
    norms = []
    for i in range(0, len(fps), batch_size):
        X = torch.from_numpy(np.stack(fps[i : i + batch_size])).requires_grad_(True)
        logits = model(X)
        target = (
            logits[:, class_idx].sum()
            if class_idx is not None
            else logits.max(dim=1).values.sum()
        )
        target.backward()
        norms.append(X.grad.norm(dim=1).detach().numpy())
    return np.concatenate(norms)


def select_per_class(
    valid: list[str],
    per_class_norms: np.ndarray,
    per_class: int = 250,
) -> list[str]:
    """Assign each molecule to the class with smallest gradient norm, pick top-N per class."""
    n_classes = per_class_norms.shape[1]
    assigned = np.argmin(per_class_norms, axis=1)
    results = []
    for c in range(n_classes):
        mask = assigned == c
        class_norms = np.where(mask, per_class_norms[:, c], np.inf)
        top_c = np.argsort(class_norms)[:per_class]
        mols = [valid[i] for i in top_c if mask[i]][:per_class]
        results.extend(mols)
        print(f"  class_{c}: {mask.sum():,} assigned | min norm={per_class_norms[top_c[0], c]:.4f} | {len(mols)} collected")
    return results


def submission1_fold_based(model1, model2, out_dir: Path):
    """Submission 1: CARBIDE training folds (1-4) + cross-entropy gradient norm."""
    print("=== Submission 1: Fold-based (folds 1-4) + loss gradient norm ===")
    splits = pd.read_csv(ROOT / "data" / "carbide" / "data_splits.tsv", sep="\t")
    train_smiles = splits[splits["Fold"] != 0]["SMILES"].dropna().tolist()
    print(f"Training folds 1-4: {len(train_smiles)} molecules")

    fps, valid = featurize(train_smiles)
    print(f"Valid: {len(valid)}")

    # Task 1
    norms_t1 = loss_grad_norms(model1, fps)
    top1k = np.argsort(norms_t1)[:1000]
    task1 = [valid[i] for i in top1k]
    pd.DataFrame({"smiles": task1}).to_csv(out_dir / "submission1_task1_fold_gradnorm.csv", index=False)
    print(f"Task1: norm range {norms_t1[top1k[0]]:.4f} – {norms_t1[top1k[-1]]:.4f} | saved {len(task1)}")

    # Task 2
    per_class_norms = np.column_stack([
        loss_grad_norms(
            model2,
            [np.full(2048, 0, dtype="float32")] * len(fps),  # placeholder
        )
        for _ in range(4)
    ])
    # compute properly per class
    per_class_norms = np.zeros((len(fps), 4))
    for c in range(4):
        X = torch.from_numpy(np.stack(fps)).requires_grad_(True)
        loss = F.cross_entropy(model2(X), torch.full((len(fps),), c, dtype=torch.long), reduction="sum")
        loss.backward()
        per_class_norms[:, c] = X.grad.norm(dim=1).detach().numpy()

    task2 = select_per_class(valid, per_class_norms)
    pd.DataFrame({"smiles": task2}).to_csv(out_dir / "submission1_task2_fold_gradnorm.csv", index=False)
    print(f"Task2: saved {len(task2)}")


def submission2_combined(model1, model2, out_dir: Path):
    """Submission 2: CARBIDE + hERG combined, raw logit gradient norm (avoids saturation)."""
    print("=== Submission 2: CARBIDE + hERG combined, logit gradient norm ===")
    carbide = pd.read_csv(ROOT / "data" / "carbide" / "smiles_mapping.tsv", sep="\t")["SMILES"].dropna().tolist()
    herg = pd.read_csv(ROOT / "data" / "hERG_data.csv")["SMILES"].dropna().tolist()
    all_smi = list(dict.fromkeys(carbide + herg))
    print(f"Combined: {len(all_smi):,} molecules")

    fps, valid = featurize(all_smi)
    print(f"Valid: {len(valid):,}")

    # Task 1 — gradient of max logit
    norms_t1 = logit_grad_norms(model1, fps, class_idx=None)
    top1k = np.argsort(norms_t1)[:1000]
    task1 = [valid[i] for i in top1k]
    pd.DataFrame({"smiles": task1}).to_csv(out_dir / "submission2_task1_combined_gradnorm.csv", index=False)
    print(f"Task1: norm range {norms_t1[top1k[0]]:.4f} – {norms_t1[top1k[-1]]:.4f} | saved {len(task1)}")

    # Task 2 — gradient of each class logit
    per_class_norms = np.zeros((len(fps), 4))
    for c in range(4):
        per_class_norms[:, c] = logit_grad_norms(model2, fps, class_idx=c)

    task2 = select_per_class(valid, per_class_norms)
    pd.DataFrame({"smiles": task2}).to_csv(out_dir / "submission2_task2_combined_gradnorm.csv", index=False)
    print(f"Task2: saved {len(task2)}")


if __name__ == "__main__":
    out_dir = ROOT / "results" / "best"
    out_dir.mkdir(parents=True, exist_ok=True)

    model1 = load_model(1)
    model2 = load_model(2)

    submission1_fold_based(model1, model2, out_dir)
    print()
    submission2_combined(model1, model2, out_dir)

    print("\nDone. Files in results/best/:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size // 1024}KB)")

#!/usr/bin/env python
"""
Screen all ChEMBL molecules through Task 1 and Task 2 models.
Ranks by RAW LOGIT (not softmax) for better membership discrimination.
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).parent.parent
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def load_model(task_num):
    spec = importlib.util.spec_from_file_location(
        f"task{task_num}_model",
        ROOT / "tasks" / f"task-{task_num}" / "model" / "model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.load_model(str(ROOT / "tasks" / f"task-{task_num}" / "model" / "model.safetensors"))
    model.eval()
    return model

def ecfp4(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return _gen.GetFingerprintAsNumPy(mol).astype(np.float32)

def screen(smiles_list, model, batch_size=2048):
    """Returns raw logits (before softmax) for all valid SMILES."""
    results = []  # (smiles, logit_class1, logit_class2, ...) tuples
    fps, valid_smiles = [], []
    for smi in smiles_list:
        fp = ecfp4(smi)
        if fp is not None:
            fps.append(fp)
            valid_smiles.append(smi)
    
    all_logits = []
    for i in range(0, len(fps), batch_size):
        batch = torch.from_numpy(np.stack(fps[i:i+batch_size]))
        with torch.no_grad():
            logits = model(batch).cpu().numpy()
        all_logits.append(logits)
        if i % 200000 == 0 and i > 0:
            print(f"  {i:,} / {len(fps):,} processed...")
    
    all_logits = np.concatenate(all_logits, axis=0)
    return valid_smiles, all_logits

def main():
    print("Loading ChEMBL...")
    smiles = pd.read_csv(ROOT / "data" / "chembl_smiles.tsv", sep="\t")["SMILES"].dropna().tolist()
    print(f"  {len(smiles):,} molecules loaded")

    # ── Task 1 ──────────────────────────────────────────────────────────────
    print("\nScreening Task 1 (binary classifier)...")
    model1 = load_model(1)
    valid_smi, logits1 = screen(smiles, model1)
    print(f"  {len(valid_smi):,} valid SMILES")

    # rank by class 1 raw logit (not softmax — more dynamic range)
    class1_logit = logits1[:, 1]
    top_idx = np.argsort(class1_logit)[::-1][:1000]
    top1 = [valid_smi[i] for i in top_idx]
    pd.DataFrame({"smiles": top1}).to_csv(ROOT / "results" / "task1_submission.csv", index=False)
    print(f"  Saved top-1000 to results/task1_submission.csv")
    print(f"  Logit range (top-1000): {class1_logit[top_idx[-1]]:.2f} – {class1_logit[top_idx[0]]:.2f}")
    print(f"  Top 3 SMILES:")
    for i in top_idx[:3]:
        print(f"    [{class1_logit[i]:.3f}] {valid_smi[i]}")

    # ── Task 2 ──────────────────────────────────────────────────────────────
    print("\nScreening Task 2 (4-class classifier)...")
    model2 = load_model(2)
    valid_smi2, logits2 = screen(smiles, model2)

    all_results = []
    for c in range(4):
        class_logit = logits2[:, c]
        top_idx2 = np.argsort(class_logit)[::-1][:250]
        top_c = [valid_smi2[i] for i in top_idx2]
        all_results.extend(top_c)
        print(f"  class_{c}: logit range {class_logit[top_idx2[-1]]:.2f} – {class_logit[top_idx2[0]]:.2f}")

    pd.DataFrame({"smiles": all_results}).to_csv(ROOT / "results" / "task2_submission.csv", index=False)
    print(f"  Saved 1000 molecules (250/class) to results/task2_submission.csv")

if __name__ == "__main__":
    main()

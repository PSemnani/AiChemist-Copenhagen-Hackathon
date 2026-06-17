"""
Track A: Gradient descent on model input to find memorized ECFP4 fingerprints,
then match to nearest ChEMBL molecule via Tanimoto similarity.
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from rdkit.Chem import AllChem

ROOT = Path(__file__).parent.parent
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def load_model(task_num):
    spec = importlib.util.spec_from_file_location(
        f"task{task_num}_model",
        ROOT / "tasks" / f"task-{task_num}" / "model" / "model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.load_model(str(ROOT / "tasks" / f"task-{task_num}" / "model" / "model.safetensors"))
    m.eval()
    return m

def optimize_fingerprints(model, class_idx, n_starts=300, steps=500, sharpness=8.0):
    """Gradient descent to find ECFP4 bit vectors that maximize class logit."""
    logit_vars = torch.randn(n_starts, 2048, requires_grad=True)
    optimizer = torch.optim.Adam([logit_vars], lr=0.05)
    for step in range(steps):
        x = torch.sigmoid(logit_vars * sharpness)
        loss = -model(x)[:, class_idx].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        fps = (torch.sigmoid(logit_vars * sharpness) > 0.5).numpy().astype(np.uint8)
    # deduplicate
    unique = {tuple(f) for f in fps}
    print(f"  class_{class_idx}: {len(unique)} unique fingerprints from {n_starts} starts")
    return [np.array(f) for f in unique]

def fp_to_rdkit(arr):
    bv = DataStructs.ExplicitBitVect(2048)
    for i, b in enumerate(arr):
        if b:
            bv.SetBit(i)
    return bv

def build_chembl_index(smiles_path, batch_size=50000):
    """Load ChEMBL SMILES and compute ECFP4 as RDKit bit vectors."""
    print("Building ChEMBL fingerprint index...")
    df = pd.read_csv(smiles_path, sep="\t")["SMILES"].dropna()
    valid_smi, fps = [], []
    for i, smi in enumerate(df):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = _gen.GetFingerprint(mol)
            fps.append(fp)
            valid_smi.append(smi)
        if i % 500000 == 0 and i > 0:
            print(f"  {i:,} processed...")
    print(f"  Index built: {len(fps):,} molecules")
    return valid_smi, fps

def find_nearest(query_fp, chembl_fps, chembl_smi, top_k=5):
    """Find top-k ChEMBL molecules by Tanimoto similarity to query fingerprint."""
    q = fp_to_rdkit(query_fp)
    sims = DataStructs.BulkTanimotoSimilarity(q, chembl_fps)
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(chembl_smi[i], sims[i]) for i in top_idx]

def main():
    print("=== Building ChEMBL index (one-time) ===")
    chembl_smi, chembl_fps = build_chembl_index(ROOT / "data" / "chembl_smiles.tsv")

    # ── Task 1 ──────────────────────────────────────────────────────────────
    print("\n=== Task 1: Binary classifier ===")
    model1 = load_model(1)
    candidates = {}  # smiles → best_sim

    for cls in [0, 1]:
        print(f"Optimizing class {cls}...")
        opt_fps = optimize_fingerprints(model1, class_idx=cls, n_starts=300, steps=500)
        for fp in opt_fps:
            hits = find_nearest(fp, chembl_fps, chembl_smi, top_k=3)
            for smi, sim in hits:
                if smi not in candidates or candidates[smi] < sim:
                    candidates[smi] = sim

    # Score all candidates by logit and rank
    smi_list = list(candidates.keys())
    print(f"  Scoring {len(smi_list)} candidates...")
    fps_arr, valid_smi = [], []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps_arr.append(_gen.GetFingerprintAsNumPy(mol).astype(np.float32))
            valid_smi.append(smi)
    X = torch.from_numpy(np.stack(fps_arr))
    with torch.no_grad():
        logits = model1(X).numpy()
    best_logit = logits.max(axis=1)
    ranked = sorted(zip(best_logit, valid_smi), reverse=True)[:1000]
    top1000 = [s for _, s in ranked]
    pd.DataFrame({"smiles": top1000}).to_csv(ROOT / "results" / "task1_inversion.csv", index=False)
    print(f"  Saved {len(top1000)} to results/task1_grad_tanimoto.csv")

    # ── Task 2 ──────────────────────────────────────────────────────────────
    print("\n=== Task 2: 4-class classifier ===")
    model2 = load_model(2)
    all_results = []

    for cls in range(4):
        print(f"Optimizing class_{cls}...")
        candidates2 = {}
        opt_fps = optimize_fingerprints(model2, class_idx=cls, n_starts=200, steps=500)
        for fp in opt_fps:
            hits = find_nearest(fp, chembl_fps, chembl_smi, top_k=3)
            for smi, sim in hits:
                if smi not in candidates2 or candidates2[smi] < sim:
                    candidates2[smi] = sim

        smi_list2 = list(candidates2.keys())
        fps_arr2, valid_smi2 = [], []
        for smi in smi_list2:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fps_arr2.append(_gen.GetFingerprintAsNumPy(mol).astype(np.float32))
                valid_smi2.append(smi)
        X2 = torch.from_numpy(np.stack(fps_arr2))
        with torch.no_grad():
            logits2 = model2(X2).numpy()
        class_logit = logits2[:, cls]
        top_idx = np.argsort(class_logit)[::-1][:250]
        top_class = [valid_smi2[i] for i in top_idx]
        all_results.extend(top_class)
        print(f"  class_{cls}: {len(top_class)} molecules collected")

    pd.DataFrame({"smiles": all_results}).to_csv(ROOT / "results" / "task2_inversion.csv", index=False)
    print(f"  Saved {len(all_results)} to results/task2_grad_tanimoto.csv")

if __name__ == "__main__":
    main()

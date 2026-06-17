from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_ecfp4(smiles: str):
    """SMILES -> (2048,) float32 ECFP4 array, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _gen.GetFingerprintAsNumPy(mol).astype(np.float32)


def _featurize(smiles_list: List[str]) -> Tuple[List[np.ndarray], List[int]]:
    """Returns (valid fingerprint arrays, their original indices). Invalid SMILES skipped."""
    fps, idx = [], []
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_ecfp4(smi)
        if fp is not None:
            fps.append(fp)
            idx.append(i)
    return fps, idx


def score_max_confidence(smiles_list: List[str], task_model: nn.Module) -> np.ndarray:
    """Task 1 scorer: max softmax probability across both classes. Shape: (n,)."""
    scores = np.zeros(len(smiles_list), dtype=np.float32)
    fps, idx = _featurize(smiles_list)
    if fps:
        X = torch.from_numpy(np.stack(fps))
        with torch.no_grad():
            probs = task_model(X).softmax(-1)
        scores[idx] = probs.max(dim=-1).values.cpu().numpy()
    return scores


def score_class(smiles_list: List[str], task_model: nn.Module, class_idx: int) -> np.ndarray:
    """Task 2 scorer: softmax probability for a specific class. Shape: (n,)."""
    scores = np.zeros(len(smiles_list), dtype=np.float32)
    fps, idx = _featurize(smiles_list)
    if fps:
        X = torch.from_numpy(np.stack(fps))
        with torch.no_grad():
            probs = task_model(X).softmax(-1)
        scores[idx] = probs[:, class_idx].cpu().numpy()
    return scores

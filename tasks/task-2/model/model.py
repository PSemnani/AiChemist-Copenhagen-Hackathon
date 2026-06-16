"""Shareable, self-contained definition for the Task 2 4-class classifier.

Recipients need only this file + model.safetensors + (torch, rdkit,
safetensors, numpy). Output: 4 logits over CLASSES (anonymized labels).
Input: ECFP4 (Morgan, radius 2, 2048 bits) float32 0/1 vector.
"""

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

RADIUS = 2
N_BITS = 2048
CLASSES = ["class_0", "class_1", "class_2", "class_3"]
DROPOUT = 0.3

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)


def smiles_to_ecfp4(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _gen.GetFingerprintAsNumPy(mol).astype(np.float32)


def featurize_smiles(smiles_iterable):
    """List of SMILES -> (n, N_BITS) float32 tensor.

    WARNING: unparseable SMILES are silently dropped, so the output can have
    FEWER rows than the input. Do not zip the result back to your original
    SMILES list by position -- the indices will be misaligned. To keep
    alignment for ranking/submission, featurize one SMILES at a time with
    smiles_to_ecfp4 and skip the Nones yourself, tracking which survived.
    """
    rows = [fp for fp in (smiles_to_ecfp4(s) for s in smiles_iterable) if fp is not None]
    return torch.from_numpy(np.asarray(rows, dtype=np.float32))


class MLP(nn.Module):
    """Regularized MLP: Linear -> BatchNorm -> ReLU -> Dropout, x2 -> 4 logits."""

    def __init__(self, n_in=N_BITS, n_out=len(CLASSES), p=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, n_out),
        )

    def forward(self, x):
        return self.net(x)


def load_model(weights_path="model.safetensors"):
    from safetensors.torch import load_file

    model = MLP()
    model.load_state_dict(load_file(weights_path))
    model.eval()
    return model

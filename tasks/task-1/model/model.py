"""Shareable model definition for the Task 1 binary classifier.

This is the only code recipients need (plus the .safetensors weights and
rdkit/torch/safetensors installed) to load and run the model. The training
data is NOT included here -- recovering it is the exercise.

Input representation: ECFP4 == Morgan fingerprint, radius 2, 2048 bits, as a
float32 vector of 0/1. Output: 2 logits (class 0 / class 1), softmax +
cross-entropy at train time.
"""

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

RADIUS = 2
N_BITS = 2048

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)


def smiles_to_ecfp4(smiles):
    """SMILES string -> (N_BITS,) float32 ECFP4 vector, or None if unparseable."""
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
    """Plain fully-connected net: 2048 -> 512 -> 128 -> 2. No dropout / batch-norm."""

    def __init__(self, n_in=N_BITS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


def load_model(weights_path="model.safetensors"):
    """Instantiate MLP and load weights from a .safetensors file (safe, no pickle)."""
    from safetensors.torch import load_file

    model = MLP()
    model.load_state_dict(load_file(weights_path))
    model.eval()
    return model

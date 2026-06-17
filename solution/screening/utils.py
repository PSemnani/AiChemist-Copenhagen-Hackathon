"""Shared utilities: model loading and ECFP4 featurization."""
import importlib.util
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).parent.parent.parent
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def load_model(task_num: int) -> nn.Module:
    """Load a task model from tasks/task-{task_num}/model/."""
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
    """Convert SMILES to ECFP4 fingerprints. Invalid SMILES are silently dropped."""
    fps, valid = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fps.append(_gen.GetFingerprintAsNumPy(mol).astype("float32"))
            valid.append(smi)
    return fps, valid

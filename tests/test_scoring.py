import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tasks" / "task-1" / "model"))
from model import load_model as load_task1_model

from solution.scoring import _featurize, score_max_confidence, score_class

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"   # valid drug-like molecule
INVALID = "not_a_smiles!!!"


@pytest.fixture(scope="module")
def task1_model():
    model = load_task1_model(str(ROOT / "tasks" / "task-1" / "model" / "model.safetensors"))
    model.eval()
    return model


@pytest.fixture(scope="module")
def task2_model():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "task2_model_module",
        ROOT / "tasks" / "task-2" / "model" / "model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.load_model(str(ROOT / "tasks" / "task-2" / "model" / "model.safetensors"))
    model.eval()
    return model


def test_featurize_filters_invalid():
    fps, idx = _featurize([ASPIRIN, INVALID, ASPIRIN])
    assert len(fps) == 2
    assert idx == [0, 2]


def test_featurize_all_invalid():
    fps, idx = _featurize([INVALID, INVALID])
    assert fps == []
    assert idx == []


def test_score_max_confidence_shape(task1_model):
    scores = score_max_confidence([ASPIRIN, INVALID], task1_model)
    assert scores.shape == (2,)
    assert scores.dtype == np.float32


def test_score_max_confidence_invalid_is_zero(task1_model):
    scores = score_max_confidence([INVALID], task1_model)
    assert scores[0] == pytest.approx(0.0)


def test_score_max_confidence_valid_in_range(task1_model):
    scores = score_max_confidence([ASPIRIN], task1_model)
    assert 0.0 <= scores[0] <= 1.0


def test_score_class_shape(task2_model):
    scores = score_class([ASPIRIN, INVALID], task2_model, class_idx=0)
    assert scores.shape == (2,)


def test_score_class_invalid_is_zero(task2_model):
    scores = score_class([INVALID], task2_model, class_idx=1)
    assert scores[0] == pytest.approx(0.0)


def test_score_class_sums_to_one_across_classes(task2_model):
    # Probabilities across all 4 classes must sum to ~1 for valid SMILES
    total = sum(score_class([ASPIRIN], task2_model, c)[0] for c in range(4))
    assert total == pytest.approx(1.0, abs=1e-5)

import pytest
import torch
from solution.prior import tokenize_smiles, Vocabulary


# Local vocab fixture so these tests run without needing RNN/Prior (Task 3).
# Uses actual REINVENT4-style special tokens: $ (pad), ^ (BOS), # (EOS).
# Confirmed from checkpoint inspection: eos_token=2 -> '#'
@pytest.fixture
def vocab():
    tokens = ["$", "^", "#", "C", "c", "N", "O", "(", ")", "=", "1", "2", "Br", "Cl", "F"]
    return Vocabulary(tokens, pad="$", start="^", end="#")


def test_tokenize_smiles_simple():
    assert tokenize_smiles("CCO") == ["C", "C", "O"]


def test_tokenize_smiles_multichar():
    # Cl and Br must NOT be split into C+l or B+r
    assert tokenize_smiles("ClCBr") == ["Cl", "C", "Br"]


def test_tokenize_smiles_brackets():
    assert "[NH]" in tokenize_smiles("c1cc[NH]cc1")


def test_vocabulary_len(vocab):
    assert len(vocab) >= 5   # at minimum: PAD, GO, EOS + some atoms


def test_vocabulary_encode_includes_start_end(vocab):
    indices = vocab.encode("CCO")
    assert indices is not None
    assert indices[0] == vocab.start_idx
    assert indices[-1] == vocab.end_idx


def test_vocabulary_encode_unknown_token_returns_none(vocab):
    # 'X' is not in our tiny vocab
    assert vocab.encode("X") is None


def test_vocabulary_decode_strips_special(vocab):
    indices = vocab.encode("CCO")
    result = vocab.decode(indices)
    assert result == "CCO"


def test_vocabulary_roundtrip(vocab):
    for smi in ["C", "CCO", "C(=O)O"]:
        indices = vocab.encode(smi)
        if indices is not None:
            assert vocab.decode(indices) == smi

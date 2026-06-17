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


# ── Task 3: RNN + Prior tests ─────────────────────────────────────────────────
from solution.prior import RNN, Prior, get_device


@pytest.fixture
def tiny_rnn(vocab):
    return RNN(vocab_size=len(vocab), embed_dim=8, hidden_size=16, num_layers=1)


@pytest.fixture
def mock_prior(vocab, tiny_rnn):
    rnn = tiny_rnn
    rnn.eval()
    for p in rnn.parameters():
        p.requires_grad_(False)
    return Prior(rnn, vocab, torch.device("cpu"))


@pytest.fixture
def mock_agent(vocab):
    rnn = RNN(vocab_size=len(vocab), embed_dim=8, hidden_size=16, num_layers=1)
    rnn.train()
    for p in rnn.parameters():
        p.requires_grad_(True)
    return Prior(rnn, vocab, torch.device("cpu"))


def test_rnn_output_shape(tiny_rnn, vocab):
    batch, seq = 4, 10
    x = torch.randint(0, len(vocab), (batch, seq))
    logits, hidden = tiny_rnn(x)
    assert logits.shape == (batch, seq, len(vocab))
    assert hidden[0].shape[1] == batch   # LSTM hidden state batch dim


def test_prior_sample_returns_n_strings(mock_prior):
    result = mock_prior.sample(n=5)
    assert len(result) == 5
    assert all(isinstance(s, str) for s in result)


def test_prior_log_prob_shape(mock_prior):
    smiles = ["CCO", "CC", "invalid!!!xyz"]
    lp = mock_prior.log_prob(smiles)
    assert lp.shape == (3,)
    assert lp.dtype == torch.float32


def test_prior_log_prob_invalid_smiles_gets_zero(mock_prior):
    lp = mock_prior.log_prob(["invalid!!!xyz"])
    assert lp[0].item() == pytest.approx(0.0)


def test_prior_log_prob_is_differentiable(mock_agent):
    smiles = mock_agent.sample(n=3)
    lp = mock_agent.log_prob(smiles)
    loss = -lp.mean()
    loss.backward()
    grads = [p.grad for p in mock_agent.rnn.parameters() if p.grad is not None]
    assert len(grads) > 0


# ── Task 6: Explore — gradient attribution ────────────────────────────────────
import importlib.util as _ilu
from pathlib import Path as _Path

def _load_task2_model():
    _ROOT = _Path(__file__).parent.parent
    spec = _ilu.spec_from_file_location(
        "task2_m",
        _ROOT / "tasks" / "task-2" / "model" / "model.py"
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.load_model(str(_ROOT / "tasks" / "task-2" / "model" / "model.safetensors"))
    model.eval()
    return model

def test_top_bits_for_class_returns_correct_length():
    from solution.explore import top_bits_for_class
    model = _load_task2_model()
    bits = top_bits_for_class(model, class_idx=0, topk=20)
    assert isinstance(bits, list)
    assert len(bits) == 20
    assert all(isinstance(b, int) for b in bits)
    assert all(0 <= b < 2048 for b in bits)

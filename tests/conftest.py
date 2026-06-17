import pytest
import torch
from copy import deepcopy
from solution.prior import Vocabulary, RNN, Prior


@pytest.fixture
def vocab():
    tokens = ["PAD", "GO", "EOS", "C", "c", "N", "O", "(", ")", "=", "1", "2", "Br", "Cl", "F"]
    return Vocabulary(tokens)


@pytest.fixture
def tiny_rnn(vocab):
    return RNN(vocab_size=len(vocab), embed_dim=16, hidden_size=32, num_layers=1)


@pytest.fixture
def mock_prior(tiny_rnn, vocab):
    rnn = deepcopy(tiny_rnn)
    rnn.eval()
    for p in rnn.parameters():
        p.requires_grad_(False)
    return Prior(rnn, vocab, torch.device("cpu"))


@pytest.fixture
def mock_agent(tiny_rnn, vocab):
    rnn = deepcopy(tiny_rnn)
    rnn.train()
    for p in rnn.parameters():
        p.requires_grad_(True)
    return Prior(rnn, vocab, torch.device("cpu"))

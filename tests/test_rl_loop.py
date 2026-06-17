# tests/test_rl_loop.py
import numpy as np
import pytest
import torch
from solution.rl_loop import run_rl


def always_zero(smiles_list):
    return np.zeros(len(smiles_list), dtype=np.float32)


def always_half(smiles_list):
    return np.full(len(smiles_list), 0.5, dtype=np.float32)


def test_run_rl_returns_list(mock_prior, mock_agent):
    result = run_rl(mock_prior, mock_agent, always_half, steps=2, batch_size=4)
    assert isinstance(result, list)


def test_run_rl_excludes_zero_score_molecules(mock_prior, mock_agent):
    result = run_rl(mock_prior, mock_agent, always_zero, steps=2, batch_size=4)
    assert result == []


def test_run_rl_sorted_descending(mock_prior, mock_agent):
    def len_scorer(smiles_list):
        return np.array([min(len(s) / 20, 1.0) for s in smiles_list], dtype=np.float32)

    result = run_rl(mock_prior, mock_agent, len_scorer, steps=3, batch_size=8)
    lengths = [len(s) for s in result]
    assert lengths == sorted(lengths, reverse=True)


def test_run_rl_deduplicates(mock_prior, mock_agent):
    result = run_rl(mock_prior, mock_agent, always_half, steps=5, batch_size=16)
    assert len(result) == len(set(result))


def test_run_rl_agent_parameters_update(mock_prior, mock_agent):
    params_before = [p.clone() for p in mock_agent.rnn.parameters()]
    run_rl(mock_prior, mock_agent, always_half, steps=3, batch_size=4, lr=1e-2)
    params_after = list(mock_agent.rnn.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
    assert changed, "Agent parameters should change after RL steps"

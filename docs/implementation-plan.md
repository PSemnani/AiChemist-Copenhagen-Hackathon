# REINVENT RL Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-phase molecular generation pipeline that uses a pre-trained REINVENT LSTM prior + REINFORCE to reconstruct training molecules for two molecular classifiers.

**Architecture:** A frozen REINVENT prior (SMILES LSTM) acts as a KL anchor while a trainable agent copy is steered by target classifier confidence via the REINFORCE loss `((prior_lp + σ·score) − agent_lp)²`. Phase 1 (`explore.py`) validates signal quality; Phase 2 runs one RL agent per task (four agents for Task 2, one per biological class).

**Tech Stack:** PyTorch, RDKit, safetensors, pixi (Python 3.14), pytest

---

## File Map

| File | Responsibility |
|---|---|
| `pixi.toml` | Add pytest dependency |
| `solution/__init__.py` | Package marker |
| `solution/prior.py` | SMILES tokenizer, Vocabulary, RNN, Prior class, `load_prior`, `get_device` |
| `solution/scoring.py` | `_featurize`, `score_max_confidence` (Task 1), `score_class` (Task 2) |
| `solution/rl_loop.py` | `run_rl` — REINFORCE training loop |
| `solution/explore.py` | Phase 1: prior baseline + gradient attribution + RL probe |
| `solution/run_task1.py` | Phase 2 entry point for Task 1 (1 agent) |
| `solution/run_task2.py` | Phase 2 entry point for Task 2 (4 agents × 250 molecules) |
| `tests/__init__.py` | Package marker |
| `tests/conftest.py` | Shared pytest fixtures (tiny mock prior, mock agent) |
| `tests/test_prior.py` | Tests for tokenizer, Vocabulary, RNN, Prior |
| `tests/test_scoring.py` | Tests for `_featurize`, `score_max_confidence`, `score_class` |
| `tests/test_rl_loop.py` | Tests for `run_rl` |
| `models/reinvent.prior` | Downloaded checkpoint (~12 MB) |
| `results/task1_submission.csv` | Output: 1000 SMILES for Task 1 |
| `results/task2_submission.csv` | Output: 1000 SMILES for Task 2 (250 per class) |

---

## Task 1: Scaffold + Download Prior

**Files:**
- Modify: `pixi.toml`
- Create: `solution/__init__.py`, `tests/__init__.py`, `tests/conftest.py`
- Create dirs: `solution/`, `tests/`, `models/`, `results/`

- [ ] **Step 1: Add pytest to pixi.toml**

Replace the `[dependencies]` section in `pixi.toml` with:

```toml
[dependencies]
python = ">=3.14.5,<3.15"
pandas = ">=3.0.3,<4"
matplotlib = ">=3.10.9,<4"
numpy = ">=2.4.6,<3"
requests = ">=2.34.2,<3"
rdkit = ">=2026.3.3,<2027"
pytorch = ">=2.11.0,<3"
safetensors = ">=0.7.0,<0.8"
scikit-learn = ">=1.9.0,<2"
pytest = ">=8.0.0,<9"
```

- [ ] **Step 2: Create directories and package markers**

```bash
mkdir -p solution tests models results
touch solution/__init__.py tests/__init__.py
```

- [ ] **Step 3: Create `tests/conftest.py` with shared fixtures**

```python
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
```

- [ ] **Step 4: Download REINVENT4 prior**

```bash
# Download ChEMBL-trained prior from MolecularAI/REINVENT4 GitHub releases
# Visit: https://github.com/MolecularAI/REINVENT4/releases
# Download the file named 'reinvent.prior' and save to models/reinvent.prior
# Alternatively via curl (replace URL with actual release asset URL):
curl -L <release-url> -o models/reinvent.prior
```

- [ ] **Step 5: Inspect checkpoint format**

```bash
pixi run python -c "
import torch
ckpt = torch.load('models/reinvent.prior', map_location='cpu', weights_only=False)
print('Top-level keys:', list(ckpt.keys()))
model_data = ckpt.get('model', ckpt)
print('Model keys:', list(model_data.keys()))
vocab = model_data.get('vocabulary', None)
print('Vocabulary type:', type(vocab))
if isinstance(vocab, list):
    print('Vocab length:', len(vocab))
    print('First 10 tokens:', vocab[:10])
params = model_data.get('network_params', {})
print('Network params:', params)
"
```

Expected output (REINVENT4 format):
```
Top-level keys: ['model_type', 'version', 'model']
Model keys: ['vocabulary', 'network_params', 'network']
Vocabulary type: <class 'list'>
Vocab length: ~40
First 10 tokens: ['PAD', 'GO', 'EOS', 'C', 'c', 'N', 'O', ...]
Network params: {'num_layers': 3, 'layer_size': 512, 'embedding_layer_size': 256, ...}
```

If the format differs (e.g. vocabulary is not a list), adjust `load_prior` in Task 3 accordingly.

- [ ] **Step 6: Commit scaffold**

```bash
git add pixi.toml solution/ tests/ models/.gitkeep results/.gitkeep
git commit -m "feat: project scaffold, pytest, directory structure"
```

---

## Task 2: Prior — Tokenizer + Vocabulary (`solution/prior.py`, Part 1)

**Files:**
- Create: `solution/prior.py` (partial — tokenizer + Vocabulary only)
- Create: `tests/test_prior.py` (partial)

- [ ] **Step 1: Write failing tests for tokenizer and Vocabulary**

```python
# tests/test_prior.py
import pytest
import torch
from solution.prior import tokenize_smiles, Vocabulary


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
    assert indices[0] == vocab._t2i["GO"]
    assert indices[-1] == vocab._t2i["EOS"]


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_prior.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `solution/prior.py` does not exist yet.

- [ ] **Step 3: Implement tokenizer + Vocabulary in `solution/prior.py`**

```python
import re
from typing import List, Optional

import torch
import torch.nn as nn

# Handles: bracket atoms [NH], multi-char atoms Cl/Br, ring bonds %10, all SMILES chars
_SMILES_RE = re.compile(
    r"(\[[^\[\]]{1,10}\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

PAD, START, END = "PAD", "GO", "EOS"


def tokenize_smiles(smiles: str) -> List[str]:
    """SMILES string → list of tokens. Multi-char atoms (Cl, Br) kept intact."""
    return _SMILES_RE.findall(smiles)


class Vocabulary:
    def __init__(self, tokens: List[str]):
        special = [PAD, START, END]
        all_tokens = special + [t for t in tokens if t not in special]
        self._tokens = all_tokens
        self._t2i = {t: i for i, t in enumerate(all_tokens)}
        self._i2t = {i: t for i, t in enumerate(all_tokens)}

    def __len__(self) -> int:
        return len(self._tokens)

    def encode(self, smiles: str) -> Optional[List[int]]:
        """SMILES → [START, token..., END] indices. None if unknown token found."""
        toks = [START] + tokenize_smiles(smiles) + [END]
        try:
            return [self._t2i[t] for t in toks]
        except KeyError:
            return None

    def decode(self, indices: List[int]) -> str:
        """Token indices → SMILES string (START/END/PAD stripped)."""
        skip = {self._t2i.get(PAD), self._t2i.get(START), self._t2i.get(END)}
        return "".join(self._i2t[i] for i in indices if i not in skip and i in self._i2t)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pixi run pytest tests/test_prior.py::test_tokenize_smiles_simple \
    tests/test_prior.py::test_tokenize_smiles_multichar \
    tests/test_prior.py::test_tokenize_smiles_brackets \
    tests/test_prior.py::test_vocabulary_len \
    tests/test_prior.py::test_vocabulary_encode_includes_start_end \
    tests/test_prior.py::test_vocabulary_encode_unknown_token_returns_none \
    tests/test_prior.py::test_vocabulary_decode_strips_special \
    tests/test_prior.py::test_vocabulary_roundtrip -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add solution/prior.py tests/test_prior.py tests/conftest.py
git commit -m "feat: SMILES tokenizer and Vocabulary"
```

---

## Task 3: Prior — RNN + Prior Class + load_prior (`solution/prior.py`, Part 2)

**Files:**
- Modify: `solution/prior.py` (append RNN, Prior, get_device, load_prior)
- Modify: `tests/test_prior.py` (append RNN + Prior tests)

**Depends on:** Task 2

- [ ] **Step 1: Append failing RNN + Prior tests to `tests/test_prior.py`**

```python
# Append to tests/test_prior.py
from solution.prior import RNN, Prior, get_device
from copy import deepcopy


def test_rnn_output_shape(tiny_rnn, vocab):
    batch, seq = 4, 10
    x = torch.randint(0, len(vocab), (batch, seq))
    logits, hidden = tiny_rnn(x)
    assert logits.shape == (batch, seq, len(vocab))
    assert hidden[0].shape[1] == batch   # LSTM hidden state


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
    # "invalid!!!xyz" contains tokens not in tiny vocab → encode returns None → log_prob = 0
    lp = mock_prior.log_prob(["invalid!!!xyz"])
    assert lp[0].item() == pytest.approx(0.0)


def test_prior_log_prob_is_differentiable(mock_agent):
    smiles = mock_agent.sample(n=3)
    lp = mock_agent.log_prob(smiles)
    loss = -lp.mean()
    loss.backward()
    # At least one parameter must have a non-None gradient
    grads = [p.grad for p in mock_agent.rnn.parameters() if p.grad is not None]
    assert len(grads) > 0
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
pixi run pytest tests/test_prior.py::test_rnn_output_shape \
    tests/test_prior.py::test_prior_sample_returns_n_strings \
    tests/test_prior.py::test_prior_log_prob_shape -v
```

Expected: `ImportError` — `RNN`, `Prior` not yet defined.

- [ ] **Step 3: Append RNN, Prior, get_device, load_prior to `solution/prior.py`**

```python
# Append to solution/prior.py (after Vocabulary class)

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class RNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 hidden_size: int = 512, num_layers: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        """x: (batch, seq_len) → logits: (batch, seq_len, vocab_size), hidden"""
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        return self.linear(out), hidden


class Prior:
    def __init__(self, rnn: RNN, vocabulary: Vocabulary, device: torch.device):
        self.rnn = rnn
        self.vocabulary = vocabulary
        self.device = device
        self.rnn.to(device)

    def sample(self, n: int = 128, max_len: int = 100) -> List[str]:
        """Sample n SMILES. Uses torch.no_grad — not differentiable."""
        start_idx = self.vocabulary._t2i[START]
        end_idx = self.vocabulary._t2i[END]
        token = torch.full((n, 1), start_idx, dtype=torch.long, device=self.device)
        hidden = None
        sequences: List[List[int]] = [[] for _ in range(n)]
        done = [False] * n

        with torch.no_grad():
            for _ in range(max_len):
                logits, hidden = self.rnn(token, hidden)      # (n, 1, vocab_size)
                probs = logits[:, 0, :].softmax(-1)           # (n, vocab_size)
                next_tok = torch.multinomial(probs, 1)        # (n, 1)
                for i in range(n):
                    if done[i]:
                        continue
                    idx = next_tok[i, 0].item()
                    if idx == end_idx:
                        done[i] = True
                    else:
                        sequences[i].append(idx)
                token = next_tok
                if all(done):
                    break

        return [self.vocabulary.decode(seq) for seq in sequences]

    def log_prob(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Log-likelihood for each SMILES. Shape: (n,).
        Differentiable w.r.t. rnn parameters — used for REINFORCE gradient.
        Invalid / unencodable SMILES get log_prob = 0 (detached).
        """
        log_probs = []
        for smi in smiles_list:
            indices = self.vocabulary.encode(smi)
            if indices is None or len(indices) < 2:
                log_probs.append(torch.tensor(0.0, device=self.device))
                continue
            x = torch.tensor(indices[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
            y = torch.tensor(indices[1:], dtype=torch.long, device=self.device)
            logits, _ = self.rnn(x)               # (1, T, vocab_size)
            lp = logits[0].log_softmax(-1)         # (T, vocab_size)
            log_probs.append(lp[range(len(y)), y].sum())
        return torch.stack(log_probs)


def load_prior(path: str, freeze: bool = True) -> Prior:
    """
    Load REINVENT4 prior from .prior checkpoint.
    Checkpoint format (REINVENT4):
      {'model_type': ..., 'model': {'vocabulary': [...], 'network_params': {...}, 'network': {...}}}
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_data = ckpt.get("model", ckpt)         # handle both REINVENT4 and flat formats

    tokens: List[str] = model_data["vocabulary"]
    params = model_data.get("network_params", {})

    vocabulary = Vocabulary(tokens)
    rnn = RNN(
        vocab_size=len(vocabulary),
        embed_dim=params.get("embedding_layer_size", 256),
        hidden_size=params.get("layer_size", 512),
        num_layers=params.get("num_layers", 3),
    )
    rnn.load_state_dict(model_data["network"])

    if freeze:
        rnn.eval()
        for p in rnn.parameters():
            p.requires_grad_(False)

    return Prior(rnn, vocabulary, get_device())
```

- [ ] **Step 4: Run all prior tests — expect pass**

```bash
pixi run pytest tests/test_prior.py -v
```

Expected: all 13 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add solution/prior.py tests/test_prior.py
git commit -m "feat: RNN, Prior (sample + log_prob), load_prior"
```

---

## Task 4: Scoring (`solution/scoring.py`)

**Files:**
- Create: `solution/scoring.py`
- Create: `tests/test_scoring.py`

**Depends on:** Task 1 (task model files exist at `tasks/`)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scoring.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_scoring.py -v
```

Expected: `ModuleNotFoundError: No module named 'solution.scoring'`

- [ ] **Step 3: Implement `solution/scoring.py`**

```python
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_ecfp4(smiles: str):
    """SMILES → (2048,) float32 ECFP4 array, or None if invalid."""
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
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pixi run pytest tests/test_scoring.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add solution/scoring.py tests/test_scoring.py
git commit -m "feat: scoring functions for task 1 (max confidence) and task 2 (per-class)"
```

---

## Task 5: RL Loop (`solution/rl_loop.py`)

**Files:**
- Create: `solution/rl_loop.py`
- Create: `tests/test_rl_loop.py`

**Depends on:** Task 3 (Prior interface)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rl_loop.py
import numpy as np
import pytest
import torch
from copy import deepcopy
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
    # Scorer returns the length of the SMILES as the score (longer = better)
    def len_scorer(smiles_list):
        return np.array([min(len(s) / 20, 1.0) for s in smiles_list], dtype=np.float32)

    result = run_rl(mock_prior, mock_agent, len_scorer, steps=3, batch_size=8)
    # Result should be sorted: longer SMILES first
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run pytest tests/test_rl_loop.py -v
```

Expected: `ModuleNotFoundError: No module named 'solution.rl_loop'`

- [ ] **Step 3: Implement `solution/rl_loop.py`**

```python
from copy import deepcopy
from typing import Callable, List
import numpy as np
import torch
from torch.optim import Adam
from solution.prior import Prior


def run_rl(
    prior: Prior,
    agent: Prior,
    scorer_fn: Callable[[List[str]], np.ndarray],
    sigma: float = 120.0,
    steps: int = 300,
    batch_size: int = 128,
    lr: float = 1e-4,
) -> List[str]:
    """
    REINVENT REINFORCE loop.

    Loss: mean( (prior_log_prob(smiles) + sigma * score - agent_log_prob(smiles))^2 )

    Args:
        prior: frozen Prior (KL anchor)
        agent: trainable Prior (deepcopy of prior, parameters unfrozen)
        scorer_fn: callable(smiles_list) -> np.ndarray of scores in [0, 1]
        sigma: reward scaling factor (REINVENT default: 120)
        steps: number of RL iterations
        batch_size: molecules sampled per step
        lr: Adam learning rate

    Returns:
        All valid molecules seen during training, deduplicated and sorted by score (best first).
    """
    optimizer = Adam(agent.rnn.parameters(), lr=lr)
    best: List[tuple] = []  # (score: float, smiles: str)

    for step in range(steps):
        smiles_batch = agent.sample(batch_size)
        scores = scorer_fn(smiles_batch)                              # (batch_size,)

        prior_lp = prior.log_prob(smiles_batch).detach()             # (batch_size,) no grad
        agent_lp = agent.log_prob(smiles_batch)                      # (batch_size,) with grad

        scores_t = torch.tensor(scores, dtype=torch.float32, device=agent.device)
        augmented = prior_lp + sigma * scores_t
        loss = ((augmented - agent_lp) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for smi, sc in zip(smiles_batch, scores.tolist()):
            if sc > 0.0:
                best.append((sc, smi))

        if step % 50 == 0:
            print(f"step {step:4d} | mean {scores.mean():.3f} | max {scores.max():.3f} | loss {loss.item():.2f}")

    # Sort by score descending, deduplicate
    best.sort(key=lambda x: x[0], reverse=True)
    seen: set = set()
    unique: List[str] = []
    for sc, smi in best:
        if smi not in seen:
            seen.add(smi)
            unique.append(smi)
    return unique
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pixi run pytest tests/test_rl_loop.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add solution/rl_loop.py tests/test_rl_loop.py
git commit -m "feat: REINFORCE RL loop with deduplication and score sorting"
```

---

## Task 6: Phase 1 Explore (`solution/explore.py`)

**Files:**
- Create: `solution/explore.py`
- Modify: `tests/test_prior.py` (append `test_top_bits_for_class`)

**Depends on:** Tasks 3, 4, 5

- [ ] **Step 1: Write failing test for `top_bits_for_class`**

Append to `tests/test_prior.py`:

```python
# Append to tests/test_prior.py
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent / "tasks" / "task-2" / "model"))
import importlib.util as _ilu

def _load_task2():
    spec = _ilu.spec_from_file_location(
        "task2_m",
        _Path(__file__).parent.parent / "tasks" / "task-2" / "model" / "model.py"
    )
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    model = mod.load_model(str(
        _Path(__file__).parent.parent / "tasks" / "task-2" / "model" / "model.safetensors"
    ))
    model.eval()
    return model

def test_top_bits_for_class_returns_correct_length():
    from solution.explore import top_bits_for_class
    model = _load_task2()
    bits = top_bits_for_class(model, class_idx=0, topk=20)
    assert isinstance(bits, list)
    assert len(bits) == 20
    assert all(isinstance(b, int) for b in bits)
    assert all(0 <= b < 2048 for b in bits)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pixi run pytest tests/test_prior.py::test_top_bits_for_class_returns_correct_length -v
```

Expected: `ModuleNotFoundError: No module named 'solution.explore'`

- [ ] **Step 3: Implement `solution/explore.py`**

```python
#!/usr/bin/env python
"""
Phase 1: Explore signal quality before full RL runs.

Run from hackathon_copenhagen/:
    pixi run python solution/explore.py
"""
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent


def _load_task_model(task_num: int) -> nn.Module:
    spec = importlib.util.spec_from_file_location(
        f"task{task_num}_model",
        ROOT / "tasks" / f"task-{task_num}" / "model" / "model.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.load_model(str(ROOT / "tasks" / f"task-{task_num}" / "model" / "model.safetensors"))
    model.eval()
    return model


def top_bits_for_class(task_model: nn.Module, class_idx: int,
                        n_bits: int = 2048, topk: int = 20) -> List[int]:
    """
    Gradient attribution: which ECFP4 bit positions most strongly activate class_idx?
    Uses a zero-vector input — identifies bits whose activation would increase class score.
    """
    task_model.eval()
    x = torch.zeros(1, n_bits, requires_grad=True)
    logit = task_model(x)[0, class_idx]
    logit.backward()
    return x.grad.squeeze().abs().topk(topk).indices.tolist()


def main():
    from solution.prior import load_prior
    from solution.scoring import score_max_confidence, score_class
    from solution.rl_loop import run_rl

    print("Loading prior and task models...")
    prior = load_prior(str(ROOT / "models" / "reinvent.prior"))
    task1 = _load_task_model(1)
    task2 = _load_task_model(2)

    # ── Step 1: Prior baseline ──────────────────────────────────────────────
    print("\n=== Step 1: Prior baseline (2000 samples) ===")
    samples = prior.sample(2000)
    valid_count = sum(1 for s in samples if s)
    print(f"Valid SMILES generated: {valid_count}/2000")

    t1 = score_max_confidence(samples, task1)
    print(f"Task 1 — mean: {t1.mean():.3f}  max: {t1.max():.3f}  "
          f"high-confidence (>0.9): {(t1 > 0.9).sum()}")

    for c in range(4):
        sc = score_class(samples, task2, c)
        print(f"Task 2 class_{c} — mean: {sc.mean():.3f}  max: {sc.max():.3f}  "
              f"high-confidence (>0.8): {(sc > 0.8).sum()}")

    # ── Step 2: Gradient attribution ────────────────────────────────────────
    print("\n=== Step 2: Gradient attribution (Task 2) ===")
    for c in range(4):
        bits = top_bits_for_class(task2, c, topk=10)
        print(f"class_{c} top-10 ECFP4 bits: {bits}")

    # ── Step 3: Short RL probe (Task 2, class_0, 50 steps) ─────────────────
    print("\n=== Step 3: RL probe — Task 2 class_0, sigma=120, 50 steps ===")
    agent = deepcopy(prior)
    for p in agent.rnn.parameters():
        p.requires_grad_(True)
    agent.rnn.train()

    scorer = lambda smiles: score_class(smiles, task2, class_idx=0)
    run_rl(prior, agent, scorer, sigma=120.0, steps=50, batch_size=64, lr=1e-4)

    # Evaluate final agent quality
    final_samples = agent.sample(500)
    final_scores = score_class(final_samples, task2, class_idx=0)
    print(f"\nFinal agent — mean score: {final_scores.mean():.3f}  max: {final_scores.max():.3f}")

    if final_scores.mean() > 0.70:
        print("✓ Strong signal. Proceed with sigma=120 for full run.")
    elif final_scores.mean() > 0.50:
        print("~ Moderate signal. Consider sigma=60 or longer warmup.")
    else:
        print("✗ Weak signal. Try sigma=240 or review scoring/checkpoint.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the attribution test**

```bash
pixi run pytest tests/test_prior.py::test_top_bits_for_class_returns_correct_length -v
```

Expected: PASS.

- [ ] **Step 5: Run explore.py end-to-end (requires `models/reinvent.prior`)**

```bash
pixi run python solution/explore.py
```

Expected output: prior baseline stats, attribution bit lists, RL probe score trajectory ending above 0.50.

- [ ] **Step 6: Commit**

```bash
git add solution/explore.py tests/test_prior.py
git commit -m "feat: Phase 1 explore — prior baseline, attribution, RL probe"
```

---

## Task 7: Run Scripts (`solution/run_task1.py`, `solution/run_task2.py`)

**Files:**
- Create: `solution/run_task1.py`
- Create: `solution/run_task2.py`

**Depends on:** Tasks 3, 4, 5

- [ ] **Step 1: Implement `solution/run_task1.py`**

```python
#!/usr/bin/env python
"""
Phase 2 — Task 1: Binary classifier reconstruction.
1 RL agent, 300 steps, max-confidence scorer → top-1000 SMILES.

Run from hackathon_copenhagen/:
    pixi run python solution/run_task1.py
"""
import importlib.util
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent


def main():
    from solution.prior import load_prior
    from solution.scoring import score_max_confidence
    from solution.rl_loop import run_rl

    # Load prior + task model
    prior = load_prior(str(ROOT / "models" / "reinvent.prior"))

    spec = importlib.util.spec_from_file_location(
        "task1_model", ROOT / "tasks" / "task-1" / "model" / "model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    task_model = mod.load_model(str(ROOT / "tasks" / "task-1" / "model" / "model.safetensors"))
    task_model.eval()

    # Create trainable agent
    agent = deepcopy(prior)
    for p in agent.rnn.parameters():
        p.requires_grad_(True)
    agent.rnn.train()

    scorer = lambda smiles: score_max_confidence(smiles, task_model)

    print("Running RL for Task 1 (300 steps)...")
    results = run_rl(prior, agent, scorer, sigma=120.0, steps=300, batch_size=128, lr=1e-4)
    top1000 = results[:1000]

    out_path = ROOT / "results" / "task1_submission.csv"
    pd.DataFrame({"smiles": top1000}).to_csv(out_path, index=False)
    print(f"Saved {len(top1000)} molecules to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Implement `solution/run_task2.py`**

```python
#!/usr/bin/env python
"""
Phase 2 — Task 2: 4-class classifier reconstruction.
4 RL agents (one per class), 300 steps each → top-250 per class → 1000 total.

Run from hackathon_copenhagen/:
    pixi run python solution/run_task2.py
"""
import importlib.util
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
N_CLASSES = 4
PER_CLASS = 250


def main():
    from solution.prior import load_prior
    from solution.scoring import score_class
    from solution.rl_loop import run_rl

    prior = load_prior(str(ROOT / "models" / "reinvent.prior"))

    spec = importlib.util.spec_from_file_location(
        "task2_model", ROOT / "tasks" / "task-2" / "model" / "model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    task_model = mod.load_model(str(ROOT / "tasks" / "task-2" / "model" / "model.safetensors"))
    task_model.eval()

    all_results = []

    for class_idx in range(N_CLASSES):
        print(f"\n=== RL agent for class_{class_idx} ===")

        # Fresh agent per class (deepcopy of frozen prior, then unfreeze)
        agent = deepcopy(prior)
        for p in agent.rnn.parameters():
            p.requires_grad_(True)
        agent.rnn.train()

        scorer = lambda smiles, c=class_idx: score_class(smiles, task_model, c)
        results = run_rl(prior, agent, scorer, sigma=120.0, steps=300, batch_size=128, lr=1e-4)

        top_n = results[:PER_CLASS]
        all_results.extend(top_n)
        print(f"class_{class_idx}: collected {len(top_n)} molecules")

    out_path = ROOT / "results" / "task2_submission.csv"
    pd.DataFrame({"smiles": all_results}).to_csv(out_path, index=False)
    print(f"\nSaved {len(all_results)} molecules to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test both scripts (requires `models/reinvent.prior`)**

```bash
# Quick sanity check: 5 steps instead of 300
pixi run python -c "
import sys; sys.argv = []
from pathlib import Path
import importlib.util, pandas as pd
from copy import deepcopy

ROOT = Path('.')
from solution.prior import load_prior
from solution.scoring import score_max_confidence
from solution.rl_loop import run_rl

prior = load_prior(str(ROOT / 'models' / 'reinvent.prior'))
agent = deepcopy(prior)
for p in agent.rnn.parameters(): p.requires_grad_(True)
agent.rnn.train()

spec = importlib.util.spec_from_file_location('t1', ROOT / 'tasks/task-1/model/model.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
task_model = mod.load_model(str(ROOT / 'tasks/task-1/model/model.safetensors'))
task_model.eval()

scorer = lambda smiles: score_max_confidence(smiles, task_model)
results = run_rl(prior, agent, scorer, steps=5, batch_size=16)
print(f'Got {len(results)} molecules. First: {results[0] if results else None}')
assert isinstance(results, list), 'Expected list'
print('Smoke test passed.')
"
```

Expected: prints molecule count and "Smoke test passed."

- [ ] **Step 4: Run full Task 1 and submit**

```bash
pixi run python solution/run_task1.py
# Check output
pixi run python -c "import pandas as pd; df = pd.read_csv('results/task1_submission.csv'); print(df.shape, df.head())"
```

Expected: `(1000, 1)` with valid SMILES in the `smiles` column.

- [ ] **Step 5: Run full Task 2 and submit**

```bash
pixi run python solution/run_task2.py
pixi run python -c "import pandas as pd; df = pd.read_csv('results/task2_submission.csv'); print(df.shape, df.head())"
```

Expected: `(1000, 1)` — 250 molecules per class.

- [ ] **Step 6: Commit**

```bash
git add solution/run_task1.py solution/run_task2.py results/.gitkeep
git commit -m "feat: Phase 2 run scripts — task1 (1 agent) and task2 (4 agents × 250)"
```

---

## Full Test Suite

Run all tests at any point with:

```bash
pixi run pytest tests/ -v
```

Expected final output:
```
tests/test_prior.py::test_tokenize_smiles_simple PASSED
tests/test_prior.py::test_tokenize_smiles_multichar PASSED
tests/test_prior.py::test_tokenize_smiles_brackets PASSED
tests/test_prior.py::test_vocabulary_len PASSED
tests/test_prior.py::test_vocabulary_encode_includes_start_end PASSED
tests/test_prior.py::test_vocabulary_encode_unknown_token_returns_none PASSED
tests/test_prior.py::test_vocabulary_decode_strips_special PASSED
tests/test_prior.py::test_vocabulary_roundtrip PASSED
tests/test_prior.py::test_rnn_output_shape PASSED
tests/test_prior.py::test_prior_sample_returns_n_strings PASSED
tests/test_prior.py::test_prior_log_prob_shape PASSED
tests/test_prior.py::test_prior_log_prob_invalid_smiles_gets_zero PASSED
tests/test_prior.py::test_prior_log_prob_is_differentiable PASSED
tests/test_prior.py::test_top_bits_for_class_returns_correct_length PASSED
tests/test_scoring.py::test_featurize_filters_invalid PASSED
tests/test_scoring.py::test_featurize_all_invalid PASSED
tests/test_scoring.py::test_score_max_confidence_shape PASSED
tests/test_scoring.py::test_score_max_confidence_invalid_is_zero PASSED
tests/test_scoring.py::test_score_max_confidence_valid_in_range PASSED
tests/test_scoring.py::test_score_class_shape PASSED
tests/test_scoring.py::test_score_class_invalid_is_zero PASSED
tests/test_scoring.py::test_score_class_sums_to_one_across_classes PASSED
tests/test_rl_loop.py::test_run_rl_returns_list PASSED
tests/test_rl_loop.py::test_run_rl_excludes_zero_score_molecules PASSED
tests/test_rl_loop.py::test_run_rl_sorted_descending PASSED
tests/test_rl_loop.py::test_run_rl_deduplicates PASSED
tests/test_rl_loop.py::test_run_rl_agent_parameters_update PASSED
27 passed
```

---

## Parallelisation Map (for subagent dispatch)

```
Task 1 (scaffold)
     ├──▶ Task 2 (prior — tokenizer/vocab)  ──▶ Task 3 (prior — RNN/load) ──▶ Task 5 (RL loop)
     └──▶ Task 4 (scoring)                                                 └──▶ Task 6 (explore)
                                                                                Task 7 (run scripts)
```

Tasks 2 and 4 can run in parallel after Task 1.
Task 3 can start after Task 2.
Tasks 5, 6, and 7 can start after Tasks 3 and 4 complete.

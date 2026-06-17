import re
from typing import List, Optional

import torch
import torch.nn as nn

# Handles: bracket atoms [NH], multi-char atoms Cl/Br, ring bonds %10, all SMILES chars
_SMILES_RE = re.compile(
    r"(\[[^\[\]]{1,10}\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

# NOTE: Do NOT hardcode PAD/GO/EOS strings — the real checkpoint uses $, ^, #.
# Confirmed: pad='$'(0), bos='^'(1), eos='#'(2). Vocabulary takes them as params.


def tokenize_smiles(smiles: str) -> List[str]:
    """SMILES string → list of tokens. Multi-char atoms (Cl, Br) kept intact."""
    return _SMILES_RE.findall(smiles)


class Vocabulary:
    def __init__(self, tokens: List[str], pad: str = "$", start: str = "^", end: str = "\n"):
        """
        tokens: ordered list of all token strings (including special tokens at their positions)
        pad/start/end: the actual string values used for those special roles in this vocabulary
        """
        self._tokens = tokens
        self._t2i = {t: i for i, t in enumerate(tokens)}
        self._i2t = {i: t for i, t in enumerate(tokens)}
        self.pad_idx = self._t2i[pad]
        self.start_idx = self._t2i[start]
        self.end_idx = self._t2i[end]
        self._pad = pad
        self._start = start
        self._end = end

    def __len__(self) -> int:
        return len(self._tokens)

    def encode(self, smiles: str) -> Optional[List[int]]:
        """SMILES → [start_idx, token..., end_idx]. None if empty or unknown token."""
        atom_tokens = tokenize_smiles(smiles)
        if not atom_tokens:
            return None
        toks = [self._start] + atom_tokens + [self._end]
        try:
            return [self._t2i[t] for t in toks]
        except KeyError:
            return None

    def decode(self, indices: List[int]) -> str:
        """Token indices → SMILES string (special tokens stripped)."""
        skip = {self.pad_idx, self.start_idx, self.end_idx}
        return "".join(self._i2t[i] for i in indices if i not in skip and i in self._i2t)


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
        self._embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self._rnn = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self._linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        """x: (batch, seq_len) → logits: (batch, seq_len, vocab_size), hidden"""
        emb = self._embedding(x)
        out, hidden = self._rnn(emb, hidden)
        return self._linear(out), hidden


class Prior:
    def __init__(self, rnn: RNN, vocabulary: Vocabulary, device: torch.device):
        self.rnn = rnn
        self.vocabulary = vocabulary
        self.device = device
        self.rnn.to(device)

    def sample(self, n: int = 128, max_len: int = 100) -> List[str]:
        """Sample n SMILES. Uses torch.no_grad — not differentiable."""
        start_idx = self.vocabulary.start_idx
        end_idx = self.vocabulary.end_idx
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
        Differentiable w.r.t. rnn parameters when rnn.requires_grad=True.
        Invalid / unencodable SMILES get log_prob = 0 (as a detached scalar).
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


def load_prior(path: str, freeze: bool = True) -> "Prior":
    """
    Load REINVENT4 prior from .prior checkpoint.

    Confirmed checkpoint format (from inspection):
      Top-level keys: max_sequence_length, metadata, model_type, network,
                      network_params, tokenizer, version, vocabulary
      vocabulary: {'tokens': {token_str: index}, 'pad_token': 0,
                   'bos_token': 1, 'eos_token': 2, 'unk_token': None}
      network_params: {'dropout': 0.0, 'layer_size': 512, 'num_layers': 3,
                       'cell_type': 'lstm', 'embedding_layer_size': 256, ...}
      network: state dict with keys _embedding.weight, _rnn.*, _linear.*
    """
    import sys
    import types

    # First try loading directly (reinvent may be installed)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        # Stub reinvent namespace so torch.load can unpickle without full install
        for mod_name in ['reinvent', 'reinvent.models', 'reinvent.models.reinvent',
                         'reinvent.models.reinvent.model', 'reinvent.chemistry',
                         'reinvent.tokenization']:
            if mod_name not in sys.modules:
                stub = types.ModuleType(mod_name)
                stub.__getattr__ = lambda name, _m=mod_name: type(name, (), {})
                sys.modules[mod_name] = stub
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Extract vocabulary: tokens dict {str: int} → ordered list by index
    vocab_data = ckpt["vocabulary"]
    tokens_dict: dict = vocab_data["tokens"]
    tokens: List[str] = [t for t, _ in sorted(tokens_dict.items(), key=lambda kv: kv[1])]

    i2t = {v: k for k, v in tokens_dict.items()}
    pad_str = i2t[vocab_data["pad_token"]]    # '$'
    start_str = i2t[vocab_data["bos_token"]]  # '^'
    end_str = i2t[vocab_data["eos_token"]]    # '#'

    vocabulary = Vocabulary(tokens, pad=pad_str, start=start_str, end=end_str)

    params = ckpt.get("network_params", {})
    rnn = RNN(
        vocab_size=len(vocabulary),
        embed_dim=params.get("embedding_layer_size", 256),
        hidden_size=params.get("layer_size", 512),
        num_layers=params.get("num_layers", 3),
    )
    rnn.load_state_dict(ckpt["network"])

    if freeze:
        rnn.eval()
        for p in rnn.parameters():
            p.requires_grad_(False)

    return Prior(rnn, vocabulary, get_device())

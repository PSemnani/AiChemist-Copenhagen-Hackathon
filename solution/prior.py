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

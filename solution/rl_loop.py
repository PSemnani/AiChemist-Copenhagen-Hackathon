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
        prior: frozen Prior (KL anchor, parameters require_grad=False)
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

        prior_lp = prior.log_prob(smiles_batch).detach()             # no grad needed
        agent_lp = agent.log_prob(smiles_batch)                      # grad flows here

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

    # Sort descending by score (primary) then by SMILES length (secondary), deduplicate preserving order
    best.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    seen: set = set()
    unique: List[str] = []
    for sc, smi in best:
        if smi not in seen:
            seen.add(smi)
            unique.append(smi)
    return unique

#!/usr/bin/env python
"""
Phase 1: Explore signal quality before full RL runs.

Run from hackathon_copenhagen/:
    pixi run python solution/explore.py
"""
import importlib.util
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
    Uses a zero-vector input — identifies bits whose activation increases the class score.
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

    final_samples = agent.sample(500)
    final_scores = score_class(final_samples, task2, class_idx=0)
    print(f"\nFinal agent — mean score: {final_scores.mean():.3f}  max: {final_scores.max():.3f}")

    if final_scores.mean() > 0.70:
        print("Strong signal. Proceed with sigma=120 for full run.")
    elif final_scores.mean() > 0.50:
        print("Moderate signal. Consider sigma=60 or longer warmup.")
    else:
        print("Weak signal. Try sigma=240 or review scoring/checkpoint.")


if __name__ == "__main__":
    main()

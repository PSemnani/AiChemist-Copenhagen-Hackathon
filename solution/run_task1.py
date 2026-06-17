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

    prior = load_prior(str(ROOT / "models" / "reinvent.prior"))

    spec = importlib.util.spec_from_file_location(
        "task1_model", ROOT / "tasks" / "task-1" / "model" / "model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    task_model = mod.load_model(str(ROOT / "tasks" / "task-1" / "model" / "model.safetensors"))
    task_model.eval()

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

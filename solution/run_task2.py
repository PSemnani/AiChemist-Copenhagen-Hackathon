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

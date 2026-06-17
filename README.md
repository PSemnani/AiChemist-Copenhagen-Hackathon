# 🧪 Training Data Reconstruction from Molecular Classifiers

> *If a model memorized its training data — can you make it confess?*

Copenhagen Hackathon 2026 · AiChemist team

---

## The Challenge

Two black-box neural networks classify molecules by biological activity. We don't know what they were trained on. Goal: submit 1000 molecules per task that match the original training set.

| | Task 1 | Task 2 |
|---|---|---|
| **Type** | Binary classifier | 4-class classifier |
| **Train accuracy** | ~100% (overfit) | ~90% (regularized) |
| **Input** | ECFP4 fingerprint (2048 bits) | same |

---

## Structure

```
solution/
  generative/     # Approach 1: REINVENT RL pipeline
  screening/
    screen.py     # Approach 2: screen any dataset → ranked submissions

results/
  best/           # Final submissions
  experiments/    # All intermediate attempts

docs/             # Design spec & implementation plan
tests/            # 27 unit tests for the generative pipeline
```

---

## Quick Start

```bash
pixi install

# Screen any molecule dataset
PYTHONPATH=. pixi run python solution/screening/screen.py \
    --data path/to/molecules.csv \
    --smiles-col SMILES \
    --output results/best/
```

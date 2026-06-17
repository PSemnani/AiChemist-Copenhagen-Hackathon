# AiChemist Copenhagen Hackathon

**Task:** Given two pre-trained molecular classifiers, reconstruct the molecules they were trained on — without access to the training data.

## The Problem

Two MLP models classify molecules using ECFP4 fingerprints (Morgan, radius 2, 2048 bits):

| Task | Type | Train acc | Test acc | Signal |
|---|---|---|---|---|
| Task 1 | Binary (class 0 / class 1) | ~1.000 | ~0.652 | Overfit — memorized training data |
| Task 2 | 4-class | ~0.898 | ~0.857 | Regularized — generalizes well |

Submit 1000 molecules per task that match the training set as closely as possible.

## Our Approach

### What We Tried

**Approach 1 — Generative (REINVENT RL):** Use a pre-trained SMILES generator steered by the model's confidence via REINFORCE. Generates novel drug-like molecules. Produced 0 correct matches — the prior generates valid molecules but not the specific training molecules.

**Approach 2 — Dataset Screening (winner):** Screen known molecule databases and rank by the model's raw logit signal.

### The Key Insight: Raw Logit as a Membership Signal

Softmax confidence saturates at 1.0 for all drug-like molecules — useless for discrimination. Raw logits keep full dynamic range:

```
ChEMBL (2.85M molecules):  max logit ≈ 10   →  0/1000 correct
CARBIDE (cardiotoxicity):  max logit ≈ 27   →  60+ correct
hERG dataset (38K mols):   max logit ≈ 66   →  best results
```

The 6× logit gap between the wrong dataset (ChEMBL) and the right one (hERG) is the model "remembering" molecules it trained on. An overfit model pushes the logit for memorized inputs toward very large values.

**Selection criterion (Experiment 3 — certainty):**
```
certainty = |logit_class0 - logit_class1|
```
The most decisive predictions correspond to the most strongly memorized molecules.

### Dataset Discovery

Training data is from the **hERG channel bioactivity dataset** — hERG is the cardiac potassium channel whose blockage causes drug-induced cardiotoxicity (the biological theme of both tasks). This connects to the **CARBIDE dataset** (cardiotoxicity, 3K molecules) which we discovered first via logit magnitude comparison.

## Repository Structure

```
tasks/                     # Given by organizers — model weights + code
  task-1/model/            # Binary MLP: 2048 → 512 → 128 → 2
  task-2/model/            # 4-class MLP: 2048 → [BN+ReLU+Dropout] 512 → 256 → 4

solution/
  generative/              # Approach 1: REINVENT RL pipeline
    prior.py               # SMILES LSTM wrapper (load, sample, log_prob)
    scoring.py             # ECFP4 scoring functions
    rl_loop.py             # REINFORCE training loop
    explore.py             # Phase 1: prior baseline + RL probe
    run_task1.py           # Entry point: Task 1 (1 agent)
    run_task2.py           # Entry point: Task 2 (4 agents × 250)
  screening/               # Approach 2: Dataset screening (successful)
    screen.py              # Unified screener — any CSV → ranked submissions
    fingerprint_inversion.py  # Gradient descent + Tanimoto lookup

results/
  best/                    # Final submissions (highest leaderboard scores)
    task1_hERG_certainty.csv
    task2_hERG_certainty.csv
  experiments/             # All intermediate attempts (reference)

tests/                     # Unit tests for generative pipeline (27 tests)
docs/                      # Design spec and implementation plan
models/transformer/        # brsynth ECFP4→SMILES Transformer tokens
```

## Running the Screening Pipeline

```bash
# Install dependencies
pixi install

# Screen any molecule dataset (CSV with SMILES column)
PYTHONPATH=. pixi run python solution/screening/screen.py \
    --data path/to/molecules.csv \
    --smiles-col SMILES \
    --output results/best/
```

## Running the Generative Pipeline

```bash
# Phase 1: validate signal quality (~2 min)
PYTHONPATH=. pixi run python solution/generative/explore.py

# Phase 2: generate molecules (requires models/reinvent.prior)
PYTHONPATH=. pixi run python solution/generative/run_task1.py  # ~10 min
PYTHONPATH=. pixi run python solution/generative/run_task2.py  # ~40 min
```

## Dependencies

Managed via [pixi](https://pixi.sh): `pixi install`

Core: `torch`, `rdkit`, `safetensors`, `numpy`, `pandas`, `scikit-learn`

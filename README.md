# Training Data Reconstruction from Molecular Classifiers

> *If a model memorized its training data — can you make it confess?*

Copenhagen Hackathon 2026 · AiChemist team

---

## The Problem

Two black-box neural networks classify molecules by biological activity. The training data is unknown. Goal: submit 1000 molecules per task that match the original training set.

| | Task 1 | Task 2 |
|---|---|---|
| Type | Binary classifier | 4-class classifier |
| Train accuracy | ~100% (overfit) | ~90% (regularized) |
| Input | ECFP4 fingerprint · 2048 bits | same |

---

## The Approach — Membership Inference via Gradient Norm

An overfit model sits at a **loss minimum** for every molecule it memorized. At a minimum, the gradient of the loss with respect to the input is approximately zero — this is a direct consequence of calculus.

```
∂L/∂x ≈ 0   →   model is at a loss minimum   →   molecule was memorized
∂L/∂x >> 0  →   model is extrapolating        →   molecule is unknown
```

We exploit this to rank candidate molecules: the smaller the gradient norm, the more likely the molecule is a training member.

**Key insight:** softmax confidence saturates to 1.0 for all drug-like molecules and is useless as a discrimination signal. Raw logit magnitude — and gradient norm derived from it — preserves the information that softmax discards.

---

## Pipeline

```
1. Load CARBIDE data_splits.tsv          (known cardiotoxicity dataset)
2. Filter to training folds (Fold ≠ 0)   (seed=0 → fold 0 held out as test)
3. Compute ECFP4 fingerprint per molecule (RDKit Morgan, radius=2, 2048 bits)
4. Forward pass through task model        (get logit scores)
5. Backpropagate cross-entropy loss       (loss.backward())
6. Rank by gradient norm (ascending)      (smallest = most memorized)
7. Export top-1000 for Task 1             (top-250 per class for Task 2)
```

---

## Usage

```bash
pixi install

# Run the full pipeline
PYTHONPATH=. pixi run python solution/screening/membership.py

# Screen any custom dataset
PYTHONPATH=. pixi run python solution/screening/screen.py \
    --data path/to/molecules.csv \
    --smiles-col SMILES \
    --output results/best/
```

**Output:** `results/best/task1_submission.csv` and `results/best/task2_submission.csv`

---

## Structure

```
solution/screening/
  membership.py   ← main pipeline (fold-based + gradient norm)
  screen.py       ← generic screener for any molecule dataset
  utils.py        ← shared: load_model(), featurize()

results/best/     ← final submission CSVs
docs/             ← design spec, implementation plan, presentation
tasks/            ← provided models (weights + architecture)
```

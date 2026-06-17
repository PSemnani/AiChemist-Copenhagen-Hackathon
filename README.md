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

## Our Journey

### Approach 1 — Generative (didn't work)
We built a full **REINVENT RL pipeline**: a pre-trained SMILES generator steered toward high-confidence predictions via REINFORCE. Beautiful pipeline, wrong tool — it generates *plausible* molecules, not the *specific* training ones.

### Approach 2 — Membership Inference (worked ✅)
**The key insight:** softmax confidence is useless — it saturates to 1.0 for all drug-like molecules. **Raw logits** tell a completely different story.

```
Wrong dataset:   max logit ≈ 10   →  0 / 1000 correct
Right dataset:   max logit ≈ 66   →  60+ correct
```

That 6× gap is the model *remembering* molecules it trained on. We used this as a membership signal to identify the training distribution by scanning candidate databases until logits jumped.

**Selection criterion:**
```
certainty = | logit_class0 − logit_class1 |
```
The more decisive the model, the more likely it memorized that molecule.

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

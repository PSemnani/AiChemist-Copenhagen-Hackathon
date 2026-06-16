# Task 2 — 4-class classifier

## The challenge
You are given a **realistic, production-style** neural network that performs **4-class
classification** on molecules (`class_0 … class_3`). Unlike Task 1, this one uses the
regularization, mini-batching and early stopping you'd use in practice and generalizes to
held-out data.

**Your goal:** Reconstruct the data the model was trained on (training-data
reconstruction).

## The model
- **Architecture:** `2048 → [Linear+BatchNorm+ReLU+Dropout(0.3)] 512 → 256 → 4`.
- **Input:** ECFP4 — RDKit Morgan fingerprint, **radius 2, 2048 bits**, float32 0/1 vector.
- **Output:** 4 logits in the order of `CLASSES` (softmax + cross-entropy at train time).
- **Training:** AdamW (lr 1e-3, weight decay 1e-4), mini-batches of 128, class-weighted
  cross-entropy (inverse-frequency), early stopping on validation loss.
- Hyperparameters and test metrics are in `model/config.json`.

## Performance (from `model/config.json`)
| metric | train | test |
|---|---|---|
| Accuracy | 0.898 | 0.857 |
| Balanced accuracy | 0.916 | 0.868 |
| Macro F1 | 0.873 | 0.814 |
| Weighted F1 | 0.902 | 0.865 |
| Macro ROC-AUC (OvR) | 0.989 | 0.967 |

## The model bundle (`model/`)
| file | what it is |
|---|---|
| `model.safetensors` | trained weights (safetensors) |
| `model.py` | architecture + ECFP4 featurization (`load_model`, `featurize_smiles`, `CLASSES`) |
| `config.json` | hyperparameters, input spec, metrics |

## Example inference
Run from inside `model/` (or put it on your path).

```python
from model import load_model, featurize_smiles, CLASSES

model = load_model("model.safetensors")              # MLP in eval mode
X = featurize_smiles(["OC(=O)C(O)c1ccccc1"])         # (n, 2048) ECFP4 features
probs = model(X).softmax(-1)                          # (n, 4) over CLASSES
pred = probs.argmax(-1)
print(CLASSES)                                       # ['class_0','class_1','class_2','class_3']
print([CLASSES[i] for i in pred])                    # predicted class per molecule
```

The model is plain PyTorch, so you can also take gradients of the output (or a loss)
with respect to the input fingerprint and optimize.

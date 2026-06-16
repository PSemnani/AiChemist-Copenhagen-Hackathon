# Task 1 — binary classifier 

## The challenge
You are given a small neural network that performs **binary classification** on molecules
(class 0 vs class 1). It was **deliberately overfit** (no regularization and trained until
the training loss was essentially zero).

**Your goal:** Reconstruct the data the model was trained on (training-data
reconstruction).

## The model
- **Architecture:** plain MLP `2048 → 512 → ReLU → 128 → ReLU → 2` (no dropout / batch-norm).
- **Input:** ECFP4 — RDKit Morgan fingerprint, **radius 2, 2048 bits**, float32 0/1 vector.
- **Output:** 2 logits → `[class 0, class 1]` (softmax + cross-entropy at train time).
- Hyperparameters and the (overfit) train/test metrics are in `model/config.json`.

## Performance (from `model/config.json`)
The model is intentionally overfit, so train and test diverge sharply:

| split | cross-entropy | accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| train | 0.0010 | 1.000 | 1.000 | 1.000 |
| test  | 1.280 | 0.652 | 0.676 | 0.747 |

## The model bundle (`model/`)
| file | what it is |
|---|---|
| `model.safetensors` | trained weights (safetensors) |
| `model.py` | architecture + ECFP4 featurization (`load_model`, `featurize_smiles`, `smiles_to_ecfp4`) |
| `config.json` | hyperparameters, input spec, metrics |

## Example inference
Run from inside `model/` (or put it on your path).

```python
from model import load_model, featurize_smiles

model = load_model("model.safetensors")              # MLP in eval mode
X = featurize_smiles(["CC(=O)Oc1ccccc1C(=O)O"])      # (n, 2048) ECFP4 features
logits = model(X)                                    # (n, 2): [class 0, class 1]
probs = logits.softmax(-1)
print(probs)
```

The model is plain PyTorch, so you can also take gradients of the output (or a loss)
with respect to the input fingerprint and optimize.

# Hackathon Copenhagen

## The task
- Recover (reconstruct) the molecules each model was trained on.
- For each model (task-1 and task-2) you should submit an ordered list of 1000 molecules that you think were in the training data. Order them by your confidence (For position 1 you are more confident that it was in the training data than for position 2 etc.)



## Repository layout
- Each model is in the tasks subfolders 

```
tasks/<task>/
  instruction.md   # the task brief + an example inference snippet
  model/           # self-contained bundle: weights + model definition + config.json
```


## The challenges

| task | what it is |
|---|---|
| [`tasks/task-1/`](tasks/task-1/instruction.md) | **Binary classifier** — an overfit model|
| [`tasks/task-2/`](tasks/task-2/instruction.md) | **4-class classifier** — a realistic, regularized model |

Both models share the same input representation: **ECFP4** (RDKit Morgan fingerprint,
radius 2, 2048 bits) and ship as **safetensors** bundles. Each bundle's `featurize_smiles`
handles the ECFP4 conversion for you, so there is no need to reimplement it.
See each task's `instruction.md` for the full brief and an inference example.


## Submission format

One CSV **per model** (two submissions total), with a single `smiles` column and **1000 rows**,
ordered most-confident first (row 1 = the molecule you are most confident was in the training data).

## Runtime dependencies

Loading and running the models needs `torch`, `rdkit`, `safetensors`, and `numpy`. The
`pixi.toml` file pins compatible versions, and `pixi install` is the easiest way to get them.
If you have never used pixi and are interested in it, check out https://pixi.prefix.dev/latest/.
Of course you can also use your own package management tool if you don't want to use pixi.


After you have your environment, see the task `instruction.md` for a runnable snippet for each model.


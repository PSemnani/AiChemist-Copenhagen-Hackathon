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

Add your molecules to the following spreadsheet: https://docs.google.com/spreadsheets/d/10x8vqq9sOj6gjPoUxh_aINKpqZX0MTCxUkjkLWxiXBY/edit?gid=0#gid=0
Please only ever modify your teams tab (column A for task 1 and column B for task 2). Every half hour (XX:00 and XX:30) the submissions for both tasks will be evaluated and the results are displayed in the leaderboard tab.

## Runtime dependencies

Loading and running the models needs `torch`, `rdkit`, `safetensors`, and `numpy`. The
`pixi.toml` file pins compatible versions, and `pixi install` is the easiest way to get them.
If you have never used pixi and are interested in it, check out https://pixi.prefix.dev/latest/.
Of course you can also use your own package management tool if you don't want to use pixi.


After you have your environment, see the task `instruction.md` for a runnable snippet for each model.


# Design Spec: REINVENT RL Pipeline for Training-Data Reconstruction

**Date:** 2026-06-17
**Project:** AiChemist Hackathon Copenhagen
**Goal:** Reconstruct 1000 training molecules per task (task-1 binary, task-2 4-class) by steering a pre-trained molecular generative model toward molecules the target classifiers are highly confident about.

---

## Problem Context

Two pre-trained molecular classifiers are provided (ECFP4 → MLP). The training data is unknown. We must submit 1000 candidate SMILES per task that best match the original training sets. No molecule library is available — molecules must be generated.

- **Task 1:** Overfit binary classifier (train acc ≈ 1.0, test acc ≈ 0.65). Sharp confidence signal — training molecules receive near-certainty scores.
- **Task 2:** Regularized 4-class classifier (train acc ≈ 0.90, test acc ≈ 0.86). Softer confidence signal due to generalization.

---

## Architecture

Five components, each with a single responsibility:

```
┌─────────────────┐     sample / log_prob      ┌───────────────┐
│  REINVENT Prior │ ──────────────────────────▶│   RL Agent    │
│  (frozen LSTM)  │ ◀── KL anchor (loss term)  │ (trainable)   │
└─────────────────┘                            └───────┬───────┘
                                                       │ SMILES batch
                                                       ▼
                                               ┌───────────────┐
                                               │    Scorer     │
                                               │  ECFP4 →      │
                                               │  task model   │
                                               │  confidence   │
                                               └───────┬───────┘
                                                       │ scores [0,1]
                                                       ▼
                                               ┌───────────────┐
                                               │   RL Loop     │
                                               │  REINFORCE    │
                                               └───────┬───────┘
                                                       │ top-1000
                                                       ▼
                                               ┌───────────────┐
                                               │  Output CSV   │
                                               └───────────────┘
```

The prior stays frozen and acts as a KL regularizer — it prevents the agent from collapsing into non-drug-like gibberish. The same pipeline runs twice with only the scorer swapped.

---

## Component Designs

### 1. Prior & Agent (`solution/prior.py`)

**Model:** Character-level SMILES LSTM from `MolecularAI/REINVENT4` (pre-trained on ~1.8M ChEMBL molecules, ~12MB `.prior` file).

**Architecture:**
```
Embedding(vocab_size, 256)
→ LSTM(256, 512, num_layers=3)
→ Linear(512, vocab_size)
→ log_softmax
```

**Interface:**
```python
prior = load_prior("models/reinvent.prior")   # frozen
agent = deepcopy(prior)                        # trainable copy

agent.sample(n=128)       # → list[str]  (SMILES strings)
agent.log_prob(smiles)    # → Tensor (n,)  (sequence log-likelihood)
```

The prior's weights are frozen for the entire run. The agent is initialized as an exact copy and updated only by the RL loop.

---

### 2. Scoring Function (`solution/scoring.py`)

Converts a batch of SMILES into scalar scores in [0, 1].

`smiles_to_ecfp4` is imported from the task's `model/model.py` (already defined there — no reimplementation needed).

```python
from tasks.task_1.model.model import smiles_to_ecfp4   # swap path for task 2

def score(smiles_list: list[str], task_model: nn.Module) -> np.ndarray:
    scores = np.zeros(len(smiles_list))
    valid_fps, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_ecfp4(smi)       # None if invalid
        if fp is not None:
            valid_fps.append(fp)
            valid_idx.append(i)
    if valid_fps:
        X = torch.from_numpy(np.stack(valid_fps))
        with torch.no_grad():
            probs = task_model(X).softmax(-1)
        scores[valid_idx] = probs.max(dim=-1).values.numpy()
    return scores
```

- Invalid SMILES receive score 0 (no penalty needed — REINFORCE naturally down-weights low-scoring sequences).
- Task 2 model uses `BatchNorm1d` + `Dropout`. `task_model.eval()` must be called before scoring to disable dropout and use running BN statistics, ensuring deterministic scores.

---

### 3. RL Loop (`solution/rl_loop.py`)

REINVENT's REINFORCE loss:

```
augmented_log_likelihood = prior_log_prob(smiles) + σ × score
loss = mean( (augmented_ll − agent_log_prob(smiles))² )
```

```python
def run_rl(prior, agent, task_model, sigma=120, steps=300, batch_size=128, lr=1e-4):
    optimizer = Adam(agent.parameters(), lr=lr)
    best = []   # (score, smiles) list

    for step in range(steps):
        smiles_batch = agent.sample(batch_size)
        scores       = score(smiles_batch, task_model)

        prior_lp = prior.log_prob(smiles_batch)
        agent_lp = agent.log_prob(smiles_batch)

        augmented = prior_lp + sigma * torch.tensor(scores)
        loss = ((augmented - agent_lp) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        best.extend((sc, smi) for smi, sc in zip(smiles_batch, scores) if sc > 0)

        if step % 50 == 0:
            print(f"step {step:4d} | mean {scores.mean():.3f} | max {scores.max():.3f}")

    best.sort(reverse=True)
    return [smi for _, smi in best[:1000]]
```

**Hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| σ (sigma) | 120 | REINVENT published default; balances reward vs. prior KL |
| steps | 300 | ~10 min on Apple Silicon MPS; increase if time allows |
| batch_size | 128 | Fits in MPS memory; matches Task 2 training batch size |
| lr | 1e-4 | Standard REINVENT learning rate |

No experience replay — the `best` buffer accumulates all valid molecules across steps and returns the top-1000 by score at the end.

---

### 4. Entry Points (`solution/run_task1.py`, `solution/run_task2.py`)

```python
# run_task1.py (run_task2.py is identical except for model path and class count)
prior = load_prior("models/reinvent.prior")
agent = deepcopy(prior)
task_model = load_model("tasks/task-1/model/model.safetensors")
task_model.eval()

results = run_rl(prior, agent, task_model)

pd.DataFrame({"smiles": results}).to_csv("results/task1_submission.csv", index=False)
```

Run from `hackathon_copenhagen/`:
```bash
pixi run python solution/run_task1.py
pixi run python solution/run_task2.py
```

---

## File Structure

```
hackathon_copenhagen/
  solution/
    prior.py             # LSTM wrapper: load_prior, sample, log_prob
    scoring.py           # score() + task model loader
    rl_loop.py           # run_rl()
    run_task1.py         # end-to-end entry point for task 1
    run_task2.py         # end-to-end entry point for task 2
  models/
    reinvent.prior       # downloaded from MolecularAI/REINVENT4 releases
  results/
    task1_submission.csv
    task2_submission.csv
  docs/superpowers/specs/
    2026-06-17-reinvent-rl-pipeline-design.md  # this file
```

---

## Time Plan (3-hour window)

| Time | Activity |
|---|---|
| 0:00–0:30 | pixi setup, download `reinvent.prior`, verify model loading |
| 0:30–1:15 | Implement `prior.py`, `scoring.py`, smoke-test both |
| 1:15–1:30 | Implement `rl_loop.py` + `run_task1.py` |
| 1:30–1:45 | Run Task 1 RL (300 steps ≈ 10 min), submit CSV |
| 1:45–2:00 | Adapt `run_task2.py`, run Task 2 RL, submit CSV |
| 2:00–3:00 | Iterate: tune σ, increase steps, re-submit based on leaderboard |

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `reinvent.prior` format incompatible with manual loading | Fall back to installing `reinvent` pip package for just the prior loader |
| Task 2 RL converges slowly (soft signal) | Lower σ to 60 to reduce mode collapse pressure; accept top-1000 by raw score |
| Mode collapse (agent repeats one molecule) | Reduce σ or add a diversity penalty (Tanimoto similarity within batch) |
| Apple Silicon MPS instability | Fall back to CPU; 300 steps still runs in ~25 min |

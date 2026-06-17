# Design Spec: REINVENT RL Pipeline for Training-Data Reconstruction

**Date:** 2026-06-17
**Project:** AiChemist Hackathon Copenhagen
**Goal:** Reconstruct 1000 training molecules per task (task-1 binary, task-2 4-class) by steering a pre-trained molecular generative model toward molecules the target classifiers are highly confident about.

---

## Problem Context

Two pre-trained molecular classifiers are provided (ECFP4 → MLP). The training data is unknown. We must submit 1000 candidate SMILES per task that best match the original training sets. No molecule library is available — molecules must be generated.

- **Task 1:** Overfit binary classifier (train acc ≈ 1.0, test acc ≈ 0.65). Binary: class 0 vs class 1 (likely inactive vs. active against a biological target). Sharp confidence signal — training molecules receive near-certainty scores.
- **Task 2:** Regularized 4-class classifier (train acc ≈ 0.90, test acc ≈ 0.86). Four classes representing distinct biological activity profiles. Softer confidence signal due to generalization.

Both tasks operate on drug-like molecules. The classes represent biological interactions (activity against targets, mechanism of action, etc.). This shapes the generative strategy: we steer toward known pharmacophore space, not arbitrary chemical space.

---

## Two-Phase Strategy

| Phase | Goal | Time | Output |
|---|---|---|---|
| **1 — Explore** | Understand signal quality and class chemistry before committing | ~30 min | Insight + validated σ |
| **2 — Generate** | Full RL runs with validated settings | ~60 min | 1000 SMILES per task |

---

## Architecture

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
                                               │  class-specific│
                                               │  confidence   │
                                               └───────┬───────┘
                                                       │ scores [0,1]
                                                       ▼
                                               ┌───────────────┐
                                               │   RL Loop     │
                                               │  REINFORCE    │
                                               └───────┬───────┘
                                                       │ top-N per class
                                                       ▼
                                               ┌───────────────┐
                                               │  Output CSV   │
                                               └───────────────┘
```

Task 1: 1 agent → top-1000.
Task 2: 4 agents (one per class) → top-250 each → concatenate to 1000.
Same RL loop and prior for all agents; only the scorer changes.

---

## Component Designs

### 1. Prior & Agent (`solution/prior.py`)

**Model:** Character-level SMILES LSTM from `MolecularAI/REINVENT4` (pre-trained on ~1.8M ChEMBL molecules, ~12MB `.prior` file downloaded from the REINVENT4 GitHub releases page).

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

Two scorers — one per task type.

`smiles_to_ecfp4` is imported from the task's `model/model.py` (already defined there — no reimplementation needed).

```python
from tasks.task_1.model.model import smiles_to_ecfp4   # swap path for task 2

def _featurize(smiles_list):
    """Returns (valid_fps, valid_indices). Invalid SMILES are silently skipped."""
    fps, idx = [], []
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_ecfp4(smi)
        if fp is not None:
            fps.append(fp)
            idx.append(i)
    return fps, idx

def score_max_confidence(smiles_list, task_model) -> np.ndarray:
    """Task 1: max softmax confidence across both classes."""
    scores = np.zeros(len(smiles_list))
    fps, idx = _featurize(smiles_list)
    if fps:
        X = torch.from_numpy(np.stack(fps))
        with torch.no_grad():
            probs = task_model(X).softmax(-1)
        scores[idx] = probs.max(dim=-1).values.numpy()
    return scores

def score_class(smiles_list, task_model, class_idx: int) -> np.ndarray:
    """Task 2: confidence for a specific class (used per-agent in Phase 2)."""
    scores = np.zeros(len(smiles_list))
    fps, idx = _featurize(smiles_list)
    if fps:
        X = torch.from_numpy(np.stack(fps))
        with torch.no_grad():
            probs = task_model(X).softmax(-1)
        scores[idx] = probs[:, class_idx].numpy()
    return scores
```

- Invalid SMILES receive score 0.
- `task_model.eval()` must be called before scoring — Task 2 uses `BatchNorm1d` + `Dropout`, which are non-deterministic in train mode.

---

### 3. Phase 1 — Explore (`solution/explore.py`)

Three sequential steps, run once before any full RL:

**Step 1: Prior baseline**
Sample 2000 molecules from the prior (no RL). Score with both task models. Report:
- Confidence distribution (histogram)
- Class breakdown for Task 2 (how many fall in each class?)
- Any molecules already scoring > 0.9 (free wins)

**Step 2: Gradient attribution per class**
For each class, identify which ECFP4 bits most strongly activate it:

```python
def top_bits_for_class(task_model, class_idx, n_bits=2048, topk=20):
    # Use first-layer weights as a proxy for bit importance per class
    # Full attribution: backprop class logit through the whole network
    task_model.eval()
    x = torch.zeros(1, n_bits, requires_grad=True)
    logit = task_model(x)[0, class_idx]
    logit.backward()
    return x.grad.squeeze().abs().topk(topk).indices.tolist()
```

These bit indices correspond to Morgan atom environments — a cheminformatician can interpret them, and they tell us what scaffolds/pharmacophores define each class.

**Step 3: Short RL probe**
Run 50 steps of RL targeting Task 2 class_0 with σ=120. Plot mean score per step. If score rises from ~0.5 to >0.7, the signal is strong — proceed with Phase 2 at σ=120. If flat, try σ=60 and σ=240 before committing.

---

### 4. RL Loop (`solution/rl_loop.py`)

REINVENT's REINFORCE loss:

```
augmented_log_likelihood = prior_log_prob(smiles) + σ × score
loss = mean( (augmented_ll − agent_log_prob(smiles))² )
```

```python
def run_rl(prior, agent, scorer_fn, sigma=120, steps=300, batch_size=128, lr=1e-4):
    """scorer_fn: callable(smiles_list) -> np.ndarray of scores in [0,1]"""
    optimizer = Adam(agent.parameters(), lr=lr)
    best = []   # (score, smiles)

    for step in range(steps):
        smiles_batch = agent.sample(batch_size)
        scores       = scorer_fn(smiles_batch)           # (batch_size,)

        prior_lp = prior.log_prob(smiles_batch)          # (batch_size,)
        agent_lp = agent.log_prob(smiles_batch)          # (batch_size,)

        augmented = prior_lp + sigma * torch.tensor(scores)
        loss = ((augmented - agent_lp) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        best.extend((sc, smi) for smi, sc in zip(smiles_batch, scores) if sc > 0)

        if step % 50 == 0:
            print(f"step {step:4d} | mean {scores.mean():.3f} | max {scores.max():.3f}")

    best.sort(reverse=True)
    return [smi for _, smi in best]
```

`scorer_fn` is the only thing that changes between agents — the loop itself is task-agnostic.

**Hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| σ (sigma) | 120 | REINVENT published default; validated in Phase 1 probe |
| steps | 300 | ~10 min per agent on Apple Silicon MPS |
| batch_size | 128 | Fits MPS memory; matches Task 2 training batch size |
| lr | 1e-4 | Standard REINVENT learning rate |

---

### 5. Entry Points

**`solution/run_task1.py`**
```python
prior = load_prior("models/reinvent.prior")
agent = deepcopy(prior)
task_model = load_model("tasks/task-1/model/model.safetensors"); task_model.eval()

scorer = lambda smiles: score_max_confidence(smiles, task_model)
results = run_rl(prior, agent, scorer)[:1000]

pd.DataFrame({"smiles": results}).to_csv("results/task1_submission.csv", index=False)
```

**`solution/run_task2.py`**
```python
prior = load_prior("models/reinvent.prior")
task_model = load_model("tasks/task-2/model/model.safetensors"); task_model.eval()

all_results = []
for class_idx in range(4):
    agent = deepcopy(prior)          # fresh agent per class
    scorer = lambda smiles, c=class_idx: score_class(smiles, task_model, c)
    molecules = run_rl(prior, agent, scorer)
    all_results.extend(molecules[:250])   # top-250 per class

pd.DataFrame({"smiles": all_results}).to_csv("results/task2_submission.csv", index=False)
```

Run from `hackathon_copenhagen/`:
```bash
pixi run python solution/explore.py      # Phase 1
pixi run python solution/run_task1.py   # Phase 2 - task 1
pixi run python solution/run_task2.py   # Phase 2 - task 2 (4 agents × 300 steps)
```

---

## File Structure

```
hackathon_copenhagen/
  solution/
    prior.py             # LSTM wrapper: load_prior, sample, log_prob
    scoring.py           # score_max_confidence(), score_class(), _featurize()
    rl_loop.py           # run_rl()
    explore.py           # Phase 1: prior baseline + attribution + RL probe
    run_task1.py         # Phase 2 entry point for task 1 (1 agent)
    run_task2.py         # Phase 2 entry point for task 2 (4 agents × 250)
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
| 0:30–1:00 | Implement `prior.py`, `scoring.py`, `rl_loop.py` |
| 1:00–1:15 | Run `explore.py` (Phase 1) — validate signal, inspect class chemistry |
| 1:15–1:30 | Run Task 1 RL (300 steps ≈ 10 min), submit CSV |
| 1:30–2:10 | Run Task 2 RL (4 agents × 10 min ≈ 40 min), submit CSV |
| 2:10–3:00 | Iterate: tune σ, increase steps, re-submit based on leaderboard |

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `reinvent.prior` format incompatible with manual loading | Fall back to installing `reinvent` pip package for just the prior loader |
| Phase 1 probe shows flat score trajectory | Try σ=60 and σ=240; if still flat, switch to `score_max_confidence` for Task 2 |
| Task 2 RL: 4 agents × 300 steps overruns time budget | Reduce to 150 steps per agent (~20 min total); accept slightly lower quality |
| Mode collapse (agent repeats one molecule) | Reduce σ; add Tanimoto diversity penalty within batch if time allows |
| Apple Silicon MPS instability | Fall back to CPU; 300 steps ≈ 25 min on CPU |

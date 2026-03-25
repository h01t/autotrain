# AutoTrain Platform — Autonomous ML Training & Self-Correction

> Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).
> Concept: What if your ML model could train itself, correct its own mistakes, and stop when it's good enough — all while you sleep?

---

## The Problem

Training ML models today is a manual, babysitting-heavy process:
- You tweak hyperparameters, kick off training, wait, check metrics, repeat
- If something goes wrong mid-training (loss plateau, overfitting), you find out hours later
- There's no structured way to track *what was tried* and *why it helped or didn't*
- Knowledge from failed experiments lives in your head, not in the codebase

## The Vision

A platform where you:
1. **Submit** your model + dataset + target metric (e.g., `AUC >= 0.85`, `mAP@0.5 >= 0.80`)
2. **Walk away** — the platform autonomously trains, evaluates, and self-corrects
3. **Get notified** when the target is hit (or when the platform is stuck)
4. **Review** a clean git history showing every experiment, what changed, and what improved

Every training iteration is a git commit. Every hyperparameter change is traceable. Every failed experiment is documented so the system (and you) learn from it.

---

## Core Concepts

### 1. The Training Loop as a Git History

```
commit abc1234 — "Epoch 10: val_auc=0.72, lr=0.001, batch=32"
commit def5678 — "Epoch 20: val_auc=0.76, increased batch to 64"
commit ghi9012 — "Epoch 25: val_auc=0.74 ↓ — reverted batch to 32, added warmup"
commit jkl3456 — "Epoch 35: val_auc=0.81, added cosine annealing"
commit mno7890 — "Epoch 50: val_auc=0.86 ✓ TARGET HIT — training complete"
```

Each commit captures:
- Model weights (or pointer to them)
- Training config (hyperparameters, architecture changes)
- Metrics snapshot (loss, target metric, per-class breakdown)
- What the agent changed and why (commit message as reasoning trace)

### 2. The Agent Loop (inspired by autoresearch)

```
┌─────────────────────────────────────────────┐
│                 AGENT LOOP                   │
│                                              │
│  1. Read current metrics + training history  │
│  2. Analyze: what's working, what's not      │
│  3. Decide: what to try next                 │
│     - Hyperparameter tuning                  │
│     - Architecture modification              │
│     - Data augmentation changes              │
│     - Learning rate schedule adjustment      │
│     - Early stopping / restart decisions     │
│  4. Modify train.py (constrained scope)      │
│  5. Run training for N minutes/epochs        │
│  6. Evaluate against target metric           │
│  7. Git commit with results + reasoning      │
│  8. If target met → STOP + NOTIFY            │
│     If stuck → ESCALATE or try new strategy  │
│  9. Loop back to step 1                      │
└─────────────────────────────────────────────┘
```

### 3. Self-Correction Strategies

The agent doesn't just blindly train — it has a playbook:

| Situation | Strategy |
|-----------|----------|
| Loss plateauing | Reduce LR, change scheduler, add warmup |
| Overfitting (train >> val) | Add dropout, reduce model size, more augmentation |
| Underfitting (both low) | Increase model capacity, longer training, higher LR |
| Metric oscillating | Reduce LR, increase batch size, add EMA |
| NaN/Inf loss | Reduce LR by 10x, check data pipeline, add gradient clipping |
| No progress after N iterations | Try fundamentally different approach (different architecture) |

### 4. Guardrails

- **Budget limits**: Max GPU hours, max experiments, max cost
- **Revert on regression**: If a change makes things worse, auto-revert
- **Human-in-the-loop**: Escalate to developer when stuck for N iterations
- **Constrained scope**: Agent can only modify specific files (like autoresearch's train.py constraint)
- **Snapshot management**: Only keep top-K model checkpoints to avoid disk bloat

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    AutoTrain Platform                  │
│                                                        │
│  ┌─────────┐    ┌──────────┐    ┌─────────────────┐  │
│  │  Web UI  │───▶│  API     │───▶│  Job Scheduler   │  │
│  │ (React)  │    │(FastAPI) │    │  (Celery/Redis)  │  │
│  └─────────┘    └──────────┘    └────────┬──────────┘  │
│                                           │             │
│                                    ┌──────▼──────┐     │
│                                    │  Agent Core  │     │
│                                    │  (LLM-based) │     │
│                                    └──────┬──────┘     │
│                                           │             │
│                  ┌────────────────────────┼──────┐     │
│                  │                        │      │     │
│           ┌──────▼──────┐  ┌──────▼─────┐ ┌─────▼───┐ │
│           │  GPU Worker  │  │  Git Repo  │ │ Metrics │ │
│           │  (Training)  │  │ (History)  │ │   DB    │ │
│           └─────────────┘  └────────────┘ └─────────┘ │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Notification Service                 │  │
│  │  (Slack / Email / Webhook when target hit)        │  │
│  └──────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### Component Breakdown

**Agent Core** — The brain. An LLM (Claude/GPT) that:
- Reads training logs and metric history
- Decides what to change next
- Generates code modifications
- Writes descriptive commit messages
- Knows when to stop or escalate

**GPU Worker** — The muscle. Runs actual training:
- Supports CUDA (local or cloud)
- Fixed time-budget per experiment (like autoresearch's 5-min constraint)
- Streams metrics back to the platform in real-time
- Supports checkpointing and resume

**Git Repo** — The memory. Every experiment is a commit:
- Branch per training job
- Commit = config + metrics + reasoning
- Easy diff between any two experiments
- `git log` becomes your experiment tracker

**Metrics DB** — The dashboard backing:
- Time-series storage (InfluxDB or TimescaleDB)
- Per-experiment metrics for comparison
- Supports custom metric definitions

**Web UI** — Developer insight:
- Real-time training curves
- Side-by-side model comparison (pre vs post)
- Git history visualization
- Abort/pause/resume controls
- Target metric configuration

---

## User Flow

```
Developer                          Platform
   │                                  │
   │  1. Submit model repo URL        │
   │  2. Define target: AUC >= 0.85   │
   │  3. Set budget: 4 GPU hours      │
   ├─────────────────────────────────▶│
   │                                  │ Clone repo
   │                                  │ Analyze model + data
   │                                  │ Create training branch
   │                                  │
   │   "Training started"             │
   │◀─────────────────────────────────┤
   │                                  │ ┌──────────────┐
   │                                  │ │  Agent Loop   │
   │   Live dashboard available       │ │  Train 5 min  │
   │                                  │ │  Evaluate     │
   │                                  │ │  Git commit   │
   │                                  │ │  Decide next  │
   │                                  │ │  Repeat...    │
   │                                  │ └──────────────┘
   │                                  │
   │   "Target hit! AUC = 0.86"       │
   │◀─────────────────────────────────┤
   │                                  │
   │  4. Review git history           │
   │  5. Pull best model weights      │
   │  6. Compare before/after         │
   ├─────────────────────────────────▶│
   │                                  │
```

---

## What Makes This Different from MLflow/W&B/etc.

| Feature | MLflow / W&B | AutoTrain |
|---------|-------------|-----------|
| Experiment tracking | ✓ Manual logging | ✓ Automatic via git |
| Hyperparameter tuning | Grid/Bayesian search | LLM-driven, context-aware |
| Self-correction | ✗ | ✓ Agent analyzes failures |
| Reasoning trace | ✗ | ✓ Commit messages explain *why* |
| Human-like decisions | ✗ | ✓ "This looks like overfitting, adding dropout" |
| Target-driven stopping | Basic early stopping | ✓ Stops at business metric target |
| Git-native history | ✗ | ✓ Every experiment is a commit |

The key differentiator: **MLflow tracks what happened. AutoTrain decides what to do next.**

---

## MVP Scope (v0.1)

Start small, prove the concept:

1. **CLI tool** (not a web platform yet)
2. **Single model, single GPU** (no distributed)
3. **Agent = Claude via API** (not a custom model)
4. **Constraints**: Agent can only modify `train.py` and `config.yaml`
5. **Metrics**: Read from stdout/log file (no custom integrations)
6. **Git**: Local repo, commit after each iteration
7. **Notification**: Print to terminal + optional webhook

### MVP File Structure

```
autotrain/
├── autotrain/
│   ├── __init__.py
│   ├── agent.py          # LLM-based decision maker
│   ├── runner.py          # Training executor
│   ├── evaluator.py       # Metric extraction + target checking
│   ├── git_manager.py     # Commit, branch, revert operations
│   ├── strategy.py        # Self-correction playbook
│   └── config.py          # Platform configuration
├── templates/
│   └── program.md         # Default agent instructions (à la autoresearch)
├── scripts/
│   └── autotrain.py       # CLI entry point
├── pyproject.toml
└── README.md
```

### MVP Usage

```bash
# Point at your ML project, set a target, go
autotrain run \
    --repo /path/to/my-model \
    --train-script train.py \
    --metric "val_auc" \
    --target 0.85 \
    --budget 4h \
    --gpu cuda:0

# Watch it work
autotrain status

# Review what it did
autotrain history  # pretty git log
autotrain compare HEAD~5 HEAD  # diff two experiments
```

---

## Tech Stack (MVP)

| Component | Choice | Why |
|-----------|--------|-----|
| Language | Python | ML ecosystem |
| LLM | Claude API | Best at code modification + reasoning |
| Git | GitPython | Programmatic git operations |
| Task runner | subprocess | Simple, run training scripts directly |
| Metrics parsing | regex + JSON | Parse stdout/log files for metrics |
| Config | YAML | Human-readable experiment configs |
| Notification | Slack webhook | Simple, most teams already use it |

---

## Future Ideas (v1.0+)

- **Web dashboard** with real-time training curves (React + FastAPI)
- **Multi-GPU support** — distribute experiments across machines
- **Model comparison UI** — side-by-side inference on test samples
- **Dataset quality agent** — detect data issues before training
- **Architecture search** — agent proposes entirely new model architectures
- **Cost optimization** — auto-select cheapest GPU that fits the model
- **Team collaboration** — multiple developers, shared experiment history
- **Pre-built strategies** — "I know this is a vision model, start with these recipes"
- **Integration with cloud GPUs** — Lambda Labs, RunPod, AWS spot instances
- **A/B testing pipeline** — auto-deploy winning model for A/B testing

---

## Open Questions

1. **How much freedom should the agent have?** Autoresearch constrains to one file. Should we allow architecture changes? New data augmentations? Different loss functions?
2. **How to handle long training runs?** 5-min iterations work for small models. What about models that need hours per epoch?
3. **How to define "stuck"?** After how many non-improving iterations should the agent try a fundamentally different approach?
4. **Model weight storage?** Git doesn't handle large binaries well. Use Git LFS? Or just store pointers to checkpoints on disk/S3?
5. **Multi-objective targets?** What if the user wants `AUC >= 0.85 AND latency <= 50ms`?

---

## Relationship to This Project (object-det)

This vehicle detection project is a perfect **first test case** for the platform:
- Clear target metric (mAP@0.5 >= 0.80)
- Well-defined training script (`scripts/train.py`)
- Remote GPU available (blackbox, 3060 Ti)
- Multiple knobs to tune (model size, augmentation, LR, epochs)

Once the platform MVP exists, we could literally point it at this repo and say "get mAP to 0.85" and watch it work.

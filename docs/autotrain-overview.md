# AutoTrain: Autonomous ML Training Platform

**Version 0.1.0 | March 2026**

---

## The Idea

Training machine learning models is an iterative, time-consuming process. A researcher adjusts hyperparameters, runs training, waits for results, analyzes metrics, and repeats. This cycle can take days or weeks of manual work.

AutoTrain replaces this manual loop with an autonomous agent. Given a training script, a target metric, and a compute budget, AutoTrain uses an LLM to propose code changes, executes training on remote GPU hardware, evaluates results, and iterates — all without human intervention.

The core insight: an LLM can read a training script, understand experiment history, and propose informed hyperparameter changes just like a researcher would — but it can run 24/7 without fatigue or context-switching.

## Use Case: Vehicle Detection

Our primary test case uses YOLO object detection trained on the KITTI autonomous driving dataset. The goal: push mAP (mean Average Precision) from a baseline to >= 0.90.

**Setup:**
- Model: YOLO (v8/v11) with pretrained weights
- Dataset: KITTI (vehicles, pedestrians, cyclists — 5,985 train / 1,496 val images)
- Compute: NVIDIA RTX 3060 Ti (8GB VRAM) via SSH
- Agent: DeepSeek Reasoner (cost-effective reasoning LLM)
- Budget: 4 hours, 50 iterations, $2.00 API cap

**Results (First Run):**

| Metric | After AutoTrain | Target |
|--------|-----------------|--------|
| mAP@0.5 | **0.8384** | 0.90 |
| mAP@0.5-0.95 | 0.5984 | — |
| Precision | 0.8349 | — |
| Recall | 0.7599 | — |

Achieved 0.84 mAP in 2 iterations, ~35 minutes total training time. The agent reduced learning rate (0.01 -> 0.008) and batch size (16 -> 8). Total API cost: < $0.01.

---

## How It Works

When `autotrain run` is started, the following sequence executes:

```
                        YOUR MACHINE                          REMOTE GPU
                        ────────────                          ──────────
                    ┌─────────────────┐
                    │  1. INITIALIZE   │
                    │  - Git init/branch│
                    │  - SQLite DB     │
                    │  - rsync + setup │──── SSH ────► mkdir + uv sync
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  2. CALL AGENT   │
                    │  - Build prompt  │
                    │  - History +     │
                    │    training curve│◄──── LLM API (DeepSeek/Claude)
                    │  - Parse response│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  3. VALIDATE     │
                    │  - Sandbox check │  Only whitelisted files
                    │  - Search/replace│  Exact text matching
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  4. APPLY + SYNC │
                    │  - Edit files    │
                    │  - Git commit    │
                    │  - rsync to GPU  │──── rsync ──► Updated train.py
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐        ┌──────────────────┐
                    │  5. TRAIN        │        │  GPU MONITOR     │
                    │  - SSH execute   │── SSH ─► setsid python    │
                    │  - Stream epochs │◄ stdout │  train.py       │
                    │  - Budget check  │        │  nvidia-smi poll │
                    │  - Extract metric│        └──────────────────┘
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  6. EVALUATE     │
                    │  - Compare to    │
                    │    best metric   │
                    │  - Target hit?   │──► YES: Stop (completed)
                    │  - Regressed?    │──► Revert commit
                    │  - Crashed?      │──► Checkpoint resume
                    └────────┬────────┘
                             │
                             ▼
                      Loop back to 2
```

### Key Design Decisions

**Sandbox Safety.** The agent can only modify files explicitly listed in `writable_files`. All changes go through search/replace validation — the agent must reference exact text from the current file, preventing hallucinated edits.

**Git-Based Rollback.** Every iteration is a git commit. If a change makes metrics worse, the commit is reverted. The experiment branch preserves the full history of what was tried.

**Process Isolation.** Training runs in a `setsid` process group on the remote machine. If SSH drops (laptop sleeps, network hiccup), training continues. AutoTrain reconnects and resumes tailing output.

**Multi-Provider LLM.** Supports Anthropic (Claude), DeepSeek (including `deepseek-v4-pro` and `deepseek-v4-flash`), and Ollama. `deepseek-v4-pro` is the stronger reasoning-oriented option, while `deepseek-v4-flash` is the faster lower-cost option.

**Budget Enforcement.** Hard limits on wall-clock time, iteration count, and API spend. Budget is checked both between iterations and mid-training — a single run cannot exceed the total time budget.

**Training Curves as Agent Context.** The agent doesn't just see final metrics — it receives per-epoch training curves with trend summaries, enabling it to diagnose overfitting, learning rate issues, and convergence problems.

**Checkpoint Recovery.** When training crashes, AutoTrain detects checkpoint files (e.g. `last.pt`, `best.pt`) on the remote and injects `AUTOTRAIN_RESUME_FROM` into the next training run's environment. The user's training script can optionally resume from the checkpoint instead of restarting from scratch.

**GPU Monitoring.** A background watchdog queries `nvidia-smi` on the remote GPU (via SSH) at regular intervals, storing utilization, memory, and temperature snapshots. The dashboard displays these in real time, eliminating the need for a separate `watch nvidia-smi` terminal.

---

## Architecture

```
src/autotrain/           4,936 lines across 43 modules
├── agent/               LLM client, prompt builder, response parser
├── config/              YAML config loader, Pydantic schemas, defaults
├── core/                Agent loop, budget tracker, state machine
├── execution/           Local and SSH executors, checkpoint detection
├── experiment/          Git ops, metrics extraction, epoch parsing, sandbox
├── monitor/             Streamlit real-time dashboard
├── notify/              Terminal + webhook notifications
├── storage/             SQLite database (WAL mode), models, queries
├── util/                Logging, signals
└── watchdog/            GPU metrics collection, disk/health monitoring
```

**Dependencies:** 8 runtime packages (anthropic, click, pydantic, pyyaml, structlog, requests, streamlit, plotly). No heavy ML frameworks — AutoTrain orchestrates, it doesn't train.

### Storage Schema (v4)

SQLite with WAL mode for concurrent access between the training process and the monitoring dashboard. Five tables with incremental migrations:

| Table | Purpose |
|-------|---------|
| `runs` | Top-level run records (status, best metric, budget, git branch) |
| `iterations` | Per-iteration data (outcome, metric, hypothesis, commit, checkpoint) |
| `metric_snapshots` | Time-series metric values per iteration |
| `epoch_metrics` | Per-epoch training metrics (JSON blobs — loss, mAP, precision, recall) |
| `gpu_snapshots` | GPU utilization, memory, temperature time-series |

### Monitoring Dashboard

A Streamlit dashboard (`autotrain monitor`) provides:

- **Metric progress chart** — color-coded by outcome with target/best lines
- **Training curves** — per-epoch loss and score metrics with multi-iteration overlay
- **GPU resources** — live utilization, memory %, temperature with history chart
- **Iteration history** — table with outcome, metric, hypothesis, duration, checkpoint flags
- **Iteration comparison** — side-by-side diff of any two iterations
- **Agent reasoning** — expandable log with color-coded outcomes, hypothesis, and changes
- **Cost & budget tracker** — progress bars with per-iteration rate estimates
- **Multi-run support** — sidebar run selector with summary stats

All data is read from the SQLite database — the dashboard is a pure read-only viewer.

---

## Configuration

A single YAML file controls the entire run:

```yaml
agent:
  provider: deepseek          # anthropic | deepseek | ollama
  model: deepseek-v4-pro      # deepseek-v4-flash, claude-sonnet-4, etc.
  temperature: 0.3
  hard_timeout_seconds: 180

metric:
  name: mAP
  target: 0.90
  direction: maximize

budget:
  time_seconds: 14400         # 4 hours
  max_iterations: 50
  api_dollars: 2.00
  experiment_timeout_seconds: 900  # 15 min per training run

execution:
  train_command: ".venv/bin/python train.py"
  mode: ssh
  ssh_host: blackbox
  ssh_remote_dir: /home/holt/dev/object-det
  ssh_setup_command: "~/.local/bin/uv sync"
  gpu_device: "0"

sandbox:
  writable_files:
    - train.py

notify:
  terminal: true
```

The `ssh_setup_command` runs once before the first training iteration — installing dependencies on a fresh remote machine with zero manual SSH work.

---

## Future Improvements

**SSH resilience.** When the SSH tail connection drops during a long training run, reconnect and capture the final output instead of marking the iteration as crashed. Store partial training output to avoid losing completed results.

**Multi-file experiments.** Allow the agent to modify multiple files per iteration — model architecture, data augmentation configs, custom loss functions — giving it more leverage beyond hyperparameter tuning.

**Parallel experiments.** Run multiple hypotheses simultaneously across different GPUs or machines. Compare results and keep the best.

**Model selection agent.** Let the agent switch between model architectures (yolov8n -> yolo11s -> yolo11m) based on performance plateaus, not just tune hyperparameters.

**Smart early stopping.** The agent can request early termination of a training run based on streaming epoch metrics — if loss has plateaued or is diverging, kill the run early and iterate faster.

**Cross-run learning.** Feed insights from previous runs into new run prompts. Build a knowledge base of what worked and what didn't across experiments.

**Web dashboard.** React/FastAPI replacement for Streamlit with real-time WebSocket updates, experiment management, and one-click deployment of the best model.

**Distributed training.** Multi-GPU and multi-node support for larger models and datasets.

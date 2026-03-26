# AutoTrain: Autonomous ML Training Platform

**Version 0.1.0 | March 2026**

---

## The Idea

Training machine learning models is an iterative, time-consuming process. A researcher adjusts hyperparameters, runs training, waits for results, analyzes metrics, and repeats. This cycle can take days or weeks of manual work.

AutoTrain replaces this manual loop with an autonomous agent. Given a training script, a target metric, and a compute budget, AutoTrain uses an LLM to propose code changes, executes training on remote GPU hardware, evaluates results, and iterates — all without human intervention.

The core insight: an LLM can read a training script, understand experiment history, and propose informed hyperparameter changes just like a researcher would — but it can run 24/7 without fatigue or context-switching.

## Use Case: Vehicle Detection

Our first test case uses YOLO object detection trained on the KITTI autonomous driving dataset. The goal: push mAP (mean Average Precision) from a baseline to >= 0.90.

**Setup:**
- Model: YOLOv11n (nano) with pretrained weights
- Dataset: KITTI (vehicles, pedestrians, cyclists)
- Compute: NVIDIA GPU workstation (SSH remote)
- Agent: DeepSeek Chat (cost-effective LLM)
- Budget: 4 hours, 50 iterations, $2.00 API cap

**Results (First Run):**

| Metric | Baseline | After AutoTrain | Target |
|--------|----------|-----------------|--------|
| mAP@0.5 | — | **0.8384** | 0.80 |
| mAP@0.5-0.95 | — | 0.5984 | — |
| Precision | — | 0.8349 | — |
| Recall | — | 0.7599 | — |

Target hit in **2 iterations**, ~35 minutes total training time. The agent reduced learning rate (0.01 -> 0.008) and batch size (16 -> 8). Total API cost: < $0.01.

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
                    │  - Send to LLM   │◄──── LLM API (DeepSeek/Claude)
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
                    ┌────────▼────────┐
                    │  5. TRAIN        │
                    │  - SSH execute   │──── SSH ────► setsid python train.py
                    │  - Tail logs     │◄─── stdout ── Training output
                    │  - Extract metric│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  6. EVALUATE     │
                    │  - Compare to    │
                    │    best metric   │
                    │  - Target hit?   │──► YES: Stop (completed)
                    │  - Regressed?    │──► Revert commit
                    └────────┬────────┘
                             │
                             ▼
                      Loop back to 2
```

### Key Design Decisions

**Sandbox Safety.** The agent can only modify files explicitly listed in `writable_files`. All changes go through search/replace validation — the agent must reference exact text from the current file, preventing hallucinated edits.

**Git-Based Rollback.** Every iteration is a git commit. If a change makes metrics worse, the commit is reverted. The experiment branch preserves the full history of what was tried.

**Process Isolation.** Training runs in a `setsid` process group on the remote machine. If SSH drops (laptop sleeps, network hiccup), training continues. AutoTrain reconnects and resumes tailing output.

**Multi-Provider LLM.** Supports Anthropic (Claude), DeepSeek, and Ollama. The vehicle detection run used DeepSeek Chat at $0.27/1M input tokens — orders of magnitude cheaper than manual researcher time.

**Budget Enforcement.** Hard limits on wall-clock time, iteration count, and API spend. Training cannot run away.

---

## Architecture

```
src/autotrain/           3,876 lines across 42 modules
├── agent/               LLM client, prompt builder, response parser
├── config/              YAML config loader, Pydantic schemas
├── core/                Agent loop, budget tracker, state machine
├── execution/           Local and SSH executors
├── experiment/          Git ops, metrics extraction, sandbox
├── monitor/             Streamlit real-time dashboard
├── notify/              Terminal + webhook notifications
├── storage/             SQLite database, queries
├── util/                Logging, signals
└── watchdog/            Disk/GPU health monitoring
```

**Dependencies:** 8 runtime packages (anthropic, click, pydantic, pyyaml, structlog, requests, streamlit, plotly). No heavy ML frameworks — AutoTrain orchestrates, it doesn't train.

**Monitoring.** A Streamlit dashboard (`autotrain monitor`) shows live metric charts, iteration history, agent reasoning logs, and cost tracking — all read from the SQLite database in WAL mode for concurrent access.

---

## Configuration

A single YAML file controls the entire run:

```yaml
agent:
  provider: deepseek          # anthropic | deepseek | ollama
  model: deepseek-chat
  temperature: 0.3

metric:
  name: mAP
  target: 0.90
  direction: maximize

budget:
  time_seconds: 14400         # 4 hours
  max_iterations: 50
  api_dollars: 2.00

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
```

The `ssh_setup_command` runs once before the first training iteration — installing dependencies on a fresh remote machine with zero manual SSH work.

---

## Future Improvements

**Multi-file experiments.** Currently the agent modifies a single training script. Supporting model architecture files, data augmentation configs, and custom loss functions would give the agent more leverage.

**Checkpoint-aware recovery.** If a long training run crashes mid-epoch, resume from the last checkpoint instead of restarting. The state machine already has hooks for this.

**Parallel experiments.** Run multiple hypotheses simultaneously across multiple GPUs or machines. Compare results and keep the best.

**Smarter agent context.** Feed training curves (loss over epochs) back to the agent, not just final metrics. This would let it diagnose overfitting, learning rate issues, and convergence problems more precisely.

**Model selection.** Let the agent switch between model architectures (yolo11n -> yolo11s -> yolo11m) based on performance plateaus, not just tune hyperparameters.

**Web dashboard.** Extend the Streamlit monitor into a full web UI with run management, comparison views, and one-click deployment of the best model.

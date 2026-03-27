# AutoTrain

Autonomous ML training platform — train, evaluate, self-correct, repeat.

Point AutoTrain at your ML project, set a target metric, and walk away. An LLM agent autonomously modifies your training script, runs experiments, tracks results via git, and self-corrects until the target is hit or the budget runs out.

## Quick Start

```bash
pip install autotrain

autotrain run \
    --repo /path/to/your-model \
    --metric val_auc \
    --target 0.85 \
    --budget 4h
```

## Features

### Core
- **Autonomous training loop** — LLM agent proposes hyperparameter changes, runs training, evaluates metrics, keeps or reverts
- **Multi-provider LLM support** — Anthropic (Claude), DeepSeek (including deepseek-reasoner), Ollama (local models)
- **Remote GPU execution** — SSH + rsync to remote machines, training survives SSH disconnects via nohup + process groups
- **Git as experiment history** — every iteration is a commit, regressions are reverted automatically
- **Crash recovery** — journaled state machine resumes from any failure point
- **Code sandboxing** — file whitelist, dangerous pattern scanning, diff validation
- **Budget enforcement** — time, iteration, and API cost limits enforced both between and during training iterations
- **Structured logging** — JSON lines via structlog, queryable with jq

### Training Intelligence
- **Per-epoch metric streaming** — captures loss, mAP, precision, recall per epoch as training runs (YOLO and Keras patterns supported)
- **Training curve analysis** — agent receives formatted training curves with trend summaries to make informed decisions
- **Checkpoint recovery** — detects checkpoints after crashes, resumes via `AUTOTRAIN_RESUME_FROM` env var instead of restarting from scratch

### Monitoring Dashboard
- **Real-time Streamlit UI** — auto-refreshing dashboard at localhost:8501
- **Metric progress chart** — color-coded by outcome (improved/regressed/crashed) with target line
- **Training curves** — per-epoch charts with loss/score split, multi-iteration overlay
- **GPU resource monitoring** — live utilization, memory, temperature metrics with history chart (replaces `watch nvidia-smi`)
- **Iteration comparison** — side-by-side diff of any two iterations with metric delta and curve overlay
- **Agent reasoning log** — color-coded outcomes, hypothesis, reasoning, changes per iteration
- **Multi-run support** — sidebar run selector, cross-run history
- **Cost & budget tracker** — progress bars for time, iterations, and API cost with per-iteration rates

## Usage

### With a config file

```yaml
# autotrain.yaml
agent:
  provider: deepseek          # anthropic | deepseek | ollama
  model: deepseek-reasoner    # or deepseek-chat, claude-sonnet-4, etc.

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
  ssh_host: my-gpu-box
  ssh_remote_dir: /home/user/project
  ssh_setup_command: "~/.local/bin/uv sync"
  gpu_device: "0"

sandbox:
  writable_files:
    - train.py

notify:
  terminal: true
```

```bash
autotrain run --repo ./my-project --config autotrain.yaml -v
```

### With CLI flags

```bash
autotrain run \
    --repo ./my-project \
    --provider deepseek \
    --model deepseek-reasoner \
    --metric mAP --target 0.90 \
    --train-command "python train.py" \
    --budget 4h -v
```

### Monitoring

```bash
# In a separate terminal
autotrain monitor --repo ./my-project
```

Opens a Streamlit dashboard at localhost:8501 with live metrics, training curves, GPU resources, iteration history, agent reasoning, and budget tracking.

### Other commands

```bash
autotrain status --repo .     # Show current/recent run status
autotrain history --repo .    # Show experiment history
autotrain stop --repo .       # Stop a running process
```

## Architecture

```
autotrain run
    |
    v
[Agent Loop] ---> [LLM Provider] (Claude / DeepSeek / Ollama)
    |                    |
    |              propose changes
    |                    |
    v                    v
[Sandbox]          [Git Commit]
    |                    |
    v                    v
[SSH Executor] --> [Remote GPU] <-- [GPU Monitor]
    |                    |                |
    v                    v                v
[Epoch Stream]   [Metric Extract]  [gpu_snapshots DB]
    |                    |
    v                    v
[Evaluate] -----> improved? --> keep
    |              regressed? -> git revert
    |              crashed? ---> checkpoint resume / retry
    v
[Next Iteration] or [Done]
```

### Subsystems

| Subsystem | Purpose |
|-----------|---------|
| `agent/` | LLM client, prompt builder, response parser |
| `config/` | YAML schema, defaults |
| `core/` | Agent loop, state machine, budget tracker |
| `execution/` | SSH + local executors, rsync, process management |
| `experiment/` | Metric extraction, epoch parsing, git ops, sandbox |
| `monitor/` | Streamlit dashboard |
| `notify/` | Webhook + terminal notifications |
| `storage/` | SQLite DB, models, queries (WAL mode for concurrent access) |
| `util/` | Signal handling |
| `watchdog/` | Background health monitor, GPU metrics collection |

### Storage Schema (v4)

SQLite with WAL mode. Tables: `runs`, `iterations`, `metric_snapshots`, `epoch_metrics`, `gpu_snapshots`. Incremental migrations (v1 -> v2 -> v3 -> v4) applied automatically.

## LLM Providers

| Provider | Model | Cost/iter | Setup |
|----------|-------|-----------|-------|
| DeepSeek | deepseek-reasoner | ~$0.002 | `DEEPSEEK_API_KEY` env var |
| DeepSeek | deepseek-chat | ~$0.001 | `DEEPSEEK_API_KEY` env var |
| Anthropic | claude-haiku-4-5 | ~$0.003 | `ANTHROPIC_API_KEY` env var |
| Anthropic | claude-sonnet-4 | ~$0.02 | `ANTHROPIC_API_KEY` env var |
| Ollama | any local model | Free | Ollama running locally |

## Requirements

- Python 3.12+
- Git
- SSH access to GPU machine (for remote execution)

## Status

v0.1.0 — 4,936 lines across 43 modules. 103 tests passing.

## Future Improvements

- **SSH resilience** — reconnect-and-resume when SSH tail drops during long training runs; store partial output to avoid losing completed training results
- **Multi-file experiments** — allow the agent to modify multiple files per iteration (e.g. model architecture + training script)
- **Parallel experiments** — run multiple training configurations simultaneously on different GPUs
- **Model selection agent** — let the agent choose between model architectures (not just hyperparameters)
- **Web dashboard** — React/FastAPI replacement for Streamlit with real-time WebSocket updates
- **Experiment diffing** — show actual code diffs between iterations in the dashboard
- **Smart early stopping** — agent can request early termination of a training run based on streaming epoch metrics
- **Cross-run learning** — feed insights from previous runs into new run prompts
- **Notification integrations** — Slack, Discord, email alerts on target hit or budget exhaustion
- **Distributed training** — multi-GPU / multi-node support

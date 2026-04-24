# AutoTrain

Autonomous ML training platform — train, evaluate, self-correct, repeat.

Point AutoTrain at your ML project, set a target metric, and walk away. An LLM agent autonomously modifies your training script, runs experiments, tracks results via git, and self-corrects until the target is hit or the budget runs out.

Works with **any ML framework** — YOLO, Hugging Face Transformers, Keras, Lightning, scikit-learn, and more. AutoTrain detects your framework and adapts its strategy automatically.

## Quick Start

```bash
# Install from source
git clone <repo> && cd platform
uv sync  # or pip install -e .

# Run with a config file (recommended)
autotrain run --repo /path/to/your-model -v

# Run with CLI flags
autotrain run \
    --repo /path/to/your-model \
    --metric val_auc \
    --target 0.85 \
    --budget 4h -v
```

## Features

### Core
- **Autonomous training loop** — LLM agent proposes hyperparameter changes, runs training, evaluates metrics, keeps or reverts
- **Framework-agnostic** — auto-detects Ultralytics, Hugging Face, Keras, Lightning, scikit-learn, XGBoost, or generic PyTorch from imports; loads framework-specific tuning strategies
- **Multi-provider LLM support** — Anthropic (Claude), DeepSeek (including `deepseek-v4-pro` and `deepseek-v4-flash`), Ollama (local models)
- **Remote GPU execution** — SSH + rsync to remote machines, training survives SSH disconnects via nohup + process groups
- **Git as experiment history** — every iteration is a commit, regressions are reverted automatically
- **Crash recovery** — journaled state machine resumes from any failure point
- **Code sandboxing** — file whitelist, dangerous pattern scanning, diff validation, whitespace-tolerant matching fallback
- **Budget enforcement** — time, iteration, and API cost limits enforced both between and during training iterations
- **Startup validation** — fails fast if train script or writable files are missing
- **Structured logging** — JSON lines via structlog, queryable with jq

### Error Handling
- **Full output capture** — executor stores last 200 lines of training output in every result (not just streamed)
- **Structured metric errors** — `MetricResult` reports exactly why extraction failed: empty output, wrong metric name, no metrics found
- **Error feedback to agent** — crash errors stored in DB, shown in iteration history table, and passed to the agent as `last_error` on the next iteration so it can self-correct
- **SCRIPT_ERROR outcome** — distinguishes non-zero exit codes from "ran fine but no metric found"

### Training Intelligence
- **Framework detection** — scans imports in writable files to auto-detect ML framework (ultralytics, transformers, lightning, keras, sklearn, xgboost)
- **Strategy modules** — per-framework tuning playbooks (learning rate, augmentation, scheduler advice tailored to YOLO vs HuggingFace vs Keras vs generic)
- **Per-epoch metric streaming** — captures loss, mAP, precision, recall per epoch as training runs (YOLO, Keras, HuggingFace Trainer, Lightning, and generic `key=value` patterns)
- **Training curve analysis** — agent receives formatted training curves with trend summaries to make informed decisions
- **Checkpoint recovery** — detects checkpoints after crashes, resumes via `AUTOTRAIN_RESUME_FROM` env var instead of restarting from scratch

### Web Dashboard (React + FastAPI)
- **Single-process server** — FastAPI serves both REST API and React SPA at localhost:8000
- **Real-time updates** — WebSocket push with direct cache injection (`setQueryData`), no polling lag for GPU/run data
- **Remote metrics agent** — standalone pynvml-based daemon on GPU machine pushes metrics every 1s via WebSocket (~20ms latency vs 3-8s with SSH polling)
- **Metric progress chart** — Recharts scatter plot color-coded by outcome (improved/regressed/crashed) with target + best reference lines
- **Training curves** — per-epoch charts with loss/score split, multi-iteration overlay with iteration selector
- **GPU resource monitoring** — 4 live metric cards (utilization, memory %, VRAM, temperature) + time-series chart, sub-second updates via remote agent, capped at 300 points with downsampling for performance
- **Iteration comparison** — two-iteration picker with metric/duration/cost deltas + epoch curve overlay
- **Agent reasoning log** — expandable accordion with color-coded outcomes, hypothesis, reasoning, changes per iteration
- **Iteration table** — sortable table with outcome badges, metric values, duration, cost, error messages
- **Multi-run support** — sidebar run selector with status markers
- **Budget tracker** — progress bars for time, iterations, and API cost with per-iteration rates (parses budget limits from config)
- **Versioned API** — `/api/v1/` prefix with 501 placeholders for Phase 2/3 features (experiment diffing, early stopping, file browser)
- **Legacy Streamlit UI** — still available via `autotrain monitor` as fallback

## Usage

### With a config file

Place `autotrain.yaml` in the project root — it's auto-detected. Or pass `--config path/to/config.yaml` explicitly.

```yaml
# autotrain.yaml
agent:
  provider: deepseek          # anthropic | deepseek | ollama
  model: deepseek-v4-pro      # or deepseek-v4-flash, claude-sonnet-4, etc.

metric:
  name: mAP
  target: 0.90
  direction: maximize          # maximize | minimize

budget:
  time_seconds: 14400         # 4 hours
  max_iterations: 50
  api_dollars: 2.00
  experiment_timeout_seconds: 1800  # 30 min per training run (default)

execution:
  train_command: ".venv/bin/python train.py"
  mode: ssh                    # ssh | local
  ssh_host: my-gpu-box
  ssh_remote_dir: /home/user/project
  ssh_setup_command: "~/.local/bin/uv sync"  # runs once before first training
  gpu_device: "0"              # optional — omit to let framework choose
  dashboard_url: "ws://192.168.1.14:8000"    # enables remote metrics agent
  rsync_excludes:              # protect remote-only dirs from --delete
    - ".venv"
    - "__pycache__"
    - ".git"
    - ".autotrain"
    - "*.pt"
    - "*.pth"
    - "*.ckpt"
    - "*.onnx"
    - "*.h5"
    - "*.safetensors"

sandbox:
  writable_files:
    - train.py

notify:
  terminal: true
```

```bash
autotrain run --repo ./my-project -v
```

### With CLI flags

```bash
autotrain run \
    --repo ./my-project \
    --provider deepseek \
    --model deepseek-v4-pro \
    --metric mAP --target 0.90 \
    --train-command ".venv/bin/python train.py" \
    --ssh-host my-gpu-box \
    --ssh-remote-dir /home/user/project \
    --dashboard-url ws://192.168.1.14:8000 \
    --budget 4h -v
```

### Monitoring

```bash
# Web dashboard (React + FastAPI) — recommended
autotrain dashboard --repo ./my-project

# With remote GPU agent (sub-second GPU updates)
autotrain dashboard --repo ./my-project --host 0.0.0.0
autotrain run --repo ./my-project --dashboard-url ws://YOUR_LAN_IP:8000 -v

# Legacy Streamlit dashboard
autotrain monitor --repo ./my-project
```

The web dashboard opens at localhost:8000 with live WebSocket updates, metric charts, training curves, GPU resources, iteration comparison, agent reasoning, and budget tracking. When `--dashboard-url` is set, a remote metrics agent is auto-deployed to the GPU machine for sub-second GPU monitoring.

### All commands

```bash
autotrain run --repo .        # Start autonomous training
autotrain dashboard --repo .  # Web dashboard (React + FastAPI)
autotrain status --repo .     # Show current/recent run status
autotrain history --repo .    # Show experiment history
autotrain stop --repo .       # Stop a running process
autotrain monitor --repo .    # Legacy Streamlit dashboard
```

## Architecture

```
autotrain run
    |
    v
[Framework Detect] --> ultralytics / huggingface / keras / generic
    |
    v
[Agent Loop] ---> [LLM Provider] (Claude / DeepSeek / Ollama)
    |                    |
    |              propose changes (framework-specific strategy)
    |                    |
    v                    v
[Sandbox]          [Git Commit]
    |                    |
    v                    v
[SSH Executor] --> [Remote GPU] <-- [Remote Agent (pynvml)]
    |                    |                |
    v                    v                v (WebSocket push)
[Epoch Stream]   [Metric Extract]  [Dashboard Server]
    |                    |
    v                    v
[Evaluate] -----> improved? --> keep
    |              regressed? -> git revert
    |              crashed? ---> error stored + fed to agent + checkpoint resume
    |              script_error? -> exit code + output captured for agent
    v
[Next Iteration] or [Done]
```

### Subsystems

| Subsystem | Purpose |
|-----------|---------|
| `agent/` | LLM client, prompt builder, response parser, framework detector, strategy modules |
| `config/` | YAML schema, defaults, cross-framework checkpoint patterns |
| `core/` | Agent loop, state machine, budget tracker, startup validation |
| `execution/` | SSH + local executors, rsync, process management, stdout capture |
| `experiment/` | Metric extraction (MetricResult), epoch parsing (multi-framework), git ops, sandbox (fuzzy matching) |
| `dashboard/` | React + FastAPI web dashboard (API, WebSocket, agent relay, SPA) |
| `remote_agent/` | Standalone GPU metrics agent (pynvml, log tailing, WebSocket push) |
| `monitor/` | Legacy Streamlit dashboard |
| `notify/` | Webhook + terminal notifications |
| `storage/` | SQLite DB, models, queries (WAL mode for concurrent access) |
| `util/` | Signal handling, shutdown callbacks |
| `watchdog/` | Background health monitor, GPU metrics collection (SSH fallback) |

### Storage Schema (v4)

SQLite with WAL mode. Tables: `runs`, `iterations`, `metric_snapshots`, `epoch_metrics`, `gpu_snapshots`. Incremental migrations (v1 -> v2 -> v3 -> v4) applied automatically.

## LLM Providers

| Provider | Model | Cost/iter | Setup |
|----------|-------|-----------|-------|
| DeepSeek | deepseek-v4-pro | ~$0.016 | `DEEPSEEK_API_KEY` env var |
| DeepSeek | deepseek-v4-flash | ~$0.001 | `DEEPSEEK_API_KEY` env var |
| Anthropic | claude-haiku-4-5 | ~$0.003 | `ANTHROPIC_API_KEY` env var |
| Anthropic | claude-sonnet-4 | ~$0.02 | `ANTHROPIC_API_KEY` env var |
| Ollama | any local model | Free | Ollama running locally |

## Supported Frameworks

AutoTrain auto-detects your ML framework from imports and loads tailored tuning strategies:

| Framework | Detection | Strategy |
|-----------|-----------|----------|
| Ultralytics (YOLO) | `from ultralytics` | lr0, mosaic, imgsz, model size, cos_lr |
| Hugging Face Transformers | `from transformers` | learning_rate, warmup, weight_decay, fp16/bf16, gradient accumulation |
| Keras / TensorFlow | `from keras` / `import tensorflow` | optimizer, callbacks, dropout, augmentation |
| PyTorch Lightning | `import lightning` | (uses generic strategy) |
| scikit-learn | `from sklearn` | (uses generic strategy) |
| XGBoost | `import xgboost` | (uses generic strategy) |
| Generic | fallback | learning rate, batch size, epochs, regularization |

## Requirements

- Python 3.12+
- Git
- SSH access to GPU machine (for remote execution)
- `websockets` in project venv on GPU machine (auto-installed by remote agent)
- Optional: `pynvml` on GPU machine (for GPU metrics — falls back to nvidia-smi)

## Status

v0.4.0 — 6,524 lines Python + 1,363 lines TypeScript across 56 modules. 125 tests passing.

## Future Improvements

- **Multi-file experiments** — allow the agent to modify multiple files per iteration (e.g. model architecture + training script)
- **Parallel experiments** — run multiple training configurations simultaneously on different GPUs
- **Model selection agent** — let the agent choose between model architectures (not just hyperparameters)
- **Experiment diffing** — show actual code diffs between iterations in the dashboard (API placeholder ready)
- **Smart early stopping** — agent can request early termination of a training run based on streaming epoch metrics (API placeholder ready)
- **Cross-run learning** — feed insights from previous runs into new run prompts
- **Notification integrations** — Slack, Discord, email alerts on target hit or budget exhaustion
- **Distributed training** — multi-GPU / multi-node support

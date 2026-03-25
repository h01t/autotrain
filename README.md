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

- **Autonomous training loop** — LLM agent proposes changes, runs training, evaluates metrics, keeps or reverts
- **Multi-provider LLM support** — Anthropic (Claude), DeepSeek, Ollama (local models)
- **Remote GPU execution** — SSH + rsync to remote machines, training survives SSH disconnects via nohup
- **Git as experiment history** — every iteration is a commit, regressions are reverted
- **Crash recovery** — journaled state machine resumes from any failure point
- **Code sandboxing** — file whitelist, dangerous pattern scanning, diff validation
- **Budget enforcement** — time, iteration, and API cost limits
- **Monitoring dashboard** — Streamlit UI with live metric charts, iteration history, agent reasoning log
- **Structured logging** — JSON lines via structlog, queryable with jq

## Usage

### With a config file

```yaml
# autotrain.yaml
agent:
  provider: deepseek          # anthropic | deepseek | ollama
  model: deepseek-chat

metric:
  name: mAP
  target: 0.80
  direction: maximize

budget:
  time_seconds: 14400         # 4 hours
  max_iterations: 50
  api_dollars: 2.00

execution:
  train_command: ".venv/bin/python train.py"
  mode: ssh
  ssh_host: my-gpu-box
  ssh_remote_dir: /home/user/project

sandbox:
  writable_files:
    - train.py
```

```bash
autotrain run --repo ./my-project --config autotrain.yaml -v
```

### With CLI flags

```bash
autotrain run \
    --repo ./my-project \
    --provider deepseek \
    --model deepseek-chat \
    --metric mAP --target 0.80 \
    --train-command "python train.py" \
    --budget 4h -v
```

### Monitoring

```bash
# In a separate terminal
autotrain monitor --repo ./my-project
```

Opens a Streamlit dashboard at localhost:8501 with:
- Metric progress chart with target line
- Iteration history table
- Agent reasoning log (expandable per iteration)
- Cost and budget tracker

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
[SSH Executor] --> [Remote GPU]
    |                    |
    v                    v
[Metric Extract]   [Evaluate]
    |                    |
    |   improved? -----> keep
    |   regressed? ----> git revert
    |   crashed? -------> retry
    |
    v
[Next Iteration] or [Done]
```

## LLM Providers

| Provider | Model | Cost/iter | Setup |
|----------|-------|-----------|-------|
| DeepSeek | deepseek-chat | ~$0.001 | `DEEPSEEK_API_KEY` env var |
| Anthropic | claude-haiku-4-5 | ~$0.003 | `ANTHROPIC_API_KEY` env var |
| Anthropic | claude-sonnet-4 | ~$0.02 | `ANTHROPIC_API_KEY` env var |
| Ollama | any local model | Free | Ollama running locally |

## Requirements

- Python 3.12+
- Git
- SSH access to GPU machine (for remote execution)

## Status

v0.1.0 — Core platform complete. 106 tests passing.

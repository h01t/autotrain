---
title: "AutoTrain: Autonomous ML Training Platform"
subtitle: "Version 0.4.0 | March 2026"
geometry: margin=2.5cm
fontsize: 11pt
---

# The Idea

Training machine learning models is an iterative, time-consuming process. A researcher adjusts hyperparameters, runs training, waits for results, analyzes metrics, and repeats. This cycle can take days or weeks of manual work.

AutoTrain replaces this manual loop with an autonomous agent. Given a training script, a target metric, and a compute budget, AutoTrain uses an LLM to propose code changes, executes training on remote GPU hardware, evaluates results, and iterates --- all without human intervention.

The core insight: an LLM can read a training script, understand experiment history, and propose informed hyperparameter changes just like a researcher would --- but it can run 24/7 without fatigue or context-switching.

**Framework-agnostic.** AutoTrain detects your ML framework from imports (Ultralytics, Hugging Face, Keras, Lightning, scikit-learn, XGBoost, or generic PyTorch) and loads tailored tuning strategies automatically.

# Use Case: Vehicle Detection

Our test case uses YOLO object detection trained on the KITTI autonomous driving dataset. The goal: push mAP (mean Average Precision) from a baseline to >= 0.90.

**Setup.** Model: YOLOv11n (nano) with pretrained weights. Dataset: KITTI (vehicles, pedestrians, cyclists). Compute: NVIDIA RTX 3060 Ti workstation (SSH remote). Agent: DeepSeek Chat (cost-effective LLM). Budget: 4 hours, 50 iterations, $2.00 API cap.

**Results (First Run):**

| Metric     | Baseline | After AutoTrain | Target |
|------------|----------|-----------------|--------|
| mAP@0.5    | ---      | **0.8384**      | 0.80   |
| mAP@0.5-0.95 | ---   | 0.5984          | ---    |
| Precision  | ---      | 0.8349          | ---    |
| Recall     | ---      | 0.7699          | ---    |

Target hit in **2 iterations**, ~35 minutes total training time. The agent reduced learning rate (0.01 -> 0.008) and batch size (16 -> 8). Total API cost: < $0.01.

# How It Works

When `autotrain run` is started, the following sequence executes:

```
YOUR MACHINE                                         REMOTE GPU
+-------------------+
| 1. INITIALIZE     |
|  - Framework detect|
|  - Git init/branch |
|  - SQLite DB       |
|  - rsync + setup  |-------> SSH -------> mkdir + uv sync
+-------------------+
         |
+-------------------+
| 2. CALL AGENT     |
|  - Build prompt   |-------> LLM API (DeepSeek/Claude)
|  - Load strategy  |         (framework-specific playbook)
|  - Parse response |
+-------------------+
         |
+-------------------+
| 3. VALIDATE       |
|  - Sandbox check  |  Only whitelisted files
|  - Search/replace |  Exact match + whitespace-tolerant fallback
+-------------------+
         |
+-------------------+
| 4. APPLY + SYNC   |
|  - Edit files     |
|  - Git commit     |
|  - rsync to GPU   |-------> rsync -------> Updated train.py
+-------------------+
         |
+-------------------+
| 5. TRAIN          |
|  - SSH execute    |-------> SSH -------> setsid python train.py
|  - Tail logs      |         stdout <---- Training output
|  - Stream epochs  |         (captures last 200 lines)
|  - Extract metric |
+-------------------+
         |
+-------------------+
| 6. EVALUATE       |
|  - Compare to best|
|  - Target hit?    |-------> YES: Stop (completed)
|  - Regressed?     |-------> Revert commit
|  - Crashed?       |-------> Store error + feed to agent
|  - Script error?  |-------> Capture exit code + output
+-------------------+
         |
    Loop back to 2
```

# Key Design Decisions

**Framework Detection.** AutoTrain scans imports in writable files to detect the ML framework (ultralytics, transformers, keras, lightning, sklearn, xgboost). It loads a framework-specific strategy module with tailored hyperparameter tuning advice. Falls back to generic strategies for unknown frameworks.

**Structured Error Handling.** Every crashed iteration stores a detailed error message: exit code, metric extraction failure reason, and last 30 lines of training output. This error is shown to the agent on the next iteration so it can self-correct. `SCRIPT_ERROR` (non-zero exit) is distinguished from `CRASHED` (ran but no metric found).

**Sandbox Safety.** The agent can only modify files explicitly listed in `writable_files`. All changes go through search/replace validation --- the agent must reference exact text from the current file. A whitespace-tolerant fallback matches when trailing whitespace drifts between iterations. Rejection messages include the actual file content so the agent can correct its search text.

**Git-Based Rollback.** Every iteration is a git commit. If a change makes metrics worse, the commit is reverted. The experiment branch preserves the full history of what was tried.

**Process Isolation.** Training runs in a `setsid` process group on the remote machine. If SSH drops (laptop sleeps, network hiccup), training continues. AutoTrain reconnects and resumes tailing output.

**Multi-Provider LLM.** Supports Anthropic (Claude), DeepSeek, and Ollama. The vehicle detection run used DeepSeek Chat at $0.27/1M input tokens --- orders of magnitude cheaper than manual researcher time.

**Budget Enforcement.** Hard limits on wall-clock time, iteration count, and API spend. Training cannot run away. Budget can expire mid-training --- the state machine handles this cleanly.

**Startup Validation.** Fails fast with clear messages if the train script doesn't exist (with default command) or no writable files are found.

# Architecture

```
src/autotrain/          6,524 lines across 56 modules
  agent/                LLM client, prompt builder, response parser,
                          framework detector, strategy modules
  config/               YAML config loader, Pydantic schemas
  core/                 Agent loop, budget tracker, state machine
  execution/            Local and SSH executors (stdout capture)
  experiment/           Git ops, metric extraction (MetricResult),
                          sandbox (fuzzy match), epoch parsing
  dashboard/            React + FastAPI web dashboard (SPA + API + WS)
  remote_agent/         Standalone pynvml GPU metrics agent
  monitor/              Legacy Streamlit dashboard
  notify/               Terminal + webhook notifications
  storage/              SQLite database, queries
  util/                 Logging, signals, shutdown callbacks
  watchdog/             Disk/GPU health monitoring

frontend/               1,363 lines TypeScript/React
  20 components         Recharts charts, WebSocket hooks, memoized
                          GPU chart with downsampling
```

**Dependencies:** 8 runtime packages (anthropic, click, pydantic, pyyaml, structlog, requests, streamlit, plotly). No heavy ML frameworks --- AutoTrain orchestrates, it doesn't train.

**Dashboard.** React + FastAPI SPA at localhost:8000 with WebSocket push. GPU chart capped at 300 points with downsampling for performance. Remote pynvml agent pushes metrics every 1s. 125 tests passing.

**Monitoring.** A Streamlit dashboard (`autotrain monitor`) is available as legacy fallback. The React dashboard is recommended for production use.

# Supported Frameworks

| Framework | Auto-Detection | Strategy Module |
|-----------|---------------|-----------------|
| Ultralytics (YOLO) | `from ultralytics` | lr0, mosaic, imgsz, model size, cos_lr |
| Hugging Face | `from transformers` | learning_rate, warmup, weight_decay, fp16 |
| Keras / TensorFlow | `from keras` | optimizer, callbacks, dropout, augmentation |
| Lightning | `import lightning` | generic strategy |
| scikit-learn | `from sklearn` | generic strategy |
| XGBoost | `import xgboost` | generic strategy |
| Generic PyTorch | `import torch` | LR, batch size, epochs, regularization |

Epoch parsing supports: YOLO training/validation lines, Keras `Epoch N/M`, HuggingFace Trainer dict output, Lightning/tqdm format, and generic `key=value` patterns.

# Configuration

A single YAML file controls the entire run:

```yaml
agent:
  provider: deepseek          # anthropic | deepseek | ollama
  model: deepseek-v4-flash
  temperature: 0.3

metric:
  name: mAP
  target: 0.90
  direction: maximize

budget:
  time_seconds: 14400         # 4 hours
  max_iterations: 50
  api_dollars: 2.00
  experiment_timeout_seconds: 1800  # 30 min default

execution:
  train_command: ".venv/bin/python train.py"
  mode: ssh
  ssh_host: blackbox
  ssh_remote_dir: /home/holt/Dev/object-det
  ssh_setup_command: "~/.local/bin/uv sync"
  gpu_device: "0"             # optional — omit to let framework choose
  dashboard_url: "ws://192.168.1.14:8000"

sandbox:
  writable_files:
    - train.py
```

The `ssh_setup_command` runs once before the first training iteration --- installing dependencies on a fresh remote machine with zero manual SSH work. The `dashboard_url` enables the remote pynvml agent for sub-second GPU monitoring. `gpu_device` is optional --- when omitted, `CUDA_VISIBLE_DEVICES` is not set, letting the framework choose.

# Future Improvements

**Multi-file experiments.** Currently the agent modifies a single training script. Supporting model architecture files, data augmentation configs, and custom loss functions would give the agent more leverage.

**Parallel experiments.** Run multiple hypotheses simultaneously across multiple GPUs or machines. Compare results and keep the best.

**Smarter agent context.** Feed training curves (loss over epochs) back to the agent, not just final metrics. This would let it diagnose overfitting, learning rate issues, and convergence problems more precisely.

**Model selection.** Let the agent switch between model architectures (yolo11n -> yolo11s -> yolo11m) based on performance plateaus, not just tune hyperparameters.

**Cross-run learning.** Feed insights from previous runs into new run prompts so the agent doesn't repeat failed approaches across sessions.

**Distributed training.** Multi-GPU / multi-node support for larger models and datasets.

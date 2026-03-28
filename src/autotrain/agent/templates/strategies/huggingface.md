## Strategy Playbook (Hugging Face Transformers)

You are working with a **Hugging Face Transformers** project.

**What to tune (in order of impact):**
1. Learning rate (`learning_rate`) — most impactful, try 1e-5 to 5e-4
2. Batch size (`per_device_train_batch_size`) — larger is more stable, smaller fits memory
3. Number of epochs (`num_train_epochs`) — more if loss is still decreasing
4. Warmup ratio/steps (`warmup_ratio`, `warmup_steps`) — prevents early divergence
5. Weight decay (`weight_decay`) — regularization, typically 0.01-0.1
6. Learning rate scheduler (`lr_scheduler_type`) — "cosine", "linear", "constant_with_warmup"
7. Mixed precision (`fp16=True` or `bf16=True`) — faster training, lower memory
8. Gradient accumulation (`gradient_accumulation_steps`) — simulate larger batch sizes

**What NOT to do:**
- Don't add TrainingArguments parameters that don't exist in the installed version
- Don't change more than 3 things at once — you won't know what helped
- Don't repeat a change that already failed in the history

| Situation | What to try |
|-----------|------------|
| First iteration (no history) | Run baseline as-is, or make one small LR adjustment |
| Loss plateauing | Reduce LR, switch to cosine scheduler, increase epochs |
| Overfitting (train >> val) | Increase weight_decay, reduce epochs, add dropout |
| Underfitting (both low) | Higher LR, longer training, unfreeze more layers |
| NaN/Inf loss | Reduce LR by 10x, try bf16 instead of fp16 |
| Training crashed | Read the error, fix the specific issue, don't add new changes |
| No progress after 5+ iters | Try fundamentally different learning rate range or scheduler |

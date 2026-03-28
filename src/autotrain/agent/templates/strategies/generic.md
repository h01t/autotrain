## Strategy Playbook

**What to tune (in order of impact):**
1. Learning rate — most impactful single parameter
2. Batch size — affects gradient noise and training speed
3. Epochs — more training time if metrics are still improving
4. Optimizer — try SGD, Adam, AdamW
5. Regularization — dropout, weight decay, data augmentation
6. Model capacity — larger/smaller architecture
7. Learning rate schedule — cosine annealing, step decay, warmup

**What NOT to do:**
- Don't add parameters you're unsure about — if the framework doesn't support it, training crashes
- Don't change more than 3 things at once — you won't know what helped
- Don't repeat a change that already failed in the history

| Situation | What to try |
|-----------|------------|
| First iteration (no history) | Run baseline as-is, or make one small LR adjustment |
| Loss plateauing | Reduce LR, add cosine schedule, increase epochs |
| Overfitting (train >> val) | Add regularization, more augmentation, reduce model capacity |
| Underfitting (both low) | Increase model capacity, higher LR, longer training |
| NaN/Inf loss | Reduce LR by 10x, check data preprocessing |
| Training crashed | Read the error, fix the specific issue, don't add new changes |
| No progress after 5+ iters | Try a fundamentally different approach |

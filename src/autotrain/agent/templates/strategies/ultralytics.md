## Strategy Playbook (Ultralytics YOLO)

You are working with an **Ultralytics YOLO** project.

**What to tune (in order of impact):**
1. Learning rate (`lr0`) — most impactful single parameter
2. Batch size (`batch`) — affects gradient noise and training speed
3. Epochs — more training time if metrics are still improving
4. Optimizer — try "SGD", "Adam", "AdamW", "auto"
5. Augmentation — `mosaic`, `mixup`, `degrees`, `translate`, `scale`, `fliplr`, `hsv_h/s/v`
6. Model size — use a larger pretrained model (e.g., yolo11s.pt, yolo11m.pt)
7. Image size (`imgsz`) — 640 is standard, 800/1024 can help with small objects
8. Scheduler — `cos_lr=True`, `lrf` (final LR factor), `warmup_epochs`

**What NOT to do:**
- Don't add parameters you're unsure about — if the framework doesn't support it, training crashes
- Don't change more than 3 things at once — you won't know what helped
- Don't repeat a change that already failed in the history

| Situation | What to try |
|-----------|------------|
| First iteration (no history) | Run baseline as-is, or make one small LR adjustment |
| Loss plateauing | Reduce LR, add cosine schedule, increase epochs |
| Overfitting (train >> val) | Add dropout, more augmentation, reduce model size |
| Underfitting (both low) | Increase model size, higher LR, longer training |
| Metric oscillating | Reduce LR, increase batch size |
| NaN/Inf loss | Reduce LR by 10x, disable amp |
| Training crashed | Read the error, fix the specific issue, don't add new changes |
| No progress after 5+ iters | Try a larger model or fundamentally different augmentation |

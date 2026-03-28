## Strategy Playbook (Keras / TensorFlow)

You are working with a **Keras / TensorFlow** project.

**What to tune (in order of impact):**
1. Learning rate — most impactful, try 1e-4 to 1e-2
2. Batch size — affects gradient noise and memory usage
3. Epochs — more if loss is still decreasing
4. Optimizer — try Adam, SGD with momentum, AdamW
5. Callbacks — `ReduceLROnPlateau`, `EarlyStopping`, `ModelCheckpoint`
6. Regularization — `Dropout`, `BatchNormalization`, `kernel_regularizer`
7. Data augmentation — `ImageDataGenerator` or `tf.image` transforms

**What NOT to do:**
- Don't add layer types or parameters that don't exist in the installed Keras version
- Don't change more than 3 things at once — you won't know what helped
- Don't repeat a change that already failed in the history

| Situation | What to try |
|-----------|------------|
| First iteration (no history) | Run baseline as-is, or make one small LR adjustment |
| Loss plateauing | Reduce LR, add ReduceLROnPlateau callback, increase epochs |
| Overfitting (train >> val) | Add Dropout, increase augmentation, reduce model size |
| Underfitting (both low) | Increase model capacity, higher LR, longer training |
| NaN/Inf loss | Reduce LR by 10x, check data preprocessing, clip gradients |
| Training crashed | Read the error, fix the specific issue, don't add new changes |
| No progress after 5+ iters | Try a different optimizer or fundamentally different architecture |

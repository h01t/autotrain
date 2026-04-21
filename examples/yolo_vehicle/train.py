"""AutoTrain wrapper — trains YOLO and prints metrics as JSON.

Supports checkpoint resume via AUTOTRAIN_RESUME_FROM env var.
When AutoTrain detects a crash with an existing checkpoint, it sets
this variable so training resumes from where it left off instead of
restarting from scratch.
"""

import os

from ultralytics import YOLO


def main():
    # Check if we should resume from a checkpoint
    resume_from = os.environ.get("AUTOTRAIN_RESUME_FROM")
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        model = YOLO(resume_from)
    else:
        model = YOLO("yolo11n.pt")

    results = model.train(
        data="kitti.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device="cuda",
        project="outputs/training",
        name="autotrain",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        patience=10,
        save=True,
        plots=True,
        amp=True,
    )

    # Print metrics as JSON for autotrain extraction
    metrics = results.results_dict
    map50 = metrics.get("metrics/mAP50(B)", 0.0)
    map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
    precision = metrics.get("metrics/precision(B)", 0.0)
    recall = metrics.get("metrics/recall(B)", 0.0)

    print(
        f'{{"mAP": {map50:.4f}, "mAP50_95": {map50_95:.4f}, '
        f'"precision": {precision:.4f}, "recall": {recall:.4f}}}'
    )


if __name__ == "__main__":
    main()

"""Tests for metric extraction."""

from __future__ import annotations

from autotrain.experiment.metrics import (
    extract_metric_from_output,
    extract_metrics_from_line,
)


class TestExtractMetricsFromLine:
    def test_json_line(self):
        line = '{"epoch": 5, "val_auc": 0.82, "loss": 0.15}'
        result = extract_metrics_from_line(line)
        assert result["val_auc"] == 0.82
        assert result["loss"] == 0.15
        assert result["epoch"] == 5.0

    def test_key_equals_value(self):
        line = "Epoch 10 | val_auc=0.85 loss=0.12"
        result = extract_metrics_from_line(line)
        assert result["val_auc"] == 0.85
        assert result["loss"] == 0.12

    def test_key_colon_value(self):
        line = "val_bpb: 0.997900"
        result = extract_metrics_from_line(line)
        assert result["val_bpb"] == 0.9979

    def test_regex_pattern(self):
        line = "mAP@0.5: 0.782"
        result = extract_metrics_from_line(
            line, target_metric="mAP@0.5", regex_pattern=r"mAP@0\.5:\s+([\d.]+)",
        )
        assert result["mAP@0.5"] == 0.782

    def test_empty_line(self):
        assert extract_metrics_from_line("") == {}

    def test_no_metrics(self):
        assert extract_metrics_from_line("Training started...") == {}

    def test_json_non_numeric_ignored(self):
        line = '{"status": "running", "val_auc": 0.85}'
        result = extract_metrics_from_line(line)
        assert "status" not in result
        assert result["val_auc"] == 0.85


class TestExtractMetricFromOutput:
    def test_returns_last_value(self):
        output = """Epoch 1 | val_auc=0.60
Epoch 2 | val_auc=0.70
Epoch 3 | val_auc=0.75"""
        result = extract_metric_from_output(output, "val_auc")
        assert result == 0.75

    def test_returns_none_when_missing(self):
        output = "Training complete. No metrics."
        assert extract_metric_from_output(output, "val_auc") is None

    def test_with_regex(self):
        output = """Step 100: mAP@0.5: 0.50
Step 200: mAP@0.5: 0.65
Step 300: mAP@0.5: 0.78"""
        result = extract_metric_from_output(
            output, "mAP@0.5", regex_pattern=r"mAP@0\.5:\s+([\d.]+)",
        )
        assert result == 0.78

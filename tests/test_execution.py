"""Tests for execution layer."""

from __future__ import annotations

import pytest

from autotrain.execution.local import LocalExecutor


class TestLocalExecutor:
    def test_simple_command(self, tmp_path):
        executor = LocalExecutor(working_dir=tmp_path)
        executor.setup()

        lines = list(executor.execute("echo 'hello world'", timeout_seconds=10))
        assert any("hello world" in line for line in lines)

        result = executor.get_result()
        assert result.exit_code == 0
        assert not result.was_timeout

    def test_multiline_output(self, tmp_path):
        executor = LocalExecutor(working_dir=tmp_path)
        script = tmp_path / "test.py"
        script.write_text("for i in range(5): print(f'line {i}')\n")

        lines = list(executor.execute(f"python {script}", timeout_seconds=10))
        assert len(lines) == 5
        assert lines[0] == "line 0"
        assert lines[4] == "line 4"

    def test_nonzero_exit_code(self, tmp_path):
        executor = LocalExecutor(working_dir=tmp_path)
        list(executor.execute("exit 42", timeout_seconds=10))
        result = executor.get_result()
        assert result.exit_code == 42

    def test_timeout_kills_process(self, tmp_path):
        executor = LocalExecutor(working_dir=tmp_path)
        script = tmp_path / "slow.py"
        script.write_text("import time\nwhile True:\n    print('tick'); time.sleep(0.1)\n")

        lines = []
        for line in executor.execute(f"python -u {script}", timeout_seconds=1):
            lines.append(line)

        result = executor.get_result()
        assert result.was_timeout
        assert len(lines) > 0  # Got some output before timeout

    def test_env_passed(self, tmp_path):
        executor = LocalExecutor(working_dir=tmp_path)
        lines = list(executor.execute(
            "python -c \"import os; print(os.environ.get('TEST_VAR', 'missing'))\"",
            timeout_seconds=10,
            env={"TEST_VAR": "hello"},
        ))
        assert lines[0] == "hello"

    def test_is_process_alive(self, tmp_path):
        executor = LocalExecutor(working_dir=tmp_path)
        assert not executor.is_process_alive()

        # Start a process that prints periodically (so we can read a line)
        script = tmp_path / "long.py"
        script.write_text(
            "import time, sys\n"
            "print('started', flush=True)\n"
            "time.sleep(30)\n"
        )
        gen = executor.execute(f"python -u {script}", timeout_seconds=60)
        # Read first line to ensure process is running
        first_line = next(gen)
        assert first_line == "started"
        assert executor.is_process_alive()

        # Kill it
        executor.kill()
        assert not executor.is_process_alive()
        result = executor.get_result()
        assert result.was_killed

    def test_working_dir_not_found(self, tmp_path):
        executor = LocalExecutor(working_dir=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            executor.setup()

    def test_result_before_execute_raises(self):
        executor = LocalExecutor()
        with pytest.raises(RuntimeError, match="No execution result"):
            executor.get_result()

    def test_metrics_in_output(self, tmp_path):
        """Simulate a training script that outputs metrics."""
        executor = LocalExecutor(working_dir=tmp_path)
        script = tmp_path / "train.py"
        script.write_text(
            'import json\n'
            'for epoch in range(3):\n'
            '    m = {"epoch": epoch, "val_auc": 0.7 + epoch*0.05, "loss": 0.5 - epoch*0.1}\n'
            '    metrics = m\n'
            '    print(json.dumps(metrics))\n'
        )

        lines = list(executor.execute(f"python {script}", timeout_seconds=10))
        assert len(lines) == 3

        # Verify we can parse metrics from the output
        import json
        last_metrics = json.loads(lines[-1])
        assert abs(last_metrics["val_auc"] - 0.8) < 1e-9
        assert last_metrics["epoch"] == 2

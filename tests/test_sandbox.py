"""Tests for code sandboxing."""

from __future__ import annotations

from autotrain.config.schema import SandboxConfig
from autotrain.experiment.sandbox import (
    FileChange,
    format_rejection_message,
    validate_changes,
)


class TestValidateChanges:
    def setup_method(self):
        self.config = SandboxConfig(
            writable_files=["train.py", "config.yaml"],
            forbidden_patterns=[r"subprocess\.", r"os\.system", r"exec\(", r"eval\("],
            max_file_size_bytes=10_000,
            max_changes_per_iteration=3,
        )

    def test_valid_change(self):
        changes = [
            FileChange(file="train.py", action="replace", search="lr=0.001", replace="lr=0.01"),
        ]
        result = validate_changes(
            changes, self.config,
            current_files={"train.py": "lr=0.001\nepochs=10\n"},
        )
        assert result.is_valid

    def test_file_not_in_whitelist(self):
        changes = [
            FileChange(file="model.py", action="replace", search="x", replace="y"),
        ]
        result = validate_changes(changes, self.config)
        assert not result.is_valid
        assert "whitelist" in result.errors[0]

    def test_too_many_changes(self):
        changes = [
            FileChange(file="train.py", action="create", content="x")
            for _ in range(5)
        ]
        result = validate_changes(changes, self.config)
        assert not result.is_valid
        assert "Too many changes" in result.errors[0]

    def test_forbidden_pattern_subprocess(self):
        changes = [
            FileChange(
                file="train.py", action="full_rewrite",
                content="import subprocess\nsubprocess.run(['rm', '-rf', '/'])",
            ),
        ]
        result = validate_changes(changes, self.config)
        assert not result.is_valid
        assert any("subprocess" in e for e in result.errors)

    def test_forbidden_pattern_eval(self):
        changes = [
            FileChange(
                file="train.py", action="full_rewrite",
                content="result = eval('2+2')",
            ),
        ]
        result = validate_changes(changes, self.config)
        assert not result.is_valid

    def test_file_too_large(self):
        changes = [
            FileChange(
                file="train.py", action="full_rewrite",
                content="x" * 20_000,
            ),
        ]
        result = validate_changes(changes, self.config)
        assert not result.is_valid
        assert any("bytes" in e for e in result.errors)

    def test_diff_validation_catches_added_dangerous_code(self):
        current = "lr = 0.001\n"
        new_content = "lr = 0.001\nimport subprocess\nsubprocess.run(['ls'])\n"
        changes = [
            FileChange(
                file="train.py", action="full_rewrite", content=new_content,
            ),
        ]
        result = validate_changes(
            changes, self.config,
            current_files={"train.py": current},
        )
        assert not result.is_valid

    def test_search_text_not_found(self):
        changes = [
            FileChange(
                file="train.py", action="replace",
                search="nonexistent", replace="something",
            ),
        ]
        result = validate_changes(
            changes, self.config,
            current_files={"train.py": "lr = 0.001\n"},
        )
        assert not result.is_valid
        assert any("search text not found" in e for e in result.errors)


class TestFormatRejection:
    def test_format(self):
        from autotrain.experiment.sandbox import ValidationResult

        result = ValidationResult(
            is_valid=False,
            errors=["File 'model.py' not in whitelist"],
            warnings=[],
        )
        msg = format_rejection_message(result)
        assert "REJECTED" in msg
        assert "model.py" in msg

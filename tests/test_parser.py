"""Tests for agent response parser."""

from __future__ import annotations

import json

import pytest

from autotrain.agent.parser import ParseError, parse_response


class TestParseResponse:
    def test_clean_json(self):
        raw = json.dumps({
            "reasoning": "test", "hypothesis": "h",
            "changes": [{"file": "train.py", "action": "replace",
                         "search": "a", "replace": "b"}],
            "expected_impact": "small",
        })
        result = parse_response(raw)
        assert result.reasoning == "test"
        assert len(result.changes) == 1
        assert result.changes[0].file == "train.py"

    def test_json_in_code_block(self):
        raw = """Here's my analysis:
```json
{
  "reasoning": "test",
  "hypothesis": "h",
  "changes": [{"file": "train.py", "action": "replace",
               "search": "old", "replace": "new"}]
}
```
"""
        result = parse_response(raw)
        assert result.reasoning == "test"

    def test_json_with_surrounding_text(self):
        inner = json.dumps({
            "reasoning": "r", "hypothesis": "h",
            "changes": [{"file": "train.py",
                         "action": "full_rewrite", "content": "x"}],
        })
        raw = f"I think we should {inner} done"
        result = parse_response(raw)
        assert len(result.changes) == 1

    def test_missing_changes_field(self):
        with pytest.raises(ParseError, match="missing 'changes'"):
            parse_response('{"reasoning": "test"}')

    def test_no_json_found(self):
        with pytest.raises(ParseError, match="No valid JSON"):
            parse_response("Just some plain text")

    def test_default_action_is_replace(self):
        raw = json.dumps({
            "changes": [{"file": "train.py", "search": "a", "replace": "b"}],
        })
        result = parse_response(raw)
        assert result.changes[0].action == "replace"

    def test_multiple_changes(self):
        raw = json.dumps({
            "changes": [
                {"file": "train.py", "action": "replace",
                 "search": "a", "replace": "b"},
                {"file": "config.yaml", "action": "full_rewrite",
                 "content": "lr: 0.01"},
            ],
        })
        result = parse_response(raw)
        assert len(result.changes) == 2

    def test_default_impact(self):
        raw = json.dumps({
            "changes": [{"file": "train.py", "search": "a", "replace": "b"}],
        })
        result = parse_response(raw)
        assert result.expected_impact == "medium"

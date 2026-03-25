"""Parse structured JSON responses from the agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import structlog

from autotrain.experiment.sandbox import FileChange

log = structlog.get_logger()


@dataclass
class AgentDecision:
    """Parsed decision from the agent's response."""

    reasoning: str
    hypothesis: str
    changes: list[FileChange]
    expected_impact: str = "medium"
    raw_json: dict = field(default_factory=dict)


class ParseError(Exception):
    """Raised when the agent response cannot be parsed."""


def parse_response(raw_text: str) -> AgentDecision:
    """Parse the agent's JSON response into an AgentDecision.

    Handles:
    - Clean JSON
    - JSON wrapped in markdown code blocks
    - JSON with trailing text
    """
    json_str = _extract_json(raw_text)
    if json_str is None:
        raise ParseError(f"No valid JSON found in response: {raw_text[:200]}")

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e

    # Validate required fields
    if "changes" not in data:
        raise ParseError("Response missing 'changes' field")
    if not isinstance(data["changes"], list):
        raise ParseError("'changes' must be a list")

    # Parse changes
    changes = []
    for change_data in data["changes"]:
        if not isinstance(change_data, dict):
            raise ParseError(f"Each change must be a dict, got: {type(change_data)}")
        if "file" not in change_data:
            raise ParseError("Each change must have a 'file' field")

        changes.append(FileChange(
            file=change_data["file"],
            action=change_data.get("action", "replace"),
            search=change_data.get("search"),
            replace=change_data.get("replace"),
            content=change_data.get("content"),
        ))

    return AgentDecision(
        reasoning=data.get("reasoning", ""),
        hypothesis=data.get("hypothesis", ""),
        changes=changes,
        expected_impact=data.get("expected_impact", "medium"),
        raw_json=data,
    )


def _extract_json(text: str) -> str | None:
    """Extract JSON from text, handling markdown code blocks."""
    text = text.strip()

    # Try: JSON in markdown code block
    block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if block_match:
        candidate = block_match.group(1).strip()
        if _is_valid_json(candidate):
            return candidate

    # Try: Raw JSON object
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
        if _is_valid_json(candidate):
            return candidate

    return None


def _is_valid_json(s: str) -> bool:
    """Check if a string is valid JSON."""
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False

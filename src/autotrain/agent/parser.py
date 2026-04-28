"""Parse structured JSON responses from the agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import structlog

from autotrain.experiment.models import FileChange as PydanticFileChange

log = structlog.get_logger()


@dataclass
class AgentDecision:
    """Parsed decision from the agent's response."""

    reasoning: str
    hypothesis: str
    changes: list[PydanticFileChange]
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

    # Parse changes — support both legacy and Pydantic formats
    changes = _parse_changes(data["changes"])

    return AgentDecision(
        reasoning=data.get("reasoning", ""),
        hypothesis=data.get("hypothesis", ""),
        changes=changes,
        expected_impact=data.get("expected_impact", "medium"),
        raw_json=data,
    )


def _parse_changes(raw_changes: list) -> list[PydanticFileChange]:
    """Parse change dicts from agent JSON into Pydantic FileChange models.

    Supports two formats:

    *Legacy* (deprecated but still accepted):
        {"file": "train.py", "action": "replace",
         "search": "...", "replace": "..."}
        {"file": "train.py", "action": "create"|"full_rewrite",
         "content": "..."}

    *Pydantic* (preferred):
        {"path": "train.py", "operation": "update",
         "content": "..."}
        {"path": "new.py", "operation": "create",
         "content": "..."}
        {"path": "stale.py", "operation": "delete"}
    """
    from pydantic import ValidationError as PydanticValidationError

    results: list[PydanticFileChange] = []
    for cd in raw_changes:
        if not isinstance(cd, dict):
            raise ParseError(
                f"Each change must be a dict, got: {type(cd)}"
            )

        # -- Pydantic format (preferred) --------------------------
        if "path" in cd or "operation" in cd:
            if "path" not in cd:
                raise ParseError(
                    "Pydantic-style change missing 'path' field"
                )
            try:
                fc = PydanticFileChange(**cd)
            except PydanticValidationError as e:
                raise ParseError(
                    f"Invalid FileChange: {e}"
                ) from e
            results.append(fc)
            continue

        # -- Legacy format (backward compat) ----------------------
        if "file" not in cd:
            raise ParseError(
                "Each change must have 'file' (legacy) or 'path' (pydantic)"
            )

        action = cd.get("action", "replace")
        if action in ("replace",):
            search = cd.get("search", "")
            if not search:
                raise ParseError(
                    "Legacy 'replace' action requires 'search' field"
                )
            # "replace" maps to "update" — content will be
            # resolved by agent loop from current file state
            results.append(PydanticFileChange(
                path=cd["file"],
                operation="update",
                content="",  # placeholder; resolved by agent loop
                description=cd.get("description"),
            ))
        elif action in ("create", "full_rewrite"):
            results.append(PydanticFileChange(
                path=cd["file"],
                operation="create" if action == "create" else "update",
                content=cd.get("content", ""),
                description=cd.get("description"),
            ))
        else:
            raise ParseError(
                f"Unknown legacy action: {action!r}"
            )

    return results


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

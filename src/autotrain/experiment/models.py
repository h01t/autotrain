"""Pydantic models for agent-proposed file changes.

These models enforce strict schema validation on every edit
before it reaches the filesystem.  All LLM-generated edits are
treated as untrusted input.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ── operation policy ──────────────────────────────────────────────
# "rename" is intentionally excluded for this milestone.
# It will be added in a follow-up once atomic rename semantics
# are proven safe.
ALLOWED_OPERATIONS = ("create", "update", "delete")


class FileChange(BaseModel, extra="forbid", frozen=True):
    """A single validated file mutation proposed by the agent.

    Exactly one content-bearing field must be set, depending on
    *operation*:

    * ``create``  – ``content`` is required, ``patch`` is forbidden.
    * ``update``  – ``content`` must be set (``patch`` is rejected).
    * ``delete``  – neither ``content`` nor ``patch`` may be set.
    """

    path: str = Field(
        ...,
        min_length=1,
        description="Repo-relative path to the target file.",
    )
    operation: Literal["create", "update", "delete"] = Field(
        ...,
        description="The type of mutation to perform.",
    )
    content: str | None = Field(
        default=None,
        description="Full replacement content (create / full-rewrite update).",
    )
    patch: str | None = Field(
        default=None,
        description="Unified-diff patch (REJECTED in this milestone — use content).",
    )
    description: str | None = Field(
        default=None,
        max_length=500,
        description="Optional human-readable explanation of the change.",
    )

    @model_validator(mode="after")
    def _check_payload_consistency(self) -> FileChange:
        op = self.operation

        if op == "create":
            if self.content is None:
                raise ValueError(
                    "create operation requires 'content' to be set."
                )
            if self.patch is not None:
                raise ValueError(
                    "create operation does not accept 'patch'."
                )

        elif op == "update":
            if self.patch is not None:
                raise ValueError(
                    "Patch-based updates are not supported in this milestone. "
                    "Use 'content' for full-file updates."
                )
            if self.content is None:
                raise ValueError(
                    "update operation requires 'content' to be set."
                )

        elif op == "delete":
            if self.content is not None or self.patch is not None:
                raise ValueError(
                    "delete operation must not include 'content' or 'patch'."
                )

        return self

    @field_validator("path")
    @classmethod
    def _path_sanitize(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("path must not be empty or whitespace-only")
        for ch in stripped:
            if ord(ch) < 0x20 and ch not in ("\t",):
                raise ValueError(
                    f"path contains control character U+{ord(ch):04X}"
                )
        return stripped


class PatchRequest(BaseModel, extra="forbid"):
    """An outer batch of file changes proposed by the agent.

    This is the top-level model parsed from the agent's JSON response.
    """

    changes: list[FileChange] = Field(
        ...,
        min_length=1,
        max_length=20,  # hard safety cap – config may be stricter
        description="Ordered list of file mutations for this iteration.",
    )
    reasoning: str = Field(
        default="",
        max_length=5000,
    )
    hypothesis: str = Field(
        default="",
        max_length=2000,
    )

    @field_validator("changes")
    @classmethod
    def _no_duplicate_paths(cls, v: list[FileChange]) -> list[FileChange]:
        seen: set[str] = set()
        for fc in v:
            n = fc.path
            if n in seen:
                raise ValueError(
                    f"Duplicate file path in batch: {n!r}"
                )
            seen.add(n)
        return v

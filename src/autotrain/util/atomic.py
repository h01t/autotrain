"""Atomic file write operations — crash-safe via temp + fsync + rename."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write(path: Path, data: str | bytes, *, mode: str = "w") -> None:
    """Write data to path atomically.

    Writes to a temporary file on the same filesystem, fsyncs, then renames.
    This guarantees the file is either fully written or not written at all.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    is_bytes = isinstance(data, bytes)
    write_mode = "wb" if is_bytes else "w"

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, write_mode) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, path)
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

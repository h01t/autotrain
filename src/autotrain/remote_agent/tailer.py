"""Async log file tailer — follows a file like tail -F."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path


async def tail_file(path: Path, poll_interval: float = 0.1) -> AsyncIterator[str]:
    """Yield new lines as they're appended to a file.

    Handles file rotation (inode change) and creation.
    """
    inode = None
    offset = 0
    fp = None

    try:
        while True:
            if not path.exists():
                await asyncio.sleep(poll_interval)
                continue

            current_inode = os.stat(path).st_ino

            # File rotated or first open
            if inode != current_inode:
                if fp:
                    fp.close()
                fp = open(path)
                inode = current_inode
                # On first open, seek to end (don't replay history)
                if offset == 0:
                    fp.seek(0, 2)
                    offset = fp.tell()
                else:
                    offset = 0  # Rotated file — read from start

            fp.seek(offset)
            new_data = fp.read()

            if new_data:
                offset = fp.tell()
                for line in new_data.splitlines():
                    if line:
                        yield line
            else:
                await asyncio.sleep(poll_interval)

    finally:
        if fp:
            fp.close()

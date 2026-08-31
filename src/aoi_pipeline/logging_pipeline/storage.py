"""Local disk-usage monitoring for the edge device's capture storage.

Storage plan per issue #4: images and logs live on local (or attached)
storage on the edge device — no off-device upload, for IP/confidentiality
reasons. The only automated safeguard needed here is alerting before the
device runs out of space.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class StorageCriticalError(RuntimeError):
    """Raised when disk usage crosses the critical threshold and the caller
    has asked to be stopped rather than merely warned."""


@dataclass
class StorageStatus:
    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int

    @property
    def percent_used(self) -> float:
        return 100.0 * self.used_bytes / self.total_bytes if self.total_bytes else 0.0


class StorageMonitor:
    """Checks free disk space at `path` against warning/critical thresholds.

    Default callbacks just log; pass `on_warning`/`on_critical` to hook in
    something else (e.g. a line-side alert). `check()` returns the current
    status regardless of threshold, so callers can also log/record it
    unconditionally (e.g. alongside each capture).
    """

    def __init__(
        self,
        path: Path,
        warning_pct: float = 80.0,
        critical_pct: float = 95.0,
        on_warning: Callable[[StorageStatus], None] | None = None,
        on_critical: Callable[[StorageStatus], None] | None = None,
    ) -> None:
        if not (0 < warning_pct < critical_pct <= 100):
            raise ValueError("require 0 < warning_pct < critical_pct <= 100")
        self.path = Path(path)
        self.warning_pct = warning_pct
        self.critical_pct = critical_pct
        self.on_warning = on_warning or self._default_warning
        self.on_critical = on_critical or self._default_critical

    def get_status(self) -> StorageStatus:
        usage = shutil.disk_usage(self.path)
        return StorageStatus(
            path=str(self.path),
            total_bytes=usage.total,
            used_bytes=usage.used,
            free_bytes=usage.free,
        )

    def check(self) -> StorageStatus:
        status = self.get_status()
        if status.percent_used >= self.critical_pct:
            self.on_critical(status)
        elif status.percent_used >= self.warning_pct:
            self.on_warning(status)
        return status

    @staticmethod
    def _default_warning(status: StorageStatus) -> None:
        logger.warning(
            "storage at %.1f%% used on %s (%d bytes free)",
            status.percent_used,
            status.path,
            status.free_bytes,
        )

    @staticmethod
    def _default_critical(status: StorageStatus) -> None:
        logger.critical(
            "storage at %.1f%% used on %s (%d bytes free) — critical threshold reached",
            status.percent_used,
            status.path,
            status.free_bytes,
        )

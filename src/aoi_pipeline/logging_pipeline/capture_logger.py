"""Edge-device logging pipeline: every triggered image gets a timestamp,
sequence number, and (if available) board ID, appended to a JSONL log.

JSONL (append-only, one record per line) is used instead of a database so a
crash or power loss mid-write only ever loses/corrupts the last partial
line, never the whole log — important for an unattended edge device.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from aoi_pipeline.logging_pipeline.storage import StorageCriticalError, StorageMonitor
from aoi_pipeline.schema.capture import CaptureRecord


class CaptureLogger:
    """Appends one CaptureRecord per captured image to a daily JSONL log.

    Sequence numbers are monotonic per logger instance. On startup, if a log
    file for today already exists, the counter resumes from the last
    recorded sequence number + 1 rather than restarting at 0, so a service
    restart doesn't create duplicate sequence numbers within the same day.
    """

    def __init__(
        self,
        log_dir: Path,
        station_id: str = "station-01",
        storage_monitor: StorageMonitor | None = None,
        stop_on_critical_storage: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.station_id = station_id
        self.storage_monitor = storage_monitor or StorageMonitor(self.log_dir)
        self.stop_on_critical_storage = stop_on_critical_storage
        self._lock = threading.Lock()
        self._sequence_number = self._resume_sequence_number()

    def _log_path(self, day: datetime | None = None) -> Path:
        day = day or datetime.now(timezone.utc)
        return self.log_dir / f"captures_{day:%Y%m%d}.jsonl"

    def _resume_sequence_number(self) -> int:
        path = self._log_path()
        if not path.exists():
            return 0
        last_seq = -1
        for record in iter_records(path):
            last_seq = max(last_seq, record.sequence_number)
        return last_seq + 1

    def log_capture(
        self,
        image_path: str | Path,
        board_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CaptureRecord:
        """Record one captured image. Thread-safe."""
        with self._lock:
            status = self.storage_monitor.check()
            if (
                self.stop_on_critical_storage
                and status.percent_used >= self.storage_monitor.critical_pct
            ):
                raise StorageCriticalError(
                    f"refusing to log capture: {status.path} at "
                    f"{status.percent_used:.1f}% used"
                )

            record = CaptureRecord(
                sequence_number=self._sequence_number,
                timestamp=datetime.now(timezone.utc),
                image_path=str(image_path),
                station_id=self.station_id,
                board_id=board_id,
                metadata=metadata or {},
            )
            with open(self._log_path(), "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
            self._sequence_number += 1
            return record


def iter_records(log_path: Path) -> Iterator[CaptureRecord]:
    """Read back a capture log file, one CaptureRecord per line."""
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield CaptureRecord.from_dict(json.loads(line))

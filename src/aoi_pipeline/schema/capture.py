"""Capture record schema: image + timestamp + sequence number (+ optional board ID).

This is what the edge-device logging pipeline writes for every triggered
image, and what the correlation module later matches against the existing
AOI's verdicts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class CaptureRecord:
    """One triggered image capture on the edge device.

    `board_id` is optional and pluggable: it is populated only if a
    barcode/serial reader is confirmed present on the line (see issue #1
    pre-flight check). When absent, correlation falls back to sequence
    number + timestamp matching.
    """

    sequence_number: int
    timestamp: datetime
    image_path: str
    station_id: str = "unknown"
    board_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.astimezone(timezone.utc).isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaptureRecord:
        return cls(
            sequence_number=data["sequence_number"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            image_path=data["image_path"],
            station_id=data.get("station_id", "unknown"),
            board_id=data.get("board_id"),
            metadata=data.get("metadata", {}),
        )

"""JSONL load/dump for LabeledCapture streams.

One JSON object per line, shaped like:
    {
      "capture": {"sequence_number": 1, "timestamp": "2026-01-01T08:00:00+00:00",
                   "image_path": "...", "station_id": "...", "board_id": null, "metadata": {}},
      "label": {"board_id": "...", "verdict": "PASS", "source": "EXISTING_AOI",
                 "defect_types": ["NONE"], "raw_label": null, "confidence": null,
                 "notes": "", "created_at": "2026-01-01T08:00:01+00:00"}
    }

This is the format real capture/labeling logs (from issue #4's pipeline)
are expected to be exported in. Nothing in comparison.py or report.py
depends on this module — it only exists to load/save that format.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from shadow_analysis.schema import BoardLabel, CaptureRecord, DefectType, LabeledCapture, LabelSource, Verdict


def _capture_from_dict(d: dict) -> CaptureRecord:
    return CaptureRecord(
        sequence_number=d["sequence_number"],
        timestamp=datetime.fromisoformat(d["timestamp"]),
        image_path=d["image_path"],
        station_id=d.get("station_id", "unknown"),
        board_id=d.get("board_id"),
        metadata=d.get("metadata", {}),
    )


def _label_from_dict(d: dict) -> BoardLabel:
    return BoardLabel(
        board_id=d["board_id"],
        verdict=Verdict(d["verdict"]),
        source=LabelSource(d["source"]),
        defect_types=[DefectType(x) for x in d["defect_types"]],
        raw_label=d.get("raw_label"),
        confidence=d.get("confidence"),
        notes=d.get("notes", ""),
        created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.now(),
    )


def _capture_to_dict(c: CaptureRecord) -> dict:
    return {
        "sequence_number": c.sequence_number,
        "timestamp": c.timestamp.isoformat(),
        "image_path": c.image_path,
        "station_id": c.station_id,
        "board_id": c.board_id,
        "metadata": c.metadata,
    }


def _label_to_dict(label: BoardLabel) -> dict:
    return {
        "board_id": label.board_id,
        "verdict": label.verdict.value,
        "source": label.source.value,
        "defect_types": [d.value for d in label.defect_types],
        "raw_label": label.raw_label,
        "confidence": label.confidence,
        "notes": label.notes,
        "created_at": label.created_at.isoformat(),
    }


def load_labeled_captures(path: str | Path) -> list[LabeledCapture]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append(
                LabeledCapture(
                    capture=_capture_from_dict(d["capture"]),
                    label=_label_from_dict(d["label"]),
                )
            )
    return records


def write_labeled_captures(records: list[LabeledCapture], path: str | Path) -> None:
    with open(path, "w") as f:
        for lc in records:
            d = {"capture": _capture_to_dict(lc.capture), "label": _label_to_dict(lc.label)}
            f.write(json.dumps(d) + "\n")

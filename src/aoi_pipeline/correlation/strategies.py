"""Match strategies for correlating new-system captures with the existing
AOI's verdicts on the same board.

Default path is sequence + timestamp matching, since a barcode/serial ID is
not confirmed to exist on the line yet (issue #1 pre-flight check). The
ID-based path is implemented as a pluggable strategy so it can be swapped in
(via `CompositeMatchStrategy`, which prefers it whenever both sides carry an
ID) without touching the rest of the pipeline once IDs are confirmed
available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from aoi_pipeline.schema.capture import CaptureRecord
from aoi_pipeline.schema.labels import DefectType, Verdict


@dataclass
class ExistingAoiVerdict:
    """A verdict pulled from the existing AOI system for one inspected board.

    `sequence_number` here is the existing AOI's own counter, if it has one
    — it is not assumed comparable to our `CaptureRecord.sequence_number`
    (different systems, different start points), so it is not used by
    `SequenceTimestampMatchStrategy`; it's carried through for audit/debug
    purposes only.
    """

    verdict_id: str
    timestamp: datetime
    verdict: Verdict
    defect_types: list[DefectType] = field(default_factory=list)
    board_id: str | None = None
    sequence_number: int | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelationMatch:
    capture: CaptureRecord
    verdict: ExistingAoiVerdict
    method: str  # "id" | "sequence_timestamp"
    time_delta_seconds: float


class MatchStrategy(ABC):
    """Matches captures to existing-AOI verdicts for the same board.

    Implementations must not mutate their inputs, and must return each
    capture/verdict in at most one match (1:1 correlation).
    """

    @abstractmethod
    def match(
        self,
        captures: list[CaptureRecord],
        verdicts: list[ExistingAoiVerdict],
    ) -> tuple[list[CorrelationMatch], list[CaptureRecord], list[ExistingAoiVerdict]]:
        """Returns (matches, unmatched_captures, unmatched_verdicts)."""


class IdMatchStrategy(MatchStrategy):
    """Matches on exact board_id equality. Only usable once a barcode/serial
    reader is confirmed present on the line."""

    def match(
        self,
        captures: list[CaptureRecord],
        verdicts: list[ExistingAoiVerdict],
    ) -> tuple[list[CorrelationMatch], list[CaptureRecord], list[ExistingAoiVerdict]]:
        verdicts_by_id: dict[str, list[ExistingAoiVerdict]] = {}
        for v in verdicts:
            if v.board_id:
                verdicts_by_id.setdefault(v.board_id, []).append(v)

        matches: list[CorrelationMatch] = []
        unmatched_captures: list[CaptureRecord] = []
        used_verdicts: set[int] = set()

        for c in captures:
            candidates = verdicts_by_id.get(c.board_id) if c.board_id else None
            match_verdict = None
            if candidates:
                for v in candidates:
                    if id(v) not in used_verdicts:
                        match_verdict = v
                        break
            if match_verdict is None:
                unmatched_captures.append(c)
                continue
            used_verdicts.add(id(match_verdict))
            delta = abs((c.timestamp - match_verdict.timestamp).total_seconds())
            matches.append(
                CorrelationMatch(
                    capture=c, verdict=match_verdict, method="id", time_delta_seconds=delta
                )
            )

        unmatched_verdicts = [v for v in verdicts if id(v) not in used_verdicts]
        return matches, unmatched_captures, unmatched_verdicts


class SequenceTimestampMatchStrategy(MatchStrategy):
    """Matches on nearest timestamp within `tolerance`, default path when no
    board ID is available.

    Greedy nearest-neighbor: all (capture, verdict) pairs within tolerance
    are ranked by time delta ascending, then assigned greedily so each side
    is used at most once. Sequence number is not used as a matching key
    (the two systems' counters aren't assumed comparable) but is preserved
    on both sides for audit trails.
    """

    def __init__(self, tolerance: timedelta = timedelta(seconds=2)) -> None:
        self.tolerance = tolerance

    def match(
        self,
        captures: list[CaptureRecord],
        verdicts: list[ExistingAoiVerdict],
    ) -> tuple[list[CorrelationMatch], list[CaptureRecord], list[ExistingAoiVerdict]]:
        candidate_pairs: list[tuple[float, int, int]] = []  # (delta, c_idx, v_idx)
        for ci, c in enumerate(captures):
            for vi, v in enumerate(verdicts):
                delta = abs((c.timestamp - v.timestamp).total_seconds())
                if delta <= self.tolerance.total_seconds():
                    candidate_pairs.append((delta, ci, vi))
        candidate_pairs.sort(key=lambda p: p[0])

        used_captures: set[int] = set()
        used_verdicts: set[int] = set()
        matches: list[CorrelationMatch] = []
        for delta, ci, vi in candidate_pairs:
            if ci in used_captures or vi in used_verdicts:
                continue
            used_captures.add(ci)
            used_verdicts.add(vi)
            matches.append(
                CorrelationMatch(
                    capture=captures[ci],
                    verdict=verdicts[vi],
                    method="sequence_timestamp",
                    time_delta_seconds=delta,
                )
            )

        unmatched_captures = [c for i, c in enumerate(captures) if i not in used_captures]
        unmatched_verdicts = [v for i, v in enumerate(verdicts) if i not in used_verdicts]
        return matches, unmatched_captures, unmatched_verdicts


class CompositeMatchStrategy(MatchStrategy):
    """Prefers ID-based matching wherever both sides carry a board_id, and
    falls back to sequence+timestamp matching for everything else.

    This is the strategy the correlation module should use by default: it
    degrades gracefully from "no ID reader on the line" (pure
    sequence+timestamp) to "ID reader confirmed" (ID-first) without any
    caller-side branching.
    """

    def __init__(
        self,
        id_strategy: IdMatchStrategy | None = None,
        fallback_strategy: SequenceTimestampMatchStrategy | None = None,
    ) -> None:
        self.id_strategy = id_strategy or IdMatchStrategy()
        self.fallback_strategy = fallback_strategy or SequenceTimestampMatchStrategy()

    def match(
        self,
        captures: list[CaptureRecord],
        verdicts: list[ExistingAoiVerdict],
    ) -> tuple[list[CorrelationMatch], list[CaptureRecord], list[ExistingAoiVerdict]]:
        id_matches, remaining_captures, remaining_verdicts = self.id_strategy.match(
            captures, verdicts
        )
        fallback_matches, unmatched_captures, unmatched_verdicts = self.fallback_strategy.match(
            remaining_captures, remaining_verdicts
        )
        return id_matches + fallback_matches, unmatched_captures, unmatched_verdicts

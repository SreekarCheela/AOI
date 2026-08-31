"""Orchestrates a MatchStrategy over a batch of captures and existing-AOI
verdicts, and reports a match rate (acceptance criterion: >=95% of captures
matched within a 1-day test run)."""

from __future__ import annotations

from dataclasses import dataclass

from aoi_pipeline.correlation.strategies import (
    CompositeMatchStrategy,
    CorrelationMatch,
    ExistingAoiVerdict,
    MatchStrategy,
)
from aoi_pipeline.schema.capture import CaptureRecord


@dataclass
class CorrelationReport:
    matches: list[CorrelationMatch]
    unmatched_captures: list[CaptureRecord]
    unmatched_verdicts: list[ExistingAoiVerdict]

    @property
    def total_captures(self) -> int:
        return len(self.matches) + len(self.unmatched_captures)

    @property
    def match_rate(self) -> float:
        """Fraction of captures that were matched to a verdict, in [0, 1]."""
        if self.total_captures == 0:
            return 1.0
        return len(self.matches) / self.total_captures


class CorrelationMatcher:
    """Correlates new-system captures with existing-AOI verdicts for the
    same board. Defaults to `CompositeMatchStrategy` (ID-first, falls back
    to sequence+timestamp) but any `MatchStrategy` can be injected."""

    def __init__(self, strategy: MatchStrategy | None = None) -> None:
        self.strategy = strategy or CompositeMatchStrategy()

    def run(
        self,
        captures: list[CaptureRecord],
        verdicts: list[ExistingAoiVerdict],
    ) -> CorrelationReport:
        matches, unmatched_captures, unmatched_verdicts = self.strategy.match(
            captures, verdicts
        )
        return CorrelationReport(
            matches=matches,
            unmatched_captures=unmatched_captures,
            unmatched_verdicts=unmatched_verdicts,
        )

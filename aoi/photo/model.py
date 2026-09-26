"""The shapes every photo-mode stage passes along.

Vocabulary, fixed by the design (the part-state table in the plan):

    check     what was asked:   missing, tombstone, wrong_part, marking,
                                polarity, offset, skew, glare, view, uninspected
    state     what was found:   see STATES
    severity  what it does to the board: FAIL, REVIEW, INFO, OK

A part can carry several calls (a POLARITY REVIEW and an OFFSET), and the
operator adjudicates each call, not the part. The machine verdict is never
overwritten by adjudication; record.py stores the operator's status beside it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

# state -> severity
STATES: dict[str, str] = {
    "MISSING": "FAIL",
    "TOMBSTONE": "FAIL",          # unvalidated on real boards
    "WRONG PART": "FAIL",         # unvalidated on real boards
    "REVERSED": "FAIL",
    "OFFSET": "FAIL",
    "SKEW": "FAIL",
    "POLARITY REVIEW": "REVIEW",
    "MARKING DIFFERS": "REVIEW",
    "NOT CHECKED": "REVIEW",      # glare or outside the photo
    "UNEXPECTED DIFFERENCE": "REVIEW",   # differs from golden 1 where no part was taught
    "UNINSPECTED": "INFO",        # golden spread makes this check undecidable
    "OK": "OK",
}
UNVALIDATED = {"TOMBSTONE", "WRONG PART"}
UNVALIDATED_NOTE = "unvalidated on real boards"

PART_TYPES = ("resistor", "capacitor", "diode", "led", "sot23", "regulator", "ic", "bridge", "other")
MARKED_ENDS = ("left", "right", "top", "bottom", "none")
# An UNEXPECTED DIFFERENCE is carried as a part of this type named DIFF1, DIFF2...
# so the report, the record and Review show and adjudicate it like any part.
DIFF_TYPE = "difference"
DIFF_PREFIX = "DIFF"


@dataclass
class Part:
    refdes: str
    x: int
    y: int
    w: int
    h: int
    type: str = "other"
    package: str = ""
    polarized: bool = False
    marked_end: str = "none"     # cathode band / pin 1 / tab side, in golden-1 pixels

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def horizontal(self) -> bool:
        return self.w >= self.h

    @property
    def group(self) -> str:
        """What Stats groups by: type and package size."""
        return f"{self.type} {self.package}".strip()


@dataclass
class Call:
    check: str
    state: str
    detail: str = ""
    score: float | None = None        # 0-1, larger = more evidence for the state
    calibrated: bool = False          # True only after Platt scaling on labels
    measured_against: str = ""        # "golden spread", "silkscreen outline", ...

    @property
    def severity(self) -> str:
        return STATES[self.state]

    @property
    def unvalidated(self) -> bool:
        return self.state in UNVALIDATED

    def score_text(self) -> str:
        if self.score is None:
            return ""
        return f"P={self.score:.2f}" if self.calibrated else f"score {self.score:.2f}"


@dataclass
class PartResult:
    part: Part
    calls: list[Call] = field(default_factory=list)
    # Found body position relative to the taught box, in golden-1 pixels, net of
    # the board-wide common shift. dtheta_deg is counter-clockwise as displayed
    # (the golden body turned by +dtheta about its centre lies on the test body).
    dx_px: float = 0.0
    dy_px: float = 0.0
    dtheta_deg: float = 0.0
    measures: dict = field(default_factory=dict)

    @property
    def refdes(self) -> str:
        return self.part.refdes

    @property
    def flagged(self) -> list[Call]:
        return [c for c in self.calls if c.severity in ("FAIL", "REVIEW")]

    @property
    def state(self) -> str:
        """The headline state: first FAIL, else first REVIEW, else first INFO, else OK."""
        for sev in ("FAIL", "REVIEW", "INFO"):
            for c in self.calls:
                if c.severity == sev:
                    return c.state
        return "OK"


@dataclass
class BoardResult:
    image: str
    program: str
    program_version: int
    verdict: str                              # PASS / FAIL / REVIEW
    reasons: list[str] = field(default_factory=list)
    parts: list[PartResult] = field(default_factory=list)
    board_calls: list[Call] = field(default_factory=list)   # board-level REVIEWs
    px_per_mm: float = 0.0
    common_shift_px: tuple[float, float] = (0.0, 0.0)
    noise_px: float = 0.0
    offset_limit_px: float = 0.0
    single_golden: bool = False
    stamp: str = ""
    registration: dict = field(default_factory=dict)

    def part(self, refdes: str) -> PartResult:
        for p in self.parts:
            if p.refdes == refdes:
                return p
        raise KeyError(refdes)

    def to_dict(self) -> dict:
        d = asdict(self)
        for pr, src in zip(d["parts"], self.parts):
            pr["state"] = src.state
            for c, cs in zip(pr["calls"], src.calls):
                c["severity"] = cs.severity
        return d

"""A package teach never saw is not inspected, and says so.

Every test in the rule layer compares against a golden statistic for that
package: correlation against its template, band distances against its
covariance, the body segmentation thresholds themselves. With no golden entry
there is nothing to compare against and nothing can be judged.

`measure` used to return early leaving `ncc` at its 0.0 default, and 0.0 is a
perfectly good measurement meaning "correlates with nothing". Every threshold
downstream read it as a gross fault, so one unprogrammed package flagged every
instance of itself as missing. On a real board with fifteen package families and
a library holding five, that is a false call on a large fraction of the board --
the exact failure mode section 8 says decides whether the machine gets used.

The fix is a third verdict state. Not pass, not fail: uninspected. Reported once
at board level as a coverage gap rather than N times as a defect.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aoi import board as B
from aoi import pipeline, rules, synth
from aoi.fusion import rule_verdict

from conftest import render_with

UNTAUGHT = "0603"


@pytest.fixture(scope="module")
def partial_golden(brd):
    """Golden fitted on a board that carries no 0603 at all."""
    taught = B.Board(brd.name, brd.width, brd.height,
                     [c for c in brd.components if c.package != UNTAUGHT],
                     brd.fiducials)
    stacks = [synth.render(taught, opts=synth.RenderOpts(seed=400 + i))
              for i in range(4)]
    g = rules.fit_golden(stacks, taught)
    assert UNTAUGHT not in g.packages, "fixture did not actually leave it untaught"
    return g


def _untaught_comps(brd):
    return [c for c in brd.components if c.package == UNTAUGHT]


# ------------------------------------------------------------- the measurement
def test_unmeasurable_fields_are_nan_not_zero(brd, partial_golden):
    """The root cause. An unknown must not be representable as a good reading.

    0.0 correlation is what a totally wrong part scores, so leaving the default
    made "not measured" indistinguishable from "grossly defective" to every
    threshold downstream.
    """
    comp = _untaught_comps(brd)[0]
    m = rules.measure(render_with(brd, {}), comp, partial_golden)

    for name in ("ncc", "dx_um", "dy_um", "dtheta_deg", "pol_margin",
                 "contrast_ratio", "band_maha", "body_area_ratio",
                 "body_aspect", "body_shift_um"):
        assert math.isnan(getattr(m, name)), \
            f"{name} is {getattr(m, name)!r}, which reads as a measurement"


def test_raw_measurements_that_need_no_golden_are_still_taken(brd, partial_golden):
    """Band fractions and gap response are absolute readings, not comparisons.
    They stay available so the crop is still worth storing for a later re-teach.
    """
    comp = _untaught_comps(brd)[0]
    m = rules.measure(render_with(brd, {}), comp, partial_golden)
    assert not math.isnan(m.gap_spec)
    assert m.bands.shape == (2, 3)
    assert m.bands.sum() > 0.0, "no band fractions measured at all"


# ----------------------------------------------------------------- the verdict
def test_an_untaught_package_produces_no_false_calls(brd, partial_golden):
    """The bug, stated as the number that matters."""
    stack = render_with(brd, {})
    flagged = [c.refdes for c in _untaught_comps(brd)
               if rule_verdict(rules.measure(stack, c, partial_golden), partial_golden).failed]
    assert not flagged, (
        f"{len(flagged)} of {len(_untaught_comps(brd))} untaught components "
        f"flagged as defects: {flagged}")


def test_untaught_components_are_not_passed_either(brd, partial_golden):
    """The half that keeps this honest.

    Suppressing the false calls by calling these parts good would trade a false
    call problem for an escape problem, which is far worse. They are uninspected,
    which is neither.
    """
    stack = render_with(brd, {})
    for c in _untaught_comps(brd):
        v = rule_verdict(rules.measure(stack, c, partial_golden), partial_golden)
        assert not v.testable, f"{c.refdes} claims to have been inspected"
        assert not v.passed, f"{c.refdes} was passed without being inspected"
        assert v.reason == "untaught"


@pytest.mark.parametrize("kind", ["missing", "tombstone", "offset", "insufficient"])
def test_a_real_defect_on_an_untaught_package_is_not_passed(brd, partial_golden, kind):
    """The consequence worth being explicit about: a genuine defect on an
    unprogrammed package is NOT caught. It cannot be, there is nothing to compare
    against. What matters is that it is never reported as good."""
    comp = _untaught_comps(brd)[0]
    defect = synth.Defect(kind, dx=0.25 if kind == "offset" else 0.0)
    m = rules.measure(render_with(brd, {comp.refdes: defect}), comp, partial_golden)
    v = rule_verdict(m, partial_golden)
    assert not v.passed, f"a {kind} defect on an untaught package was passed"
    assert not v.testable


def test_taught_packages_are_unaffected(brd, partial_golden):
    """The gap must be scoped to the package that has no program, not to the
    board."""
    stack = render_with(brd, {})
    for c in brd.components:
        if c.package == UNTAUGHT:
            continue
        v = rule_verdict(rules.measure(stack, c, partial_golden), partial_golden)
        assert v.testable, f"{c.refdes} ({c.package}) went untestable but was taught"
        assert not v.failed, f"{c.refdes} ({c.package}) false-called"


# ------------------------------------------------------------- board reporting
def test_the_board_report_separates_coverage_from_defects(brd, partial_golden):
    """One line at board level, not N lines in the review queue. Mixing them is
    how an unprogrammed package buries the real defects."""
    stack = synth.render(brd, defects={"R1": synth.Defect("missing")},
                         opts=synth.RenderOpts(seed=9, place_sigma_um=0.0))
    res = pipeline.Inspector(partial_golden).inspect(stack, brd)

    assert len(res.untestable) == len(_untaught_comps(brd))
    assert res.untaught_packages == [UNTAUGHT]
    assert "R1" in {v.refdes for v in res.failed}, "the real defect was lost"
    assert not any(v.package == UNTAUGHT for v in res.failed), \
        "untaught components leaked into the defect queue"

    text = res.summary()
    assert "NOT INSPECTED" in text and UNTAUGHT in text, \
        f"board summary hides the coverage gap: {text!r}"


def test_the_fusion_layer_cannot_override_untestable(brd, partial_golden):
    """The tree scores the same feature vector the rules could not read -- every
    column it would use is NaN. It does not get a vote here."""
    class AlwaysDefect:
        def score(self, X):
            return np.ones(len(X), np.float32)

    stack = render_with(brd, {})
    res = pipeline.Inspector(partial_golden, fusion=AlwaysDefect()).inspect(stack, brd)
    leaked = [v.refdes for v in res.verdicts if v.package == UNTAUGHT and v.failed]
    assert not leaked, f"fusion overrode {len(leaked)} untestable verdicts: {leaked}"
    assert len(res.untestable) == len(_untaught_comps(brd))


def test_a_fully_taught_board_reports_no_coverage_gap(brd, golden):
    """The normal case must not gain a spurious warning."""
    res = pipeline.Inspector(golden).inspect(render_with(brd, {}), brd)
    assert not res.untestable
    assert res.untaught_packages == []
    assert "NOT INSPECTED" not in res.summary()

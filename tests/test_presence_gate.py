"""The presence gate: package-relative thresholds learned at teach.

Replaces a global `ncc < 0.62`. That constant could not mean the same thing for
two different packages, because golden NCC is a property of the package: an 0603
matches its own template at 0.92, a SOD-123 at 0.99. One absolute number
therefore sat inside one package's defect distribution while being unreachable
for another. Concretely, a tombstoned 0402 scored 0.599 when pad 1 survived and
0.647 when pad 0 did, so which land happened to survive decided whether the
fault was found at all.

These tests pin the replacement so it cannot drift back.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aoi import config, pipeline, rules, synth
from aoi.fusion import rule_verdict

from conftest import render_with

CHIPS = ["0402", "0603", "0805"]
CHIP_REFDES = {"0402": "R1", "0603": "R3", "0805": "R5"}


# ------------------------------------------------------- what teach now records
def test_golden_records_the_good_part_ncc_distribution(golden):
    """fit_golden has to measure how well a GOOD part matches its own template,
    or there is nothing to make the threshold relative to."""
    for name, g in golden.packages.items():
        assert 0.5 < g.ncc_mean <= 1.0, \
            f"{name} golden NCC mean {g.ncc_mean:.3f} is not a plausible score"
        assert g.ncc_std >= 0.0
        assert g.ncc_absent < g.ncc_fail < g.ncc_mean, (
            f"{name} thresholds out of order: absent={g.ncc_absent:.3f} "
            f"fail={g.ncc_fail:.3f} mean={g.ncc_mean:.3f}")


def test_golden_ncc_differs_enough_between_packages_to_matter(golden):
    """The justification for the whole change. If every package scored the same,
    a global constant would have been fine."""
    means = {n: g.ncc_mean for n, g in golden.packages.items()}
    spread = max(means.values()) - min(means.values())
    assert spread > 0.04, (
        f"golden NCC spread across packages is only {spread:.3f} "
        f"({means}); a package-relative threshold would be unnecessary")


def test_good_parts_sit_clear_of_their_own_fail_threshold(brd, golden):
    """A correctly placed part must not be anywhere near tripping presence,
    otherwise the gate becomes a false-call generator."""
    stack = render_with(brd, {})
    for comp in brd.components:
        m = rules.measure(stack, comp, golden)
        g = golden.packages[comp.package]
        assert m.ncc > g.ncc_fail, (
            f"{comp.refdes} ({comp.package}) scores {m.ncc:.3f} against a fail "
            f"threshold of {g.ncc_fail:.3f}")


# --------------------------------------------------------- the fix, on the chips
@pytest.mark.parametrize("package", CHIPS)
@pytest.mark.parametrize("pad", [0, 1])
def test_tombstone_is_caught_on_either_pad(brd, golden, package, pad):
    """The bug. Detection must not depend on which land survived.

    Both pads are exercised for every chip package, because the original failure
    was invisible on one pad and present on the other.
    """
    refdes = CHIP_REFDES[package]
    stack = render_with(brd, {refdes: synth.Defect("tombstone", pad=pad)})
    m = rules.measure(stack, brd.by_refdes(refdes), golden)
    verdict = rule_verdict(m, golden)
    assert verdict.reason == "tombstone", (
        f"{package} tombstone on pad {pad} reported as {verdict.reason!r} "
        f"(ncc={m.ncc:.3f}, golden={golden.packages[package].ncc_mean:.3f})")


@pytest.mark.parametrize("package", CHIPS)
def test_tombstone_detection_does_not_depend_on_which_pad_survives(brd, golden, package):
    """Stated as its own assertion because it is the property that broke: the
    two pads must reach the same verdict, whatever that verdict is."""
    refdes = CHIP_REFDES[package]
    got = []
    for pad in (0, 1):
        stack = render_with(brd, {refdes: synth.Defect("tombstone", pad=pad)})
        m = rules.measure(stack, brd.by_refdes(refdes), golden)
        got.append(rule_verdict(m, golden).reason)
    assert got[0] == got[1], \
        f"{package} tombstone reads {got[0]!r} on pad 0 but {got[1]!r} on pad 1"


# ------------------------------------------- displaced parts are not tombstones
@pytest.mark.parametrize("package", CHIPS)
def test_a_displaced_part_is_not_called_a_tombstone(brd, golden, package):
    """The guard that lets the gate stay strict.

    A part that walked off its lands also has a degraded NCC and a dead joint,
    so NCC alone cannot tell it from a tombstone. The difference is that a
    tombstone's surviving joint is fully formed (the part IS soldered, at one
    end) while a displaced part has both joints compromised. Mislabelling here
    sends the operator to a reflow problem when the fault is placement.
    """
    refdes = CHIP_REFDES[package]
    stack = render_with(brd, {refdes: synth.Defect("offset", dx=0.25)})
    m = rules.measure(stack, brd.by_refdes(refdes), golden)
    verdict = rule_verdict(m, golden)
    assert verdict.failed
    assert verdict.reason != "tombstone", (
        f"{package} displaced 250 um reported as a tombstone "
        f"(ncc={m.ncc:.3f})")


@pytest.mark.parametrize("package", CHIPS)
def test_missing_is_still_missing(brd, golden, package):
    """The gate's other exit. Nothing wetted and no body means nothing was
    placed, and that must survive the rewrite."""
    refdes = CHIP_REFDES[package]
    stack = render_with(brd, {refdes: synth.Defect("missing")})
    m = rules.measure(stack, brd.by_refdes(refdes), golden)
    assert rule_verdict(m, golden).reason == "missing"


def test_a_well_placed_part_with_one_starved_joint_reads_insufficient(brd, golden):
    """The boundary the gate must not cross. Same single dead joint as a
    tombstone, but the body has not moved, so this is a solder fault."""
    for pad in (0, 1):
        stack = render_with(brd, {"R3": synth.Defect("insufficient", pad=pad)})
        m = rules.measure(stack, brd.by_refdes("R3"), golden)
        verdict = rule_verdict(m, golden)
        assert verdict.reason == "insufficient", (
            f"starved pad {pad} reported as {verdict.reason!r}; the body is on "
            f"its footprint (ncc={m.ncc:.3f}) so this is not a placement fault")


# ---------------------------------------------------------------- cold start
def test_untaught_package_still_uses_the_absolute_threshold(brd, golden):
    """The cold-start path is unchanged by the package-relative gate.

    With no golden entry there is no distribution to be relative to, so the
    absolute constant is still the only thing available. This pins that the
    rewrite left that branch alone; what the branch actually does with an
    untaught package is a separate problem, recorded in test_known_limits.py.
    """
    empty = rules.Golden()
    m = rules.measure(render_with(brd, {"R1": synth.Defect("missing")}),
                      brd.by_refdes("R1"), empty)
    assert m.ncc < config.NCC_FAIL_UNTAUGHT
    assert rule_verdict(m, empty).reason == "missing"


def test_package_relative_gate_is_strictly_tighter_than_the_old_constant(golden):
    """Every taught package now gates well above the 0.62 that used to be used
    for all of them. This is the sentence the whole change comes down to."""
    for name, g in golden.packages.items():
        assert g.ncc_fail > config.NCC_FAIL_UNTAUGHT, (
            f"{name} gates at {g.ncc_fail:.3f}, no tighter than the old global "
            f"{config.NCC_FAIL_UNTAUGHT}; the fix has been undone")


# --------------------------------------------------------------- gate headroom
def test_good_parts_keep_real_headroom_against_the_gate(brd, golden):
    """The gate must not sit close to the good-part distribution.

    Measured over fresh boards rather than the teach set, because teach
    understates the spread badly: the worst good 0402 sits 7.4 teach-sigma below
    the golden mean, and the true spread runs up to 2.2x the teach estimate. A
    threshold placed off teach statistics alone would look safe and not be.
    """
    worst = 0.0
    for seed in (5001, 5002, 5003, 5004, 5005):
        stack = synth.render(brd, opts=synth.RenderOpts(seed=seed))
        stack, _, _ = pipeline.align_board(stack, brd)
        for c in brd.components:
            m = rules.measure(stack, c, golden)
            worst = max(worst, golden.packages[c.package].ncc_fail / max(m.ncc, 1e-9))
    assert worst < 0.88, (
        f"the closest good part came within {(1 - worst) * 100:.0f}% of tripping "
        "the presence gate; that is where false calls start")


def test_the_gate_is_not_lopsided(golden):
    """The threshold sits between two populations and should be roughly centred.

    Being generous on one side buys nothing and costs the other. The old 0.85
    ran 12% against false calls and 25% against missed detections, which is
    detection margin nobody asked for paid for with headroom that was needed.
    """
    # 0402 tombstone is the highest-scoring defect whose LABEL depends on this
    # gate firing, so it is the real lower bound. Measured at 0.638 of golden.
    lower, upper = 0.638, 0.961
    f = config.NCC_FAIL_FRACTION
    assert lower < f < upper, \
        f"NCC_FAIL_FRACTION {f} is outside the usable window [{lower}, {upper}]"

    headroom = (upper - f) / upper
    detection = (f - lower) / f
    assert headroom > 0.15, f"only {headroom:.0%} headroom against false calls"
    assert detection > 0.15, f"only {detection:.0%} margin against missed detections"
    assert 0.5 < headroom / detection < 2.0, (
        f"lopsided: {headroom:.0%} headroom vs {detection:.0%} detection margin")


def test_a_wide_spread_package_gets_a_looser_gate(golden):
    """The sigma safety net.

    Inert on synthetic data, where teach std runs 0.0008 to 0.0125 and the
    fraction always binds. It exists for a real package whose correlation
    genuinely scatters, where a fixed fraction would put the gate inside the
    good distribution. Whichever rule is looser has to win.
    """
    tight = replace(next(iter(golden.packages.values())), ncc_mean=0.95, ncc_std=0.002)
    wide = replace(tight, ncc_std=0.05)

    assert tight.ncc_fail == pytest.approx(0.95 * config.NCC_FAIL_FRACTION), \
        "with a narrow spread the fraction should bind"
    assert wide.ncc_fail < tight.ncc_fail, (
        f"a package scattering at std=0.05 gates at {wide.ncc_fail:.3f}, no "
        f"looser than one at std=0.002 ({tight.ncc_fail:.3f})")

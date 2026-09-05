"""Fillet grading: Mahalanobis distance over the three slope bands.

Replaces a z-score on the low band alone, which was the only source of false
calls in the rule layer (3 of 3, over 1080 good-component inspections).

The three bands are correlated at r = 0.5 to 0.85 on good joints, and exploiting
that correlation is the entire point. Anything that changes how much light
reaches a component moves all three bands together, along the axis where the
covariance is large, so it costs almost no distance. A starved joint loses the
steep face while keeping the shallow one, which moves across that axis where the
covariance is small, so it costs a lot. A univariate test cannot tell those
apart, and illumination non-uniformity is the largest single source of false
calls in a rule-based AOI layer.
"""
from __future__ import annotations

import numpy as np
import pytest

from aoi import config, pipeline, rules, synth
from aoi.fusion import rule_verdict

from conftest import render_with

PACKAGES = ["0402", "0603", "0805", "SOD-123", "TANT-A"]
REFDES = {"0402": "R1", "0603": "R3", "0805": "R5",
          "SOD-123": "D1", "TANT-A": "C1"}


# ------------------------------------------------------- what teach now records
def test_golden_records_a_band_covariance_per_pad(golden):
    for name, g in golden.packages.items():
        assert g.band_cov_inv.shape == (2, 3, 3), \
            f"{name} band_cov_inv is {g.band_cov_inv.shape}, expected (2, 3, 3)"
        for pad in range(2):
            M = g.band_cov_inv[pad]
            assert np.allclose(M, M.T, atol=1e-8), \
                f"{name} pad {pad} inverse covariance is not symmetric"
            assert np.all(np.linalg.eigvalsh(M) > 0), \
                f"{name} pad {pad} inverse covariance is not positive definite"


def test_regularisation_keeps_the_inverse_bounded(golden):
    """The reason the ridge exists.

    Teach runs on a handful of boards and the bands are near-collinear, which
    leaves one almost-collapsed axis. Without shrinkage its tiny eigenvalue
    inverts to an enormous one and a negligible deviation along it scores as a
    huge distance -- turning the test meant to cut false calls into a source of
    them. A bounded condition number is what stops that.
    """
    for name, g in golden.packages.items():
        for pad in range(2):
            cond = np.linalg.cond(g.band_cov_inv[pad])
            assert cond < 1e4, (
                f"{name} pad {pad} inverse covariance is ill-conditioned "
                f"(cond={cond:.1e}); a near-collapsed axis will manufacture "
                "false calls")


def test_the_distance_uses_all_three_bands_not_just_the_low_one(brd, golden):
    """If only the low band moved the needle, this would still be the univariate
    test wearing a covariance matrix."""
    g = golden.packages["0603"]
    base = np.asarray(g.band_mean, np.float32).copy()

    for band, name in ((1, "mid"), (2, "high")):
        probe = base.copy()
        probe[0, band] += 0.15                     # perturb one pad, one band
        d = rules.band_distance(probe, g)[0]
        assert d > 1.0, \
            f"a 0.15 shift in the {name} band alone moves the distance only {d:.2f}"


# ------------------------------------------------------------------ separation
@pytest.mark.parametrize("package", PACKAGES)
@pytest.mark.parametrize("pad", [0, 1])
def test_starved_joints_are_caught_on_either_pad(brd, golden, package, pad):
    refdes = REFDES[package]
    stack = render_with(brd, {refdes: synth.Defect("insufficient", pad=pad)})
    m = rules.measure(stack, brd.by_refdes(refdes), golden)
    assert rule_verdict(m, golden).reason == "insufficient", \
        f"{package} pad {pad} starved joint scored d={m.band_maha:.1f}"


def test_good_and_starved_joints_are_far_apart(brd, golden):
    """The margin the threshold sits in. If these populations ever approach each
    other the threshold is doing real work and needs re-deriving, rather than
    sitting in an empty gap as it does now."""
    good = []
    for seed in (301, 302, 303):
        stack = synth.render(brd, opts=synth.RenderOpts(seed=seed))
        for c in brd.components:
            m = rules.measure(stack, c, golden)
            good.extend(rules.band_distance(m.bands, golden.packages[c.package]))

    starved = []
    for package in PACKAGES:
        refdes = REFDES[package]
        for pad in (0, 1):
            stack = render_with(brd, {refdes: synth.Defect("insufficient", pad=pad)})
            m = rules.measure(stack, brd.by_refdes(refdes), golden)
            starved.append(rules.band_distance(m.bands, golden.packages[package]).max())

    worst_good, weakest_starved = max(good), min(starved)
    assert worst_good < config.BAND_MAHALANOBIS_FAIL, (
        f"a good joint reaches d={worst_good:.2f}, at or past the "
        f"{config.BAND_MAHALANOBIS_FAIL} threshold")
    assert weakest_starved > config.BAND_MAHALANOBIS_FAIL, (
        f"the weakest starved joint only reaches d={weakest_starved:.2f}")
    assert weakest_starved / max(worst_good, 1e-9) > 3.0, (
        f"only {weakest_starved / worst_good:.1f}x between the worst good joint "
        f"({worst_good:.2f}) and the weakest starved one ({weakest_starved:.2f})")


# ----------------------------------------------------- the property it exists for
@pytest.mark.parametrize("vignette,gain,label", [
    (0.12, 1.00, "vignette 0.12"),
    (0.06, 0.92, "LED output down 8%"),
    (0.12, 0.95, "vignette 0.12 with LED output down 5%"),
])
def test_illumination_drift_does_not_manufacture_false_calls(brd, golden, vignette, gain, label):
    """Teach happens once, under one lighting condition; the machine then runs
    for months while LEDs age and the dome drifts. Common-mode brightness
    changes move all three bands together, so they must stay cheap in distance.

    The univariate test this replaced went from 0.37% to 0.74% false calls under
    the LED-drift case here. That is the difference between a machine operators
    use and one they click through.
    """
    flagged = []
    for i in range(3):
        stack = synth.render(brd, opts=synth.RenderOpts(
            seed=7100 + i, vignette=vignette, gain=gain))
        stack, _, _ = pipeline.align_board(stack, brd)
        for c in brd.components:
            m = rules.measure(stack, c, golden)
            v = rule_verdict(m, golden)
            if v.failed and v.reason == "insufficient":
                flagged.append((c.refdes, c.package, round(m.band_maha, 2)))
    assert not flagged, \
        f"{len(flagged)} false 'insufficient' calls under {label}: {flagged[:6]}"


# ------------------------------------------------------------- the one-sided guard
def test_a_heavy_fillet_is_not_reported_as_insufficient(brd, golden):
    """Mahalanobis distance is symmetric, so a joint carrying MORE solder than
    golden sits just as far from the mean as a starved one. Reporting that as
    insufficient solder names the opposite fault and sends the operator to the
    wrong process fix. Grading excess is a separate test that does not exist yet.
    """
    comp = brd.by_refdes("R3")
    m = rules.measure(render_with(brd, {}), comp, golden)
    g = golden.packages[comp.package]

    # push both pads well above golden in every band
    m.bands = np.clip(np.asarray(g.band_mean, np.float32) * 1.6, 0.0, 1.0)
    d = rules.band_distance(m.bands, g)
    assert d.max() > config.BAND_MAHALANOBIS_FAIL, (
        "test is not exercising the guard: an over-wetted joint scored only "
        f"d={d.max():.2f}, below the threshold anyway")

    verdict = rule_verdict(m, golden)
    assert verdict.reason != "insufficient", (
        f"a joint at 160% of golden in every band was reported as insufficient "
        f"solder (d={d.max():.2f})")

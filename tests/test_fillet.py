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


# ------------------------------------------------- covariance regularisation
# The docstring on `_band_cov_inv` calls the regularisation "load-bearing, not
# hygiene", and until now nothing held it to that. These pin the two properties
# the claim rests on: without shrinkage a near-collapsed axis blows the distance
# up, and with it the axis stays usable while the correlation that makes the
# test work is preserved.
def _golden_like(samples):
    """A GoldenPackage carrying one pad's band statistics."""
    g = object.__new__(rules.GoldenPackage)
    g.band_mean = samples.mean(axis=0)[None, :]
    g.band_std = np.maximum(samples.std(axis=0), config.BAND_COV_FLOOR)[None, :]
    g.band_cov_inv = rules._band_cov_inv(samples)[None, :, :]
    return g


BAND_MEAN = np.array([0.42, 0.33, 0.27])


def _teachlike_samples(n=8, r=0.85, sd=0.04, seed=0):
    """Band fractions the way teach sees them: few samples, strongly correlated.

    Drawn from an explicit correlation structure rather than built up from a
    shared factor, so `r` is the correlation these samples actually have.
    README records r = 0.5 to 0.85 across the five package families measured on
    good joints; 0.85 is the tightest of those and therefore the case the
    regularisation has to survive.
    """
    corr = np.full((3, 3), r)
    np.fill_diagonal(corr, 1.0)
    cov = corr * sd ** 2
    rng = np.random.RandomState(seed)
    return np.clip(rng.multivariate_normal(BAND_MEAN, cov, size=n), 0.0, 1.0)


def _dim_distance(samples, inv, frac=0.92):
    """Mahalanobis distance of a uniformly dimmed good joint from its own mean.

    The common-mode move: vignetting, LED ageing, a different board finish.
    It has to stay cheap, and that is the whole reason the bands are scored
    jointly rather than one at a time.
    """
    d = samples.mean(axis=0) * frac - samples.mean(axis=0)
    return float(np.sqrt(max(d @ inv @ d, 0.0)))


def _plain_inv(samples):
    cov = np.cov(samples.astype(np.float64), rowvar=False)
    return np.linalg.inv(cov + 1e-12 * np.eye(3))


def test_an_unregularised_covariance_manufactures_the_false_calls_it_removes():
    """The failure the ridge exists to prevent, measured rather than asserted.

    Section 10 allows teach on as few as 5 golden boards. At that size a 3x3
    covariance over bands correlated at 0.85 leaves one near-collapsed axis, and
    a plain inverse turns its small eigenvalue into an enormous distance -- so a
    good joint under nothing worse than an 8% dim scores as though it were
    starved, and the test written to cut false calls generates them.

    Measured over 400 draws at n=5: unregularised, the 99th percentile distance
    for that dim is 12.7 against a threshold of 6.0. Regularised, the worst of
    all 400 is 3.3.
    """
    plain, ridged = [], []
    for seed in range(400):
        s = _teachlike_samples(5, r=0.85, seed=seed)
        plain.append(_dim_distance(s, _plain_inv(s)))
        ridged.append(_dim_distance(s, rules._band_cov_inv(s)))

    fail = config.BAND_MAHALANOBIS_FAIL
    assert np.percentile(plain, 99) > fail, (
        f"unregularised p99 is {np.percentile(plain, 99):.2f}, below the {fail} "
        f"threshold -- the failure this test describes is not reproducing, so "
        f"the ridge can no longer be justified from measurement")
    assert max(ridged) < fail, (
        f"a uniformly dimmed good joint scores up to {max(ridged):.2f} against "
        f"a threshold of {fail} even WITH regularisation -- the fillet test is "
        f"itself a source of false calls")
    assert max(ridged) < 0.7 * fail, (
        f"worst regularised distance {max(ridged):.2f} leaves under 30% margin "
        f"below {fail}; real joints spread wider than these")


def test_regularisation_keeps_the_starved_direction_expensive():
    """Shrinkage must not flatten the test into uselessness: a joint that loses
    its steep face while keeping the shallow one still has to score far away.

    This is the other half of the trade. Bounding the weak axis costs sensitivity
    along it, and the test is only worth having if what remains still separates
    a starved joint from a dim corner of the board by a wide margin.
    """
    for seed in range(50):
        s = _teachlike_samples(8, r=0.85, seed=seed)
        g = _golden_like(s)
        mean = s.mean(axis=0)

        dimmed = (mean * 0.92)[None, :]               # all three down together
        starved = mean.copy()
        starved[0] = 0.0                              # low band collapses only

        d_dim = float(rules.band_distance(dimmed, g)[0])
        d_starved = float(rules.band_distance(starved[None, :], g)[0])

        assert d_dim < config.BAND_MAHALANOBIS_FAIL < d_starved, (
            f"seed {seed}: common-mode dimming scores {d_dim:.2f} and a starved "
            f"joint {d_starved:.2f} against a threshold of "
            f"{config.BAND_MAHALANOBIS_FAIL} -- the two are not separated")
        assert d_starved > 4 * d_dim, (
            f"seed {seed}: starved is only {d_starved / max(d_dim, 1e-9):.1f}x "
            f"the cost of a brightness change; the covariance axis does no work")


def test_the_ridge_bounds_the_condition_number_at_every_teach_size():
    """5 to 20 golden boards, per section 10. The point of shrinking the
    off-diagonals is to bound the condition number, so that is what is checked
    -- unregularised it reaches four figures at n=5.
    """
    for n in (5, 8, 12, 20):
        worst_plain = worst_ridged = 0.0
        for seed in range(200):
            s = _teachlike_samples(n, r=0.85, seed=seed)
            worst_plain = max(worst_plain, np.linalg.cond(np.cov(s, rowvar=False)))
            inv = rules._band_cov_inv(s)
            assert np.all(np.isfinite(inv)), f"non-finite inverse at n={n}, seed={seed}"
            worst_ridged = max(worst_ridged, np.linalg.cond(inv))
        assert worst_ridged < 50.0, (
            f"n={n}: regularised condition number reaches {worst_ridged:.1f}; a "
            f"tiny deviation along the weak axis will read as a defect")
        if n <= 8:
            assert worst_plain > 5 * worst_ridged, (
                f"n={n}: unregularised conditioning ({worst_plain:.1f}) is not "
                f"materially worse than regularised ({worst_ridged:.1f})")


def test_identical_golden_boards_do_not_produce_a_singular_inverse():
    """The degenerate case: every golden board measures exactly the same. Real
    enough on a synthetic renderer, and it must not divide by zero."""
    identical = np.tile(BAND_MEAN, (8, 1))
    inv = rules._band_cov_inv(identical)
    assert np.all(np.isfinite(inv)), "identical golden boards gave a singular inverse"
    g = _golden_like(identical)
    d = float(rules.band_distance(BAND_MEAN[None, :], g)[0])
    assert np.isfinite(d) and d < 1e-6, f"a joint at the golden mean scored {d}"


def test_the_floor_not_the_ridge_is_what_bounds_the_distance():
    """Which of the two regularisers does which job, pinned.

    The docstring on `_band_cov_inv` used to credit the ridge for keeping a
    good joint's distance small. Measurement says otherwise: the absolute floor
    carries almost all of that, and with the ridge alone a dimmed good joint
    still reaches 13 against a threshold of 6. The ridge earns its place on
    conditioning instead.

    If this test goes red, the attribution in that docstring is wrong again --
    fix the docstring, not the assertion.
    """
    def worst_dim(ridge, floor):
        old = (config.BAND_COV_RIDGE, config.BAND_COV_FLOOR)
        config.BAND_COV_RIDGE, config.BAND_COV_FLOOR = ridge, floor
        try:
            return max(_dim_distance(_teachlike_samples(5, r=0.85, seed=i),
                                     rules._band_cov_inv(
                                         _teachlike_samples(5, r=0.85, seed=i)))
                       for i in range(400))
        finally:
            config.BAND_COV_RIDGE, config.BAND_COV_FLOOR = old

    fail = config.BAND_MAHALANOBIS_FAIL
    shipped = worst_dim(0.15, 0.01)
    floor_only = worst_dim(0.0, 0.01)
    ridge_only = worst_dim(0.15, 1e-7)

    assert floor_only < fail, (
        f"the floor alone should hold the distance under {fail}; got {floor_only:.2f}")
    assert ridge_only > fail, (
        f"the ridge alone gives {ridge_only:.2f}, under the {fail} threshold -- "
        f"if the ridge is now sufficient on its own, the docstring's account of "
        f"which constant does what needs rewriting")
    assert abs(shipped - floor_only) < 0.5 * fail, (
        f"shipped ({shipped:.2f}) and floor-only ({floor_only:.2f}) should be "
        f"close, since the floor is what bounds this")

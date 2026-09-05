"""Regression tests for the nine traps recorded in README.md.

Each of these silently broke the pipeline before it was fixed. "Silently" is the
important word: none of them raised, none of them crashed, and every one of them
left the demo still printing a plausible-looking board report. They were found by
noticing a number was wrong, which is not a thing that reliably happens twice.

Every test below is named for the README bullet it protects and quotes it, so a
failure points straight at the paragraph explaining why the code is shaped that
way. If you are here because one is red, read the docstring before changing the
assertion.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aoi import config, models, pipeline, rules, synth
from aoi.fusion import rule_verdict

from conftest import render_with


# --------------------------------------------------------------------- trap 1
def test_trap1_placement_measures_the_body_not_the_footprint(brd, golden):
    """"The match template must be the body, not the footprint. Lands are fixed
    to the board and never move, so including them anchors the correlation peak
    at zero displacement - a 250 um offset measured as 8 um."

    Guards both halves: the golden template really is cropped down to the body,
    and a known injected offset comes back at roughly its true size.
    """
    g = golden.packages["0805"]
    assert g.body.shape[0] < g.template.shape[0], \
        "body template is not cropped in Y; the lands are still inside it"
    assert g.body.shape[1] < g.template.shape[1], \
        "body template is not cropped in X; the lands are still inside it"

    comp = brd.by_refdes("R5")
    stack = render_with(brd, {"R5": synth.Defect("offset", dx=0.25)})
    m = rules.measure(stack, comp, golden)
    off = math.hypot(m.dx_um, m.dy_um)
    assert 170.0 <= off <= 330.0, (
        f"250 um offset measured as {off:.0f} um. A near-zero reading means the "
        "correlation is anchored on the lands, which never move.")


# --------------------------------------------------------------------- trap 2
def test_trap2_skew_needs_a_masked_correlation(brd, golden):
    """"Skew needs a masked correlation. Rotating the region drags the lands
    round with it [...] Either way the score argues for 0 deg."

    The discriminating check is the pair: an unrotated part must read ~0 and a
    rotated one must read its true angle. An unmasked correlation returns ~0 for
    both, so testing only the skewed case would pass on a broken implementation
    if the search happened to land anywhere near the right answer.
    """
    comp = brd.by_refdes("R5")

    straight = rules.measure(render_with(brd, {}), comp, golden)
    assert abs(straight.dtheta_deg) <= 3.0, \
        f"a correctly placed part reports {straight.dtheta_deg:+.1f} deg of skew"

    skewed = rules.measure(
        render_with(brd, {"R5": synth.Defect("skew", dtheta=18.0)}), comp, golden)
    assert abs(abs(skewed.dtheta_deg) - 18.0) <= 4.0, (
        f"18 deg skew measured as {skewed.dtheta_deg:+.1f} deg. A reading near "
        "zero means the template rotation is dragging the lands with it.")


# --------------------------------------------------------------------- trap 3
def test_trap3_rotation_search_extends_past_the_tolerance_it_tests():
    """"Rotation search must extend past the tolerance it tests. A sweep
    stopping at +/-12 deg reports a 20 deg skew as exactly 12.0 and never trips
    a 12 deg limit."
    """
    reach = max(abs(d) for d in rules.ROT_SEARCH_DEG)
    assert reach > config.MAX_SKEW_DEG, (
        f"rotation search reaches {reach:.0f} deg but the skew limit is "
        f"{config.MAX_SKEW_DEG:.0f} deg; a skew past the limit can only ever be "
        "reported as the limit, so it never fails")


def test_trap3_skew_past_the_limit_actually_trips_it(brd, golden):
    """The functional half of trap 3: a 20 deg skew has to exceed the 12 deg
    threshold, not saturate at it."""
    comp = brd.by_refdes("R5")
    m = rules.measure(render_with(brd, {"R5": synth.Defect("skew", dtheta=20.0)}),
                      comp, golden)
    assert abs(m.dtheta_deg) > config.MAX_SKEW_DEG, (
        f"20 deg skew reported as {m.dtheta_deg:+.1f} deg, which does not exceed "
        f"the {config.MAX_SKEW_DEG:.0f} deg limit")


# --------------------------------------------------------------------- trap 4
def test_trap4_body_and_solder_albedo_are_genuinely_separated():
    """"Body and solder need genuinely different albedo. Set them close and the
    diffuse channel becomes a near-uniform blob with no structure to match."

    Without contrast here the body template has nothing to correlate on, and
    placement measurement quietly returns zero displacement for every part.
    """
    gap = abs(synth.BODY[0] - synth.SOLDER[0])
    assert gap >= 0.20, (
        f"body albedo {synth.BODY[0]} and solder albedo {synth.SOLDER[0]} differ "
        f"by only {gap:.2f}; the diffuse channel loses the structure the body "
        "template needs")


def test_trap4_diffuse_channel_keeps_the_structure_the_template_needs(brd, golden):
    """The measured consequence of trap 4.

    The README's failure chain is albedo -> structure -> correlation: if body and
    solder sit at similar albedo "the body match template loses its structure,
    and placement measurement silently returns zero displacement". NCC is the
    right probe for that, not body_contrast. Body contrast is a signed
    difference against the whole-region mean and is legitimately negative for a
    dark body sitting next to bright solder, so it does not separate present
    from absent on its own.
    """
    comp = brd.by_refdes("R5")
    present = rules.measure(render_with(brd, {}), comp, golden)
    absent = rules.measure(render_with(brd, {"R5": synth.Defect("missing")}),
                           comp, golden)

    assert present.ncc > 0.85, \
        f"a correctly placed part correlates at only {present.ncc:.3f}"
    assert absent.ncc < 0.62, \
        f"a missing part still correlates at {absent.ncc:.3f}"
    assert present.ncc - absent.ncc > 0.30, (
        f"present and missing are only {present.ncc - absent.ncc:.3f} apart in "
        "NCC; the diffuse channel has lost the structure the template matches on")


# --------------------------------------------------------------------- trap 5
@pytest.mark.parametrize("refdes,defect,expected", [
    ("R1", synth.Defect("missing"), "missing"),
    ("R6", synth.Defect("bridge"), "bridge"),
    ("R2", synth.Defect("tombstone", pad=1), "tombstone"),
    ("R3", synth.Defect("offset", dx=0.25), "offset"),
    ("R5", synth.Defect("skew", dtheta=20.0), "skew"),
    ("D1", synth.Defect("reversed"), "reversed"),
    ("R4", synth.Defect("insufficient", pad=1), "insufficient"),
])
def test_trap5_rules_are_tested_in_precedence_order(brd, golden, refdes, defect, expected):
    """"Rules are tested in precedence order, not by severity. A part that is
    absent, bridged or displaced makes its fillet windows meaningless, and the
    fillet test fires hard on all of them - so scoring by severity labels every
    defect 'insufficient solder'."

    Every defect has to come back under its own name. The failure this catches
    is not a missed defect, it is a correct detection with the wrong label,
    which sends the operator to the wrong process fix.
    """
    stack = render_with(brd, {refdes: defect})
    m = rules.measure(stack, brd.by_refdes(refdes), golden)
    verdict = rule_verdict(m, golden)

    assert verdict.failed, f"{expected} on {refdes} was not flagged at all"
    assert verdict.reason == expected, (
        f"{refdes} injected as {expected!r} but reported as {verdict.reason!r}. "
        "Gross faults must gate the joint tests, not compete with them on severity.")


# --------------------------------------------------------------------- trap 6
def test_trap6_alignment_is_reported_at_the_board_centre(brd):
    """"Report alignment at the board centre. The raw affine translation is
    measured from the pixel origin and is dominated by the rotation lever arm."

    The tolerance matters here. On this board a 0.6 deg rotation puts roughly
    0.11 mm of lever-arm error into an origin-referenced translation, so the
    0.08 mm bound below is what separates a centre-referenced report from a raw
    one. Loosen it and the test stops testing anything.
    """
    stack = synth.render(brd, opts=synth.RenderOpts(
        seed=42, board_shift_mm=(0.40, -0.30), board_rot_deg=0.6))
    _, shift, rot = pipeline.align_board(stack, brd)

    assert abs(abs(shift[0]) - 0.40) < 0.08, \
        f"x correction {shift[0]:+.3f} mm does not match the 0.40 mm injected"
    assert abs(abs(shift[1]) - 0.30) < 0.08, \
        f"y correction {shift[1]:+.3f} mm does not match the 0.30 mm injected"
    assert abs(abs(rot) - 0.6) < 0.2, \
        f"rotation correction {rot:+.3f} deg does not match the 0.60 deg injected"


# --------------------------------------------------------------------- trap 7
def test_trap7_result_hands_back_the_registered_stack(brd, golden):
    """"Draw the defect map on the registered stack. Boxing CAD positions over
    the raw capture puts every box off its component by exactly the
    misregistration the pipeline just corrected - which reads to an operator as
    a broken machine."
    """
    stack = synth.render(brd, opts=synth.RenderOpts(
        seed=43, board_shift_mm=(0.40, -0.30), board_rot_deg=0.6))
    res = pipeline.Inspector(golden).inspect(stack, brd)

    assert res.aligned is not None, "Result.aligned is None; the defect map has "\
        "nothing to draw on but the raw capture"
    assert res.aligned.shape == stack.shape
    assert not np.array_equal(res.aligned, stack), (
        "Result.aligned is the raw capture. Every box on the defect map will sit "
        "off its component by the misregistration that was just corrected.")


# --------------------------------------------------------------------- trap 8
def test_trap8_anomaly_subspace_is_per_package_never_global(good_patches):
    """"Anomaly models are per package family, never global. [...] A 0402 and a
    tantalum share no appearance, so one model asked to call both normal ends up
    calling almost anything normal."

    Fit the subspace on one package and score another through it. A genuinely
    package-specific model puts the foreign package far outside its subspace; a
    global one scores both about the same, which is the failure mode that cost
    an order of magnitude of separation.
    """
    torch = pytest.importorskip("torch")

    train = good_patches["0402"][:-2]
    held_out = good_patches["0402"][-2:]
    foreign = good_patches["0805"]

    mu, V = models.fit_subspace(train, k=4)
    net = models.build_subspace_anomaly(mu, V)

    def score(p):
        with torch.no_grad():
            return float(net(torch.from_numpy(models.to_tensor(p))).mean())

    same, other = score(held_out), score(foreign)
    assert other > 3.0 * same, (
        f"a 0805 scores {other:.4f} against the 0402 subspace while a held-out "
        f"0402 scores {same:.4f}. The subspace is not package-specific, so one "
        "model is being asked to call every package normal.")


# --------------------------------------------------------------------- trap 9
def test_trap9_measure_leaves_the_anomaly_column_at_zero(brd, golden):
    """"Backfill the anomaly column before fitting fusion. `dataset` builds
    feature vectors before any model exists, so that column is zeros on disk."

    This asserts the precondition rather than the fix: measurement genuinely
    cannot fill the anomaly column, so anything fitting the fusion tree on
    freshly built features has to backfill it first or train on a dead input.
    """
    i = rules.FEATURE_NAMES.index("anomaly")
    m = rules.measure(render_with(brd, {}), brd.components[0], golden)
    assert m.vector()[i] == 0.0, (
        "rules.measure now fills the anomaly column. If that is deliberate, the "
        "backfill in cli.cmd_train is dead code and should go; if not, this is "
        "the bug.")


def test_trap9_inspector_fills_the_anomaly_column_from_the_backend(brd, golden):
    """The other half: once a backend exists, its score has to reach the feature
    vector the fusion tree consumes."""
    sentinel = 4.25

    class ConstantBackend(models.Backend):
        def classify(self, patches):
            return np.zeros((len(patches), 2), np.float32)

        def anomaly(self, patches, packages=None):
            return np.full(len(patches), sentinel, np.float32)

    res = pipeline.Inspector(golden, ConstantBackend()).inspect(
        render_with(brd, {}), brd)
    i = rules.FEATURE_NAMES.index("anomaly")

    assert res.measures, "inspect produced no measurements"
    for m in res.measures:
        assert abs(m.vector()[i] - sentinel) < 1e-5, (
            f"{m.refdes} carries anomaly {m.vector()[i]:.3f}, not the backend's "
            f"{sentinel}. The fusion tree would train on a column of zeros.")

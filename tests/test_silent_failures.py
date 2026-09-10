"""Failures the suite could not see, from a review and from an audit after it.

Same character as the traps in `test_traps.py` -- nothing raised, nothing
crashed, and the demo kept printing a plausible board report -- but these lived
in paths the existing tests never entered: a verdict only reachable when the
fusion tree disagrees with the rules, a window only reachable when it falls off
the crop, a channel only reachable when a real frame is dark, an import nothing
in the repository called, a training split whose damage shows up as a number
rather than as an error, and a search bound that was configured and never
applied.

Each test is named for the failure it prevents and states what the broken
behaviour actually was, so a red one points at the reason rather than at the
assertion.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from aoi import board as B
from aoi import config, models, pipeline, roi, rules, synth
from aoi.fusion import Verdict


# ------------------------------------------------------- 1 · verdict coherence
class _AlwaysFail:
    """Stands in for a fusion tree that disagrees with every rule verdict."""
    def score(self, X):
        return np.ones(len(X), np.float32)


class _AlwaysPass:
    def score(self, X):
        return np.zeros(len(X), np.float32)


def _inspect_with(fusion_model, defects=None):
    brd = B.random_board(24.0, 18.0, seed=4)
    golden = rules.fit_golden(
        [synth.render(brd, opts=synth.RenderOpts(seed=200 + i)) for i in range(5)], brd)
    stack = synth.render(brd, defects=defects or {}, opts=synth.RenderOpts(seed=7))
    insp = pipeline.Inspector(golden, models.load_backend("none", None), fusion_model)
    return insp.inspect(stack, brd)


def test_a_verdict_never_says_failed_while_naming_no_fault():
    """`failed` came from the tree and `reason` came from the rules, so the two
    contradicted each other in exactly the case the tree exists for.

    A component the tree flagged and the rules cleared was stored
    `failed=True, reason="ok"`. `defect_map` looks the reason up in its colour
    table, so the box was drawn in the PASS colour and labelled `ok`, while the
    review queue served it to an operator with no reason on screen to adjudicate.
    """
    res = _inspect_with(_AlwaysFail())
    bad = [v for v in res.verdicts if v.failed and v.reason == "ok"]
    assert not bad, (
        f"{len(bad)} verdict(s) marked failed while naming no fault, e.g. "
        f"{bad[0].refdes}: the defect map will draw these as passing parts")

    # and the reason it does give has to be one the defect map can colour
    for v in res.verdicts:
        if v.failed:
            assert v.reason in pipeline._COL, (
                f"reason {v.reason!r} has no colour, so `_COL.get` falls back to "
                f"the pass colour and a flagged part is drawn as though it passed")


def test_a_cleared_verdict_never_carries_a_fault_name():
    """The converse: a rule fail the tree clears was stored `failed=0` with the
    rule's reason still attached, and drawn in that fault's colour."""
    defects = {"R3": synth.Defect(kind="insufficient", pad=0)}
    res = _inspect_with(_AlwaysPass(), defects)
    bad = [v for v in res.verdicts if not v.failed and v.reason not in ("ok", "untaught")]
    assert not bad, (
        f"{len(bad)} passing verdict(s) still name a fault, e.g. "
        f"{bad[0].refdes}={bad[0].reason!r}")


# --------------------------------------------------------- 2 · off-crop windows
def test_a_window_entirely_off_the_crop_measures_nothing():
    """The slice start was clamped and the stop was not.

    A negative stop is legal Python and numpy reads it as an offset from the far
    end, so a window off the left edge returned `slice(0, -29)` -- 171 columns of
    a 200-column crop instead of nothing. Every caller guards with
    `win.size == 0`, and that guard could never fire, so the test silently
    measured most of the crop rather than skipping.
    """
    shape = (100, 200)
    a = np.zeros(shape, np.float32)
    for centre in ([-1.6, 0.0], [1.6, 0.0], [0.0, -1.0], [0.0, 1.0]):
        ys, xs = roi.local_box_px(np.array(centre), np.array([0.05, 0.05]), shape)
        assert a[ys, xs].size == 0, (
            f"window at {centre} mm is off the crop but returned "
            f"{a[ys, xs].shape} -- the size==0 guard cannot fire")


def test_a_window_partly_on_the_crop_keeps_only_the_part_that_overlaps():
    """The fix must not turn a partial overlap into nothing."""
    shape = (100, 200)
    a = np.zeros(shape, np.float32)
    ys, xs = roi.local_box_px(np.array([-1.15, 0.0]), np.array([0.2, 0.2]), shape)
    win = a[ys, xs]
    assert win.size > 0, "a window overlapping the crop edge must still measure"
    assert win.shape[1] < 2 * 0.2 / config.MM_PER_PX, "it must be clipped, not whole"


# ------------------------------------------------- 3 · one scale, not per channel
def test_a_dark_channel_does_not_come_back_255x_too_bright():
    """`_norm` guessed 8-bit-or-not per array, and `specular_max` called it on
    each angled channel separately.

    A channel dark enough never to exceed 1.5 -- an underexposed high-angle
    frame, which is what the physics gate expects those frames to look like --
    escaped the divide its siblings received. `specular_max` then read 1.0
    everywhere, `body_blob` found no body anywhere, and every component on the
    board reported as `missing`.
    """
    r = np.zeros((10, 10, 4), np.float32)
    r[..., config.CHANNELS.index("low")] = 5.0
    r[..., config.CHANNELS.index("mid")] = 5.0
    r[..., config.CHANNELS.index("high")] = 1.0          # never exceeds 1.5

    sm = rules.specular_max(r)
    assert sm.max() < 0.1, (
        f"specular_max returned {sm.max():.3f} for an 8-bit region whose "
        f"brightest pixel is 5/255 -- the dark channel escaped normalisation")


def test_scale_is_decided_once_for_arrays_that_get_compared():
    """Whichever way the guess goes, it has to go the same way for every array
    whose values are then subtracted or maximised together."""
    eight_bit = np.full((4, 4), 200.0, np.float32)
    dark = np.full((4, 4), 1.0, np.float32)
    assert rules._norm(dark, ref=eight_bit).max() == pytest.approx(1.0 / 255.0)
    assert rules._norm(eight_bit, ref=eight_bit).max() == pytest.approx(200.0 / 255.0)


# ------------------------------------------------------------ 4 · centroid import
def test_a_board_loaded_from_a_centroid_file_can_actually_be_rendered(tmp_path):
    """`load_centroid` hard-coded board width to 0.0 and offered no way to pass
    it, so the documented entry point for the first real assembly produced a
    board that renders as a zero-width image. Nothing in the repository called
    it, which is why no test caught it.
    """
    p = tmp_path / "centroid.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["RefDes", "Package", "X", "Y", "Rotation"])
        w.writerow(["R1", "0402", 8.0, 6.0, 0])
        w.writerow(["R2", "0603", 14.0, 9.0, 90])

    brd = B.load_centroid(p, board_width=24.0, board_height=18.0)
    assert brd.width == 24.0 and brd.height == 18.0
    stack = synth.render(brd)
    assert stack.shape[1] > 0, "board rendered zero-width -- width was not carried"
    assert stack.shape[:2] == (int(18.0 / config.MM_PER_PX), int(24.0 / config.MM_PER_PX))


def test_centroid_round_trip_preserves_placements(tmp_path):
    brd = B.random_board(30.0, 20.0, seed=2)
    p = tmp_path / "rt.csv"
    B.save_centroid(brd, p)
    back = B.load_centroid(p, brd.width, brd.height, y_up=False)
    assert len(back.components) == len(brd.components)
    for a, b in zip(brd.components, back.components):
        assert a.refdes == b.refdes and a.package == b.package
        assert b.x == pytest.approx(a.x, abs=1e-3)
        assert b.y == pytest.approx(a.y, abs=1e-3)


# ----------------------------------------------- 5 · why the anomaly column is cross-fitted
def test_a_subspace_scores_its_own_training_patches_too_low():
    """The fact that makes cross-fitting necessary, pinned directly.

    `cli.py train` fills the anomaly feature for every row before fitting the
    verdict tree. Fill it from a model that was fitted on those same patches and
    their scores come back deflated, and every number the tree then reports is
    measuring the leak rather than the model. Two arrangements were tried and
    neither described the model: filling the whole column from `tr`-fitted
    models and splitting positionally reported 0.88% false calls, and splitting
    on `tr`/`va` instead reported 20.5%. Cross-fitting reports 0.00%.

    If this test fails, the leak is no longer detectable and the cross-fit in
    `cli.py` can no longer be justified from measurement -- check that before
    simplifying it away.
    """
    rng = np.random.RandomState(0)
    patches = (rng.rand(60, 96, 96, 4) * 40 + 100).astype(np.uint8)

    held_out = patches[:10]
    fitted_on_all = models.fit_subspace(patches, 16)
    fitted_without = models.fit_subspace(patches[10:], 16)

    x = torch.from_numpy(models.to_tensor(held_out))
    with torch.no_grad():
        seen = models.build_subspace_anomaly(*fitted_on_all)(x).numpy().ravel()
        unseen = models.build_subspace_anomaly(*fitted_without)(x).numpy().ravel()

    assert seen.mean() < unseen.mean(), (
        f"patches scored {seen.mean():.4f} by a model fitted on them and "
        f"{unseen.mean():.4f} by one that never saw them -- if these are equal, "
        f"in-sample scoring is no longer deflating and the cross-fit is moot")


# ------------------------------------------------ 6 · the bounded placement search
def test_the_placement_search_is_bounded_to_the_configured_window():
    """`config.SEARCH_UM` held the spec's +-0.3 mm search window and nothing
    applied it: the correlation peak was taken over the whole surface, so the
    real bound was whatever the crop margin happened to be -- incidental, and
    different for every package.

    What the bound buys is refusing a peak that is nowhere near the part. A
    bright neighbour or a stretch of bare copper can correlate better than a
    body that has genuinely moved, and an unbounded search takes it and reports
    a confident placement for the wrong object.
    """
    rng = np.random.RandomState(0)
    region = (rng.rand(200, 300) * 20).astype(np.float32)
    tpl = (rng.rand(20, 20) * 20).astype(np.float32)

    # plant a perfect match far outside the window
    far = int(config.SEARCH_UM / config.UM_PER_PX) * 3
    cx = (region.shape[1] - tpl.shape[1]) // 2
    cy = (region.shape[0] - tpl.shape[0]) // 2
    region[cy:cy + 20, cx + far:cx + far + 20] = tpl

    bound = config.SEARCH_UM / config.UM_PER_PX
    _, dx, dy = rules._match(region, tpl, max_shift_px=bound)
    assert abs(dx) <= bound + 1 and abs(dy) <= bound + 1, (
        f"peak reported at ({dx:.0f}, {dy:.0f}) px, outside the "
        f"{bound:.0f} px window it was given")

    _, udx, _ = rules._match(region, tpl)          # unbounded finds the planted one
    assert abs(udx) > bound, (
        "the planted match is inside the window, so this test is not "
        "demonstrating that the bound does anything")


def test_the_search_window_is_wider_than_the_tolerance_it_tests():
    """The same rule the rotation sweep follows. A search that stops at the
    tolerance can only ever report the tolerance, so a part outside it reads as
    exactly on the limit and never trips."""
    assert config.SEARCH_UM >= 2 * config.MAX_OFFSET_UM, (
        f"search window {config.SEARCH_UM} um against a "
        f"{config.MAX_OFFSET_UM} um tolerance leaves no room to measure a part "
        f"that is out of tolerance")
    assert max(rules.ROT_SEARCH_DEG) >= 2 * config.MAX_SKEW_DEG

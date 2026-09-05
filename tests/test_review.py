"""The operator review screen.

Driven directly rather than through the event loop: build the window, call the
handlers a key press would call, and check what reached the store. That covers
the part that can be wrong in a way nobody notices -- a decision that does not
get recorded, or gets recorded against the wrong component -- without pretending
to test what the screen looks like.

Skipped where there is no display. The station has one; a build machine may not.
"""
from __future__ import annotations

import numpy as np
import pytest

from aoi import pipeline, roi, store, synth

tk = pytest.importorskip("tkinter", reason="no Tk binding in this Python")


@pytest.fixture(scope="module")
def has_display():
    try:
        r = tk.Tk()
        r.withdraw()
        r.destroy()
    except Exception as e:                      # noqa: BLE001 - any Tk failure
        pytest.skip(f"no display available: {e}")
    return True


@pytest.fixture
def session(tmp_path, brd, golden):
    """A recorded board with fails on it, plus golden references."""
    from conftest import render_with
    defects = {"R1": synth.Defect("missing"),
               "R3": synth.Defect("offset", dx=0.25),
               "R4": synth.Defect("insufficient", pad=1)}
    stack = synth.render(brd, defects=defects,
                         opts=synth.RenderOpts(seed=63, place_sigma_um=0.0))
    res = pipeline.Inspector(golden).inspect(stack, brd)
    patches = np.stack([roi.extract_patch(res.aligned, c, c.pkg)
                        for c in brd.components])
    st = store.Store(tmp_path / "s.db")
    good = synth.render(brd, opts=synth.RenderOpts(seed=64))
    st.save_golden({c.package: roi.extract_patch(good, c, c.pkg)
                    for c in brd.components})
    bid = st.save_result(brd, res, patches)
    yield st, bid
    st.close()


def _app(st, bid):
    from aoi import review
    app = review.ReviewApp(st, bid, operator="tester")
    app.root.withdraw()             # build it, do not show it
    return app


# ------------------------------------------------------------------ rendering
def test_false_colour_puts_the_low_band_in_red():
    """The convention the whole screen rests on: a good joint sweeps blue to
    red, and a starved one has lost its red. Getting the channel order wrong
    would invert what the operator is being asked to judge."""
    from aoi import config, review
    patch = np.zeros((8, 8, 4), np.uint8)
    patch[..., config.CHANNELS.index("low")] = 255
    rgb = review.false_colour(patch)
    assert rgb[..., 0].max() == 255, "low band is not in the red plane"
    assert rgb[..., 1].max() == 0 and rgb[..., 2].max() == 0


def test_a_crop_converts_to_something_tk_can_draw(has_display):
    from aoi import review
    root = tk.Tk()
    root.withdraw()
    try:
        rng = np.random.default_rng(0)
        photo = review._to_photo(
            rng.integers(0, 256, (96, 96, 3), dtype=np.uint8), master=root)
        assert photo.width() == 96 * review.ZOOM
        assert photo.height() == 96 * review.ZOOM
    finally:
        root.destroy()


# ----------------------------------------------------------------- the flow
def test_the_screen_queues_exactly_the_failed_components(has_display, session):
    st, bid = session
    app = _app(st, bid)
    try:
        assert len(app.queue) == len(st.review_queue(bid))
        assert app.queue, "nothing queued, so this test proves nothing"
        for row in app.queue:
            assert row["failed"] == 1 and row["testable"] == 1
    finally:
        app.root.destroy()


def test_a_decision_reaches_the_store_against_the_right_component(has_display, session):
    st, bid = session
    app = _app(st, bid)
    try:
        first = app.queue[0]
        app._decide("defect")
        row = st.db.execute("SELECT * FROM adjudication WHERE verdict_id = ?",
                            (first["id"],)).fetchone()
        assert row is not None, "the decision never reached the store"
        assert row["call"] == "defect"
        assert row["operator"] == "tester"
        assert row["seconds"] is not None and row["seconds"] >= 0.0
    finally:
        app.root.destroy()


def test_working_through_the_queue_empties_it(has_display, session):
    st, bid = session
    app = _app(st, bid)
    try:
        n = len(app.queue)
        for i in range(n):
            app._decide("false_call" if i % 2 else "defect")
        assert not st.review_queue(bid), "components left unadjudicated"
        _, labels, _ = st.export_labels()
        assert len(labels) == n, "not every decision became a labelled sample"
    finally:
        app.root.destroy()


def test_skipping_leaves_the_component_for_someone_else(has_display, session):
    """Skip has to mean undecided, not silently good."""
    st, bid = session
    app = _app(st, bid)
    try:
        before = len(st.review_queue(bid))
        app._skip()
        assert len(st.review_queue(bid)) == before
    finally:
        app.root.destroy()


def test_the_screen_reports_the_board_time_against_the_target(has_display, session):
    """Section 10's exit criterion is in seconds, so the screen has to say
    whether the board met it rather than leaving an operator to guess."""
    st, bid = session
    app = _app(st, bid)
    try:
        for _ in range(len(app.queue)):
            app._decide("defect")
        assert app.i >= len(app.queue)
        assert "adjudicated" in app.reason.cget("text")
        assert "60 s" in app.detail.cget("text")
    finally:
        app.root.destroy()


def test_a_coverage_gap_is_surfaced_at_the_end(has_display, tmp_path, brd, golden):
    """An operator must not close a board believing it was fully inspected."""
    from aoi import board as B
    from aoi import rules
    from conftest import render_with
    partial = B.Board(brd.name, brd.width, brd.height,
                      [c for c in brd.components if c.package != "0603"], brd.fiducials)
    g = rules.fit_golden(
        [synth.render(partial, opts=synth.RenderOpts(seed=480 + i)) for i in range(4)],
        partial)
    res = pipeline.Inspector(g).inspect(render_with(brd, {}), brd)
    patches = np.stack([roi.extract_patch(res.aligned, c, c.pkg) for c in brd.components])
    with store.Store(tmp_path / "gap.db") as st:
        bid = st.save_result(brd, res, patches)
        app = _app(st, bid)
        try:
            app._finish()
            text = app.detail.cget("text")
            assert "NOT INSPECTED" in text and "0603" in text, text
        finally:
            app.root.destroy()

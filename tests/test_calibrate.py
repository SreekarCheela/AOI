"""Measuring the sampling pitch, and refusing to when the image cannot support it.

`config.UM_PER_PX` is a design intent. Every micron the machine reports is that
constant times a pixel count, and the error from it being wrong is systematic,
invisible and uncontradicted by anything downstream: teach and inspect share it,
so it cancels out of every golden comparison and survives only in the numbers a
person reads and the tolerances they are judged against. That is why the refusal
cases below carry as much weight as the accuracy ones -- a calibration nobody
looked at is worse than a constant somebody chose.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from aoi import calibrate, config


# ------------------------------------------------------------------- accuracy
@pytest.mark.parametrize("true_pitch", [9.6, 11.0, 12.0, 12.4, 13.5, 14.5])
def test_the_pitch_is_recovered_from_a_target_of_known_spacing(true_pitch):
    """Against a ground truth that is known exactly, which a photograph never
    is. If it cannot recover the pitch from an image this clean it will not
    recover it from glass."""
    img = calibrate.render_target(dot_pitch_mm=1.0, um_per_px=true_pitch)
    cal = calibrate.measure_pitch(img, dot_pitch_mm=1.0)
    err = abs(cal.um_per_px - true_pitch) / true_pitch
    assert err < 0.002, (
        f"measured {cal.um_per_px:.4f} against a true {true_pitch} -- "
        f"{err:.3%} error")


@pytest.mark.parametrize("dot_pitch_mm", [0.5, 1.0, 2.0, 2.5])
def test_any_certified_target_spacing_gives_the_same_answer(dot_pitch_mm):
    """The target's pitch is an input, not an assumption. A 0.5 mm grid and a
    2.5 mm grid photographed through the same optics must agree."""
    img = calibrate.render_target(dot_pitch_mm=dot_pitch_mm, um_per_px=12.0)
    cal = calibrate.measure_pitch(img, dot_pitch_mm=dot_pitch_mm)
    assert cal.um_per_px == pytest.approx(12.0, rel=0.005)


def test_the_target_renderer_places_dots_sub_pixel():
    """The test's own ground truth has to be exact. At 12 um/px a 1 mm pitch is
    83.33 px; rounding dot centres to whole pixels makes the rendered spacing 83
    and hands the measurement a 0.4% error that belongs to the target rather
    than to the method. This pins that it does not."""
    img = calibrate.render_target(dot_pitch_mm=1.0, um_per_px=12.0)
    pts = calibrate.find_dots(img)
    gaps = calibrate._nearest_neighbour_spacings(pts)
    assert float(np.median(gaps)) == pytest.approx(1000.0 / 12.0, rel=0.002)


def test_a_calibration_reports_what_it_measured():
    cal = calibrate.measure_pitch(calibrate.render_target(um_per_px=12.4), 1.0)
    assert cal.dots >= calibrate.MIN_DOTS
    assert cal.mm_per_px == pytest.approx(cal.um_per_px / 1000.0)
    text = cal.report()
    assert "um/px" in text and "dots" in text
    assert f"{cal.deviation:+.1%}" in text, "the report hides how far off design it is"


# ------------------------------------------------------------------- refusals
def test_an_image_with_no_target_is_refused():
    with pytest.raises(calibrate.CalibrationError) as e:
        calibrate.measure_pitch(np.full((300, 300), 235, np.uint8), 1.0)
    assert "dots" in str(e.value)


def test_blobs_that_are_not_a_regular_grid_are_refused():
    """The dangerous case: enough blobs to look like a measurement, not evenly
    spaced enough to be one."""
    img = calibrate.render_target(1.0, um_per_px=12.0)
    rng = np.random.RandomState(3)
    for _ in range(25):
        y = rng.randint(20, img.shape[0] - 20)
        x = rng.randint(20, img.shape[1] - 20)
        cv2.circle(img, (x, y), 12, 20, -1)
    with pytest.raises(calibrate.CalibrationError) as e:
        calibrate.measure_pitch(img, 1.0)
    assert "spacings vary" in str(e.value)


def test_a_result_far_from_the_optical_design_is_refused():
    """25% off is not a calibration, it is a detection failure -- most often the
    target's pitch typed in wrong, which otherwise silently rescales every
    micron the machine will ever report."""
    img = calibrate.render_target(dot_pitch_mm=1.0, um_per_px=12.0)
    with pytest.raises(calibrate.CalibrationError) as e:
        calibrate.measure_pitch(img, dot_pitch_mm=2.0)     # stated pitch wrong
    assert "too far to be a calibration" in str(e.value)


def test_a_nonsense_target_pitch_is_rejected_before_anything_is_measured():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            calibrate.measure_pitch(calibrate.render_target(), bad)


def test_long_thin_blobs_are_not_counted_as_dots():
    """Scratches, the edge of the target and the frame of the fixture are all
    long and thin. Counting them drags the spacing toward nonsense."""
    img = calibrate.render_target(1.0, um_per_px=12.0)
    cv2.rectangle(img, (5, 5), (img.shape[1] - 5, 12), 20, -1)     # a long bar
    before = len(calibrate.find_dots(calibrate.render_target(1.0, um_per_px=12.0)))
    after = len(calibrate.find_dots(img))
    assert after == before, "a long thin blob was counted as a dot"


# ------------------------------------------------------------------- adoption
def test_applying_a_calibration_moves_every_micron_the_machine_reports():
    """`apply` is deliberately explicit rather than automatic. This checks it
    actually takes effect, and restores the module afterwards."""
    before_um, before_mm = config.UM_PER_PX, config.MM_PER_PX
    try:
        cal = calibrate.measure_pitch(calibrate.render_target(um_per_px=12.4), 1.0)
        calibrate.apply(cal)
        assert config.UM_PER_PX == cal.um_per_px
        assert config.MM_PER_PX == pytest.approx(cal.um_per_px / 1000.0)
        assert config.UM_PER_PX != before_um, "the fixture pitch matched by chance"
    finally:
        config.UM_PER_PX, config.MM_PER_PX = before_um, before_mm


def test_the_design_intent_is_what_the_optics_were_costed_for():
    """A guard on the constant itself. Section 2 sizes the sensor, the lens, the
    tile raster and the whole bill of materials from 12 um/px; changing it
    without re-deriving those is a much larger edit than it looks."""
    assert config.UM_PER_PX == 12.0
    assert config.MM_PER_PX == pytest.approx(0.012)

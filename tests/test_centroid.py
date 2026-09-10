"""Reading the placement files real tools emit.

Every EDA tool and every contract manufacturer exports this differently, and the
failure mode is not an exception. A file whose mils are read as millimetres, or
whose title block is read as a header, produces a board full of components in
plausible wrong places and the first symptom is a false-call rate weeks later.

The core assertion below is that seven files describing the SAME board, in seven
formats, parse to the same coordinates. Anything else is a detail.

Honest limit: these samples are written from the documented shapes of the common
exports. None came from a real contract manufacturer, so what is pinned here is
"handles the documented formats", not "validated against production data".
"""
from __future__ import annotations

import pytest

from aoi import board as B
from aoi import centroid, synth


# The same two parts, seven ways. R1 at (12.70, 8.25) rot 90, C1 at
# (15.24, 8.25) rot 270, plus one bottom-side part that must be filtered out.
SAMPLES = {
    "altium": (
        'Designator,Comment,Layer,Footprint,Center-X(mm),Center-Y(mm),Rotation\n'
        '"R1","10k","TopLayer","0402",12.7000,8.2500,90.000\n'
        '"C1","100n","TopLayer","0402",15.2400,8.2500,270.000\n'
        '"U9","MCU","BottomLayer","QFP-32",20.0000,15.0000,0.000\n'),
    "kicad_pos": (
        '### Module positions - created 2026-09-10 ###\n'
        '## Unit = mm, Angle = deg.\n'
        '# Ref     Val    Package   PosX      PosY      Rot       Side\n'
        'R1        10k    0402      12.7000   8.2500    90.0000   top\n'
        'C1        100n   0402      15.2400   8.2500    270.0000  top\n'
        'U9        MCU    QFP-32    20.0000   15.0000   0.0000    bottom\n'
        '## End\n'),
    "mils_in_heading": (
        'RefDes,Package,Mid X (mil),Mid Y (mil),Rotation,Side\n'
        'R1,0402,500,324.803,90,T\n'
        'C1,0402,600,324.803,270,T\n'
        'U9,QFP-32,787.402,590.551,0,B\n'),
    "semicolon_decimal_comma": (
        'Reference;Pattern;PosX;PosY;Angle;Layer\n'
        'R1;0402;12,70;8,25;90;top\n'
        'C1;0402;15,24;8,25;270;top\n'
        'U9;QFP-32;20,00;15,00;0;bottom\n'),
    "title_block": (
        'Pick and Place Report\n'
        'Board: DEMO-1\n'
        'Units: mm\n'
        '\n'
        'Designator,Footprint,Mid X,Mid Y,Rotation,Layer\n'
        'R1,0402,12.7,8.25,90,top\n'
        'C1,0402,15.24,8.25,270,top\n'
        'U9,QFP-32,20.0,15.0,0,bottom\n'),
    "units_in_values": (
        'Ref,Package,X,Y,Rot,Side\n'
        'R1,0402,12.7mm,8.25mm,90,top\n'
        'C1,0402,600mil,324.803mil,270,top\n'
        'U9,QFP-32,20mm,15mm,0,bottom\n'),
    "tab_separated": (
        'RefDes\tPackage\tX\tY\tRotation\tSide\n'
        'R1\t0402\t12.7\t8.25\t90\ttop\n'
        'C1\t0402\t15.24\t8.25\t270\ttop\n'
        'U9\tQFP-32\t20.0\t15.0\t0\tbottom\n'),
}


@pytest.fixture
def files(tmp_path):
    out = {}
    for name, body in SAMPLES.items():
        p = tmp_path / f"{name}.txt"
        p.write_text(body)
        out[name] = p
    return out


# ------------------------------------------------------------ the core claim
@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_every_documented_format_reads_to_the_same_board(files, name):
    """One board, seven files. If any of them disagrees, the disagreement is
    silent everywhere else in the pipeline."""
    rep = centroid.read(files[name])
    assert rep.ok, f"{name} did not parse:\n{rep.summary()}"

    got = {p.refdes: p for p in rep.placements}
    assert set(got) == {"R1", "C1"}, (
        f"{name}: expected the two top-side parts, got {sorted(got)}")
    assert got["R1"].x == pytest.approx(12.70, abs=0.01)
    assert got["R1"].y == pytest.approx(8.25, abs=0.01)
    assert got["R1"].rot == pytest.approx(90.0, abs=0.01)
    assert got["C1"].x == pytest.approx(15.24, abs=0.01)
    assert got["C1"].rot == pytest.approx(270.0, abs=0.01)
    assert all(p.package == "0402" for p in got.values())


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_bottom_side_parts_are_left_out_by_default(files, name):
    """A double-sided assembly is two scans. Mixing both sides into one program
    puts half the components at coordinates mirrored with respect to the image,
    which reads as a board where half the parts are misplaced."""
    rep = centroid.read(files[name])
    assert "U9" not in {p.refdes for p in rep.placements}
    assert any("U9" in why for _, why in rep.skipped), (
        "the bottom-side part was dropped without saying so")


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_the_other_side_can_be_asked_for(files, name):
    rep = centroid.read(files[name], side="bottom")
    assert {p.refdes for p in rep.placements} == {"U9"}


# ------------------------------------------------------------------- units
def test_mils_in_a_heading_are_not_read_as_millimetres():
    """The most expensive single mistake available here: a factor of 39.4,
    applied consistently, producing a board that is internally coherent and
    entirely wrong."""
    assert centroid._unit_from("Mid X (mil)") == pytest.approx(0.0254)
    assert centroid._unit_from("Center-X(mm)") == 1.0
    assert centroid._unit_from("X [inch]") == 25.4
    assert centroid._unit_from("PosX") is None


def test_taking_the_first_word_as_the_unit_would_be_wrong():
    """`Mid X (mil)` begins with "Mid". A reader that takes the first run of
    letters finds no unit, silently assumes millimetres, and every coordinate
    comes out 39x too large."""
    assert centroid._unit_from("Mid X (mil)") != centroid._unit_from("Mid")


def test_a_unit_in_the_value_overrides_the_column():
    """A file that writes 12.7mm in one row and 600mil in the next is telling
    you something the heading cannot."""
    assert centroid._number("600mil", scale=1.0) == pytest.approx(15.24, abs=0.01)
    assert centroid._number("12.7mm", scale=0.0254) == pytest.approx(12.7, abs=0.01)


def test_a_decimal_comma_is_read_as_a_decimal_point():
    assert centroid._number("12,70", 1.0) == pytest.approx(12.70)
    assert centroid._number("-8,25", 1.0) == pytest.approx(-8.25)


def test_a_thousands_comma_is_not_read_as_a_decimal_point():
    """`1,270` is one thousand two hundred and seventy, not 1.27. Three digits
    after the comma is the tell, and getting it wrong is a factor of 1000."""
    assert centroid._number("1,270", 1.0) == pytest.approx(1270.0)


# ----------------------------------------------------------- being loud enough
def test_a_file_with_no_recognisable_columns_says_which_are_missing():
    rep = centroid.ParseReport(columns={"package": "Footprint"})
    assert set(rep.missing) == {"refdes", "x", "y"}
    assert not rep.ok


def test_rows_that_cannot_be_read_are_reported_not_dropped(tmp_path):
    """A file that half-loads has to say so. Silently inspecting two thirds of
    a board is the failure this whole module exists to avoid."""
    p = tmp_path / "ragged.csv"
    p.write_text('RefDes,Package,X,Y,Rotation,Side\n'
                 'R1,0402,12.7,8.25,90,top\n'
                 'R2,0402,,8.25,90,top\n'          # no x
                 ',0402,1.0,1.0,0,top\n'           # no designator
                 'R4,,5.0,5.0,0,top\n')            # no package
    rep = centroid.read(p)
    assert len(rep.placements) == 1
    assert len(rep.skipped) == 3
    text = rep.summary()
    assert "skipped 3" in text
    for token in ("R2", "no usable coordinate", "no reference designator"):
        assert token in text, f"the report does not mention {token!r}"


def test_the_unit_source_is_always_stated():
    """Whether the unit was read or assumed is the difference between a
    measurement and a guess, and the report must not hide which."""
    rep = centroid.ParseReport(unit_scale=1.0, unit_source="assumed -- no unit")
    assert "assumed" in rep.summary()


# ---------------------------------------------------------------- integration
def test_a_board_loaded_from_a_real_shaped_export_renders(tmp_path):
    p = tmp_path / "altium.csv"
    p.write_text(SAMPLES["altium"])
    brd = B.load_centroid(p, board_width=30.0, board_height=20.0)
    assert [c.refdes for c in brd.components] == ["R1", "C1"]
    assert brd.components[0].value == "10k"
    stack = synth.render(brd)
    assert stack.shape[1] > 0


def test_a_file_missing_a_required_column_raises_rather_than_half_loading(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("Designator,Footprint,Rotation\nR1,0402,90\n")
    with pytest.raises(ValueError) as e:
        B.load_centroid(p, 30.0, 20.0)
    assert "x" in str(e.value) and "y" in str(e.value)


def test_extent_covers_every_placement(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text(SAMPLES["altium"])
    rep = centroid.read(p)
    x0, y0, w, h = centroid.extent(rep, margin_mm=2.0)
    for pl in rep.placements:
        assert x0 <= pl.x <= x0 + w
        assert y0 <= pl.y <= y0 + h

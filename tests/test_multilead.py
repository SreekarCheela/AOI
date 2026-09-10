"""Packages with more than two terminals.

Land geometry used to be a pair: two centres at +-pad_pitch/2, a bands array of
shape (2, 3), `for pad in range(2)` in the verdict. That put every IC, connector
and QFP out of scope, which meant the pipeline could not run on a real assembly
at all -- passives are most of a board by count and none of what makes it work.

Lands are now a list generated from row descriptors, and everything downstream
indexes that list. A chip is the degenerate case of one land a side.
"""
from __future__ import annotations

import collections

import numpy as np
import pytest

from aoi import board as B
from aoi import packages, pipeline, rules, synth
from aoi.fusion import rule_verdict

MULTI = ["SOIC-8", "SOT-23-6", "QFP-32"]
CHIPS = ["0402", "0603", "0805", "SOD-123", "TANT-A"]


# --------------------------------------------------------------- the library
@pytest.mark.parametrize("name", sorted(packages.LIBRARY))
def test_every_package_is_geometrically_valid(name):
    """The guard for the class of error that cost the most time here.

    A package whose body reaches past its own land leaves nothing for solder to
    climb: the fillet window degenerates, every joint reads permanently starved,
    and the fillet test is silently dead for that package. It does not raise, it
    does not look wrong in the library, and it only shows up as a false-call
    rate. SOIC-8 and SOT-23 both shipped that way here, from having body length
    and width the wrong way round.
    """
    problems = packages.get(name).validate()
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name,expected", [
    ("0402", 2), ("TANT-A", 2), ("SOT-23-6", 6), ("SOIC-8", 8), ("QFP-32", 32)])
def test_land_counts(name, expected):
    pkg = packages.get(name)
    assert pkg.n_pads == expected
    assert len(pkg.lands()) == expected


# Nominal body dimensions and lead pitch, from the package standards. Pinned
# because the synthetic loop CANNOT catch a library that disagrees with reality:
# the renderer and the inspector read the same numbers, so a wrong package is
# perfectly self-consistent and every test passes. Only a real board, or this
# table, catches it.
DATASHEET = {
    # name:        (long, short, height, lead pitch or None)
    "0402":        (1.00, 0.50, 0.35, None),
    "0603":        (1.60, 0.80, 0.45, None),
    "0805":        (2.00, 1.25, 0.55, None),
    "SOD-123":     (2.70, 1.60, 1.00, None),
    "TANT-A":      (3.20, 1.60, 1.60, None),   # EIA 3216
    "SOIC-8":      (4.90, 3.90, 1.75, 1.27),   # JEDEC MS-012 narrow body
    "SOT-23-6":    (2.90, 1.60, 1.10, 0.95),
    "QFP-32":      (7.00, 7.00, 1.40, 0.80),   # LQFP 7x7
}


@pytest.mark.parametrize("name", sorted(DATASHEET))
def test_library_matches_the_datasheet(name):
    pkg = packages.get(name)
    long_, short, height, pitch = DATASHEET[name]
    assert max(pkg.body_l, pkg.body_w) == pytest.approx(long_), f"{name} long axis"
    assert min(pkg.body_l, pkg.body_w) == pytest.approx(short), f"{name} short axis"
    assert pkg.body_h == pytest.approx(height), f"{name} height"
    if pitch is not None:
        assert pkg.pitch == pytest.approx(pitch), f"{name} lead pitch"


def test_dual_row_leads_run_along_the_long_side():
    """The rule a swapped body_l/body_w breaks.

    On a two-row leaded package the leads are on the LONG sides, so the row
    spreads along the long axis and faces across the short one. Getting that
    backwards is geometrically valid -- the fillet run stays positive, the
    library validates, everything renders -- and simply describes a part that
    does not exist. It cost more time here than any other single error.
    """
    for name, pkg in packages.LIBRARY.items():
        if pkg.per_row < 2 or set(pkg.rows) != {"left", "right"}:
            continue
        assert pkg.body_w > pkg.body_l, (
            f"{name} rows spread along {pkg.body_w:.2f} mm and face across "
            f"{pkg.body_l:.2f} mm: the leads are on the short sides")


def test_only_two_terminal_packages_can_tombstone():
    for name in CHIPS:
        assert packages.get(name).is_two_terminal
    for name in MULTI:
        assert not packages.get(name).is_two_terminal


def test_a_chip_still_produces_exactly_the_old_two_lands():
    """The degenerate case has to stay exactly what it was, or every number
    measured before this change stops meaning anything."""
    pkg = packages.get("0402")
    d = pkg.pad_pitch / 2.0
    got = [(l.cx, l.cy, l.hx, l.hy) for l in pkg.lands()]
    assert got == [(-d, 0.0, pkg.pad_l / 2, pkg.pad_w / 2),
                   (+d, 0.0, pkg.pad_l / 2, pkg.pad_w / 2)]
    assert pkg.footprint() == pytest.approx((2.20, 1.30))


def test_lead_rows_face_the_body(name="QFP-32"):
    """Each land's axis has to point inward, or its fillet window is built
    running away from the part."""
    pkg = packages.get(name)
    for l in pkg.lands():
        inward_x = -l.cx * l.ax
        inward_y = -l.cy * l.ay
        assert inward_x + inward_y > 0, \
            f"{name} {l.row} {l.index} axis ({l.ax},{l.ay}) does not point at the body"


def test_bridges_are_between_neighbours_in_a_row():
    """On a chip the one gap is across the body. On a lead row it is every
    consecutive pair, and those are the gaps that actually bridge: a QFP-32
    leaves ~0.4 mm between leads against 6.2 mm across the body."""
    assert len(packages.get("0402").adjacent_pairs()) == 1
    assert len(packages.get("SOIC-8").adjacent_pairs()) == 6      # 2 rows of 4
    assert len(packages.get("QFP-32").adjacent_pairs()) == 28     # 4 rows of 8
    for a, b in packages.get("QFP-32").adjacent_pairs():
        assert a.row == b.row and abs(a.index - b.index) == 1


# ------------------------------------------------------- shapes through the stack
@pytest.mark.parametrize("name", sorted(packages.LIBRARY))
def test_band_and_joint_arrays_follow_the_land_count(name):
    pkg = packages.get(name)
    assert len(rules.joint_boxes(pkg)) == pkg.n_pads
    # generous enough that every land's window lands inside it
    n = int(max(pkg.footprint()) / 0.012) + 8
    region = np.zeros((n, n, 4), np.uint8)
    assert rules.band_fractions(region, pkg).shape == (pkg.n_pads, 3)


@pytest.mark.parametrize("name", sorted(packages.LIBRARY))
def test_the_feature_vector_is_the_same_width_for_every_package(name):
    """A fixed-width vector is what lets one fusion tree serve a 2-pad chip and a
    32-lead QFP. Per-land columns cannot do that, so the band evidence is
    summarised as worst-and-mean per channel instead."""
    m = rules.Measure("U1", name)
    m.bands = np.random.rand(packages.get(name).n_pads, 3).astype(np.float32)
    assert len(m.vector()) == len(rules.FEATURE_NAMES)


# ------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def mixed():
    comps = [
        B.Component("U1", "QFP-32",   x=12.0, y=12.0, rot=0.0),
        B.Component("U2", "SOIC-8",   x=26.0, y=8.0,  rot=0.0),
        B.Component("U3", "SOIC-8",   x=26.0, y=18.0, rot=90.0),
        B.Component("Q1", "SOT-23-6", x=34.0, y=8.0,  rot=0.0),
        B.Component("R1", "0402",     x=34.0, y=14.0, rot=0.0),
        B.Component("C1", "TANT-A",   x=34.0, y=19.0, rot=0.0),
    ]
    brd = B.Board("MULTI", 40.0, 26.0, comps, [(2.0, 2.0), (38.0, 2.0), (38.0, 24.0)])
    # Ten golden boards, not five, and the reason is C1. Section 10 allows 5-20;
    # a TANT-A 1.6 mm tall sitting 15 mm off the optical axis has its own body
    # image displaced 73 um across a fillet window only 200 um long, so the
    # visible part of that joint is both smaller and more sensitive to placement
    # scatter than any other joint on this board. Measured over 30 boards, the
    # worst joint distance on a good C1 runs 10.5 at five golden boards, 5.6 at
    # ten and 5.3 at twenty, against a fail threshold of 6.0.
    #
    # That is a real requirement the renderer surfaced rather than a number
    # chosen to make this file pass: tall packages away from the optical axis
    # need more teach data than short ones. It is recorded in known-limits.md.
    stacks = [synth.render(brd, opts=synth.RenderOpts(seed=500 + i)) for i in range(10)]
    return brd, rules.fit_golden(stacks, brd)


def test_teach_learns_per_land_statistics(mixed):
    brd, golden = mixed
    for name, g in golden.packages.items():
        n = packages.get(name).n_pads
        assert g.band_mean.shape == (n, 3), f"{name} band_mean is {g.band_mean.shape}"
        assert g.band_cov_inv.shape == (n, 3, 3)


def test_no_false_calls_on_good_multilead_parts(mixed):
    brd, golden = mixed
    flagged = []
    for seed in (911, 912, 913):
        stack = synth.render(brd, opts=synth.RenderOpts(seed=seed))
        stack, _, _ = pipeline.align_board(stack, brd)
        for c in brd.components:
            v = rule_verdict(rules.measure(stack, c, golden), golden)
            if v.failed:
                flagged.append((c.refdes, c.package, v.reason, v.detail))
    assert not flagged, f"{len(flagged)} false calls: {flagged[:5]}"


@pytest.mark.parametrize("refdes,package", [("U1", "QFP-32"), ("U2", "SOIC-8"),
                                            ("Q1", "SOT-23-6")])
@pytest.mark.parametrize("kind", ["missing", "offset", "insufficient", "bridge"])
def test_defects_are_caught_on_multilead_parts(mixed, refdes, package, kind):
    brd, golden = mixed
    d = synth.Defect(kind, dx=0.25 if kind == "offset" else 0.0)
    stack = synth.render(brd, defects={refdes: d},
                         opts=synth.RenderOpts(seed=11, place_sigma_um=0.0))
    v = rule_verdict(rules.measure(stack, brd.by_refdes(refdes), golden), golden)
    assert v.failed, f"{package} {kind} not caught"
    assert v.reason == kind or kind == "missing", \
        f"{package} {kind} reported as {v.reason!r}"


@pytest.mark.parametrize("refdes,package", [("U1", "QFP-32"), ("U2", "SOIC-8"),
                                            ("Q1", "SOT-23-6")])
def test_a_starved_joint_is_caught_on_any_lead(mixed, refdes, package):
    """First, middle and last lead: an index error in the land list would show
    up at one end and nowhere else."""
    brd, golden = mixed
    n = packages.get(package).n_pads
    for lead in (0, n // 2, n - 1):
        stack = synth.render(brd, defects={refdes: synth.Defect("insufficient", pad=lead)},
                             opts=synth.RenderOpts(seed=11, place_sigma_um=0.0))
        v = rule_verdict(rules.measure(stack, brd.by_refdes(refdes), golden), golden)
        assert v.reason == "insufficient", \
            f"{package} lead {lead} of {n} starved but reported {v.reason!r}"


def test_a_quad_package_is_never_called_a_tombstone(mixed):
    """A QFP body is square, so its extent along the pad axis equals its width
    and the standing-body test reads ~1.0. Without the two-terminal gate every
    QFP on the board would be flagged as a tombstone."""
    brd, golden = mixed
    stack = synth.render(brd, opts=synth.RenderOpts(seed=914))
    m = rules.measure(stack, brd.by_refdes("U1"), golden)
    from aoi import config
    assert m.body_aspect < config.BODY_STANDING_ASPECT, \
        "test is not exercising the gate: the QFP aspect is above the limit anyway"
    assert rule_verdict(m, golden).reason != "tombstone"


# --------------------------------------------------------- the layout generator
@pytest.mark.parametrize("brd", [B.random_board(40.0, 30.0, seed=1), B.mixed_board(1)],
                         ids=["passive", "mixed"])
def test_components_clear_the_fiducials_and_the_board_edge(brd):
    """Two failure modes that both surface as a board-wide false-call storm
    rather than as an error.

    A component whose metal reaches into a fiducial's +-2.5 mm search window can
    win the correlation peak and hand back a bogus board transform, which then
    displaces every crop on the board. A component whose crop runs off the image
    gets BORDER_REPLICATE fill, and the lands landing in that strip read as bare.
    Both are sized by the part, so a single grid margin cannot express them: a
    0402 needs 1.1 mm of room and a QFP-32 needs 4.9 mm.
    """
    for c in brd.components:
        half = max(c.pkg.footprint(0.35)) / 2.0
        for fx, fy in brd.fiducials:
            assert np.hypot(c.x - fx, c.y - fy) >= half, \
                f"{c.refdes} ({c.package}) reaches into the fiducial at ({fx},{fy})"
        assert half <= c.x <= brd.width - half and half <= c.y <= brd.height - half, \
            f"{c.refdes} ({c.package}) crop runs off the board"


def test_the_fiducial_keepout_actually_drops_slots():
    """The keep-out has to be exercised by a layout that would otherwise violate
    it, or the test passes for unrelated reasons.

    A board of nothing but QFP-32 puts a 4.85 mm crop half-extent in the corner
    slot, 4.03 mm from the origin fiducial. Without the check that component is
    placed, its leads sit inside the fiducial's search window, and alignment can
    lock onto them and hand back a bogus transform for the whole board.
    """
    width = height = 40.0
    brd = B.random_board(width, height, seed=3, pitch=11.0, mix=["QFP-32"])
    half = max(packages.get("QFP-32").footprint(0.35)) / 2.0

    slots = [(x, y)
             for y in B._frange(max(3.0, half), height - max(3.0, half), 11.0)
             for x in B._frange(max(3.0, half), width - max(3.0, half), 11.0)]
    assert len(brd.components) < len(slots), (
        f"all {len(slots)} grid slots were filled, so the keep-out never fired "
        "and this test proves nothing")
    for c in brd.components:
        for fx, fy in brd.fiducials:
            assert np.hypot(c.x - fx, c.y - fy) >= half, \
                f"{c.refdes} sits {np.hypot(c.x-fx, c.y-fy):.2f} mm from a fiducial"


def test_the_mixed_board_actually_carries_leaded_parts():
    got = collections.Counter(c.package for c in B.mixed_board(1).components)
    for name in MULTI:
        assert got[name] > 0, f"mixed board has no {name}; it is not testing anything"


# ------------------------------------------------- library guards, added later
def test_the_whole_library_is_validated_at_import():
    """`validate` existed and only a test ever called it, so a bad entry reached
    the pipeline and surfaced weeks later as a false-call rate on one package.
    It now runs at import, which is a second and names the problem."""
    assert packages.validate_library() == []


def test_a_swapped_length_and_width_is_caught():
    """The error that cost more time than any other on this project. SOIC-8 and
    SOT-23 both shipped with body_l and body_w swapped, putting the leads on the
    short sides -- geometrically valid, fillet runs positive, every test green,
    describing a part that does not exist.

    Synthetic data cannot catch it: the renderer and the inspector read the same
    numbers and agree with each other. Only a rule stated independently can, and
    the rule is that a multi-lead package runs its leads along its LONG axis.
    """
    for name in ("SOIC-8", "SOT-23-6"):
        good = packages.get(name)
        swapped = packages.Package(
            **{**good.__dict__, "body_l": good.body_w, "body_w": good.body_l})
        problems = swapped.validate()
        assert any("swapped" in m for m in problems), (
            f"a swapped {name} was accepted: {problems}")


def test_a_missing_or_absurd_height_is_caught():
    """Height is no longer cosmetic -- the lens perspective correction reads it,
    so a wrong one silently mis-corrects every placement for that package."""
    good = packages.get("0603")
    flat = packages.Package(**{**good.__dict__, "body_h": 0.0})
    assert any("body_h" in m for m in flat.validate())

    tower = packages.Package(**{**good.__dict__, "body_h": 20.0})
    assert any("implausible" in m for m in tower.validate())


def test_every_package_declares_a_usable_height():
    """A guard on the shipped data, not on the code. Every family the library
    carries has to have a height the correction can use."""
    for name, pkg in packages.LIBRARY.items():
        assert pkg.body_h > 0.0, f"{name} has no height"
        assert pkg.body_h < 4.0 * max(pkg.body_l, pkg.body_w), f"{name} height implausible"

# Grading excess

The one-sided guard in [grading-the-fillet.md](grading-the-fillet.md) stops a
*uniformly heavier* fillet being called starved. It does nothing for a
**balled** joint, which is the shape excess solder actually takes, and which
fails the starvation test's own condition.

Solder wets outward only as far as flux and land allow. Volume past that piles
up, and the surface inverts: a concave ramp — shallow at the toe, steep at the
termination — becomes a convex cap, flat over its apex and steep only at the
rim. The cap costs the mid-slope face, so **the low band collapses exactly as a
starved joint's does**: measured on an 0603, 0.44 → 0.19 while the high band
goes 0.28 → 0.63. Every balled joint the renderer produced came back labelled
`insufficient` — the opposite fault, and an operator sent to add paste to a
joint that already carries too much.

Two measurements from the spec's own sentence for this test ("total wetted area
vs golden, plus high-angle flat fraction") were tried and neither survives
contact with a real land pattern:

| measurement | why it fails |
|---|---|
| total metal area over the crop | Separates only ×1.16. Bare copper is as specular as solder and the land is a fixed size, so a starved joint reads the same as a good one — which is what makes area *specific* to excess, but the signal is only the spill, diluted by the whole crop. |
| high-angle flat fraction | Does not discriminate, and on wide lands it actively vetoes. Bare copper is **flat**, so a *starved* joint scores a higher high-angle fraction (×2.6–3.1) than a balled one (×1.5–2.2). On a QFP-32 lead the cap replaces flat copper with curved solder and the high band *falls*, 0.29 → 0.20. |

What works is asking where the solder is, not how much light comes back: the
specular fraction of a 0.30 mm strip **just outside each land**, where CAD says
bare substrate. Same question the bridge test asks between two lands, asked on
the outer side. A starved joint leaves its own land bare and puts nothing
outside it. Measured over all eight packages, good side sampled across placement
scatter, vignetting 0.02–0.12 and LED gain 0.92–1.06:

| population | reading |
|---|---|
| good joints | ≤ 0.038 |
| starved joints | ≤ 0.038 |
| balled joints | 0.216 – 0.671 |

Threshold 0.12: 3.2× above the worst good reading, 1.8× below the weakest real
defect, biased toward the false-call side because a 500-component board carries
~1000 of these tests.

Two things this measurement is not. It is **not** "metal is present" — a pixel
passes only if its slope falls in one of the three rings, roughly 0–45°, so a
balled joint's near-vertical rim is genuinely invisible and the signal is the
shoulder inboard of it. And it is **not** an absolute threshold: golden's own
reading is subtracted, so an oversized etch or a copper pour near the footprint
cancels instead of becoming a permanent excess call on every instance of the
package.

The strip has to fit inside the crop's 0.35 mm margin. On every package in the
library the outermost land *is* the package extent, so a wider window is clipped
by the crop boundary — silently, with no error, measuring nothing. A first
attempt grew the fillet window 1.6× instead, which put its outer edge 0.60 mm
past the land on a QFP-32 and read ×1.01 there while every chip package read
cleanly. `test_excess.py` pins the relationship.

---

[← design notes index](README.md) · [repository README](../../README.md)

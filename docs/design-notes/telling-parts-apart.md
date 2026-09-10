# Telling one part from another

Wrong part is scored on the **area** of the segmented body, not on its bounding
box. The bbox is a maximum-extent statistic: one speck of noise surviving the
morphological open moves it (measured across good boards, bbox width wanders
9.1 % against area's 1.3 %), and the axis-aligned bbox of a skewed part inflates
39–46 % — enough to report every skewed component on the board as a wrong part.
Area is a bulk statistic and is invariant to rotation.

| population | body area vs golden |
|---|---|
| good boards | ×0.802 – ×1.013 |
| offset, skew, starved, balled | ×0.96 – ×1.01 |
| one-size-step substitutions | ×0.19 – ×4.81 |

Bounds are ×0.72 and ×1.12. They are asymmetric because the good side is: the
low tail is entirely one case, a SOD-123 at ×0.802 with LED output down 8 %.
`body_dif_min` and `body_spec_max` are absolute levels learned at teach, so
dimming the whole image erodes the blob edge and shrinks the measured body. That
was the same common-mode sensitivity the fillet test was rebuilt to avoid, and
roughly a 10 % dim was the limit.

**Fixed 10 September**, the way the note above said it should be: both
thresholds are now scaled by the crop's own bare-substrate level, which teach
already measures. Measured over a 54-component board, counting good parts read
as substitutions:

| LED gain | absolute thresholds | scaled to the crop |
|---|---|---|
| 1.00 | 0 | 0 |
| 0.92 | **2 false calls** | 0 |
| 0.84 | **24 false calls** | 0 |
| 0.75 | **24 false calls** | 0 |
| 1.30 | **24 false calls** | 0 |

The measured area ratio now sits between ×0.956 and ×1.025 across that whole
range, against bounds of ×0.72 and ×1.12 — so the bounds are no longer being
defended by the illumination staying still. The scaling is bounded either way:
a crop that is not merely dim, but wrong — a part over a copper pour, a crop
half off the image — has to be measured rather than accommodated.

The test sits with presence, not with joint quality, because it decides whether
the golden model may be applied at all: every test after it compares against
statistics for the package the centroid file *names*. It runs *after* the
tombstone test on purpose — both faults make the body measure small, and
standing on one land is far the commoner of the two, so the reading with a
specific explanation claims it first. What that costs is in Known limits.

The marking-classifier half of the spec's method for this test is not
implemented. It needs a character model per manufacturer and a legible marking,
which an 0402 does not have.

---

[← design notes index](README.md) · [repository README](../../README.md)

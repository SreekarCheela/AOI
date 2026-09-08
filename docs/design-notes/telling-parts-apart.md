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
`body_dif_min` is an absolute diffuse level learned at teach, so dimming the
whole image erodes the blob edge and shrinks the measured body. That is the same
common-mode sensitivity the fillet test was rebuilt to avoid, and this test does
not yet have the same defence — roughly a 10 % dim is the limit. The fix is to
scale both segmentation thresholds by the crop's own substrate level, which
teach already measures.

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

# Grading the fillet

The joint test is the only one §5 calls hard, and it was the only source of
false calls: 3 of 3, measured over 1080 good-component inspections. It now
scores a Mahalanobis distance of each pad's three band fractions against the
golden distribution for that package, rather than a z-score on the low band
alone.

The reason is that the bands are not independent. On good joints they correlate
at r = 0.5 to 0.85 across every package family:

| package | r(low,mid) | r(low,high) | r(mid,high) |
|---|---|---|---|
| 0402 | 0.78 | 0.66 | 0.67 |
| 0603 | 0.80 | 0.75 | 0.71 |
| 0805 | 0.76 | 0.81 | 0.66 |
| SOD-123 | 0.70 | 0.64 | 0.62 |
| TANT-A | 0.84 | 0.63 | 0.51 |

That correlation is the signal, not a nuisance. Anything changing how much light
reaches a component — vignetting, a dim corner, LED ageing, a different board
finish — moves all three bands **together**, along the axis where the covariance
is large, so it costs almost no distance. A starved joint loses the steep face
while keeping the shallow one, moving **across** that axis where the covariance
is small, so it costs a lot. A univariate test cannot separate those two, which
is why illumination non-uniformity is the largest single source of false calls
in a rule layer.

Measured on defect-free boards, teaching under nominal light and then inspecting
under drift:

| inspection condition | z-score on low band | Mahalanobis |
|---|---|---|
| nominal, as taught | 0.37 % | **0.00 %** |
| vignette 0.10 | 0.37 % | **0.00 %** |
| vignette 0.14 | 0.37 % | **0.00 %** |
| LED output −4 % | 0.37 % | **0.00 %** |
| LED output −8 % | 0.74 % | **0.00 %** |
| vignette 0.12, LED −5 % | 0.56 % | **0.00 %** |

Separation on the good/bad axis: over 2160 good pad-tests the distance ran
median 1.56, p99 3.50, max 4.63; a starved joint scores 36.8 at worst. The
threshold sits at 6.0, above the observed good maximum with ~23 % margin and
still 6× below the nearest real defect.

Two things had to be right for this to work at all. The covariance is estimated
from 5–20 golden boards, so it is **regularised** — bands correlated at 0.8
leave one near-collapsed axis whose small eigenvalue a plain inverse turns into
an enormous distance, and unregularised the test that is meant to cut false
calls generates them. And the comparison is **one-sided**: Mahalanobis distance
is symmetric, so a joint carrying more solder than golden sits just as far from
the mean as a starved one, and without a direction guard a heavy fillet gets
reported as insufficient solder. Grading excess is a separate test:
[grading-excess.md](grading-excess.md).

---

[← design notes index](README.md) · [repository README](../../README.md)

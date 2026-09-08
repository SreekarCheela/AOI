# Finding the body

Presence used to be a correlation score. NCC answers one question -- does this
look like a good part -- and collapses every way of not being one into a single
low number. It cannot say *where* the body went, and two label bugs came from
exactly that:

| | NCC of golden | |
|---|---|---|
| missing | down to 0.31, up to **0.572** | overlapping ranges, so no |
| displaced 500 µm | down to **0.505**, up to 0.97 | threshold separates them |
| tombstone, SOD-123 | **0.81** | above anything a gate can use |
| tombstone, TANT-A | **0.77** | without swallowing good parts |

So gross displacement read as `missing`, and tombstones on tall polarised parts
read as `offset` -- or as `reversed`, because a body standing on one land flips
the 180° template margin.

The fix is to measure the body rather than correlate it. It is the one large
region that is dark in **every** angled ring, so neither copper nor solder, yet
clearly brighter than bare substrate in the diffuse channel:

| | specular max | diffuse |
|---|---|---|
| bare substrate | 0.05 | 0.20 |
| **component body** | **0.082–0.086** | **0.26** |
| polarity marking | 0.114–0.118 | 0.78 |
| copper / solder | 0.62–0.72 | 0.42 / 0.60 |

Both thresholds are learned per package at teach. The specular one is a multiple
of the measured body level rather than a midpoint against bright metal, because
on a SOD-123 or TANT-A the body nearly covers the land pattern and there is no
bare copper left in the crop to anchor against.

Segment that, take the largest component, and the verdict falls out of its size
and shape:

```
no blob                  nothing was placed
spans its own width      standing on one land
spans its own length     lying flat; offset and skew will name it
```

**Normalise the extent by the body's own width, not its golden length.** A part
standing on one land presents its width along the pad axis whatever package it
is, so every tombstone measures 1.00 and every part lying flat measures its
package aspect ratio, 1.5 to 2.2 here. Against golden *length* the numbers
collide: a tombstone reads 0.22–0.61 and a part displaced far enough to be
clipped by the crop edge reads 0.758, short for an entirely unrelated reason.

Measured blob against the datasheet, so the segmentation is provably finding the
real part:

| package | body_l | measured | body_w | measured |
|---|---|---|---|---|
| 0402 | 1.00 | 1.05 | 0.50 | 0.55 |
| 0603 | 1.60 | 1.65 | 0.80 | 0.83 |
| 0805 | 2.00 | 2.06 | 1.25 | 1.26 |
| SOD-123 | 2.70 | 2.74 | 1.60 | 1.61 |
| TANT-A | 3.20 | 3.23 | 1.60 | 1.61 |

Correlation keeps one job: a second opinion before a part is called missing.
Segmentation is the sharper instrument and also the one that can fail wholesale
if a real board's body and substrate sit closer in tone than the golden boards
suggested -- and on its own, a segmentation that finds nothing would flag every
component on the board. If NCC still sees a body, that verdict is not allowed.

---

[← design notes index](README.md) · [repository README](../../README.md)

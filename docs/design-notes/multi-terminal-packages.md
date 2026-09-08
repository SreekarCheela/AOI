# Packages with more than two terminals

Land geometry used to be a pair: two centres at ±`pad_pitch/2`, a bands array of
shape `(2, 3)`, `for pad in range(2)` in the verdict. That put every IC,
connector and QFP out of scope, which meant the pipeline could not run on a real
assembly at all — passives are most of a board by count and none of what makes
it work.

Lands are now generated from row descriptors, and everything downstream indexes
that list. A chip is the degenerate case of one land a side:

| package | rows | per row | terminals | bridge gaps |
|---|---|---|---|---|
| 0402 … TANT-A | left, right | 1 | 2 | 1 |
| SOT-23-6 | left, right | 3 | 6 | 4 |
| SOIC-8 | left, right | 4 | 8 | 6 |
| QFP-32 | all four | 8 | 32 | 28 |

Three things needed a decision rather than just widening.

**The feature vector cannot have per-land columns.** One fixed width has to
serve a 2-pad chip and a 32-lead QFP, so the six `band_*_0` / `band_*_1` columns
became worst-and-mean per channel. The worst land is what decides the component
— one bad joint fails the part — and the mean says whether the whole part is off
or just one lead.

**A bridge is between neighbours, not across the body.** On a chip those are the
same thing. On a lead row they are not, and the lead-to-lead gaps are the ones
that actually bridge: a QFP-32 leaves ~0.4 mm between leads against 6.2 mm
across the body. `gap_specular` now scores every consecutive pair and takes the
worst.

**The tombstone test has to be gated on two terminals.** Not for tidiness: a QFP
body is square, so its extent along the pad axis equals its width and the
standing-body test reads ~1.0. Ungated, every quad package on the board is
flagged as a tombstone. Physically the gate is exact — a part needs exactly two
terminals to stand up on one of them.

Verified against the two-terminal path by hashing 50 rendered boards across
every defect type before and after: **bit-for-bit identical**.

---

[← design notes index](README.md) · [repository README](../../README.md)

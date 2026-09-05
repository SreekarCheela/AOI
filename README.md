# AOI Station One — prototype

Software prototype for the in-house post-reflow AOI cell specified in
[docs/aoi-station-one.html](docs/aoi-station-one.html).

It runs end to end **today, with no hardware**, against a synthetic board
renderer that implements the same reflection physics the machine relies on.
Train on the workstation, export, run on the Pi — the pipeline does not change.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python cli.py demo
```

## What it does

```
centroid + BOM ──▶ ROI generation ──▶ fiducial align ──▶ per-component measure
   (CAD prior)        (computed,          (registers        (8 rule tests,
                     never searched)       the board)        native 12 µm/px)
                                                                   │
                                            96×96×4 aligned patch ──┴──▶ anomaly + classifier
                                                                              │
                                                                     fusion tree ──▶ verdict
```

Nothing searches the image for components. Every region comes from the CAD
prior, which is what keeps the models small enough to be trainable and fast
enough to run on a Pi.

## Commands

| | |
|---|---|
| `cli.py demo` | teach on golden boards, inject defects, inspect, write a defect map |
| `cli.py dataset` | build training crops and feature vectors |
| `cli.py train` | anomaly autoencoder + classifier + fusion tree |
| `cli.py export` | write ONNX for deployment |
| `cli.py bench` | torch vs onnx parity and throughput |
| `cli.py demo --save` | inspect and record to the local station log |
| `cli.py review` | operator review screen for the last board |
| `cli.py yield` | per-package false call rate and review time |

`demo` takes `--backend none|torch|onnx`. The rule layer works with `none`, so
the pipeline is useful before anything is trained.

## The physics it implements

A specular surface tilted `α` from horizontal reflects into a vertical camera
only when light arrives at elevation `θ = 90 − 2α`. Each ring therefore reports
on one slope band:

| ring | elevation | slope seen | what that is on a joint |
|---|---|---|---|
| high | 72° | 9° | bare pad, board, component top |
| mid | 45° | 22.5° | mid-slope of the fillet sweep |
| low | 20° | 35° | steep face where solder climbs the termination |
| diffuse | dome | all | flat and shadowless, for body markings |

A healthy fillet sweeps high → mid → low from the pad toe up to the
termination. A starved joint never develops the steep face, so **the low band
collapses to zero** — measured, not eyeballed:

```
package   defect          low-angle band fraction
0402      ok              0.429
0402      insufficient    0.000
```

`synth.py` models fillet **peak slope** (52° healthy, 22° starved) rather than
fillet height. Height alone is wrong: it makes a tall package render an 80°
face that overshoots every band and inverts the test.

## Results, synthetic ground truth

| quantity | result |
|---|---|
| placement offset | 192–253 µm measured for 250 µm injected, all packages |
| skew | 0.6° mean absolute error (3° search grid) |
| board registration | 0.42 mm / 0.60° misregistration recovered to ~6 µm |
| anomaly separation | ×32 mean across package families |
| classifier | 100 % validation accuracy and recall |
| fusion tree | 100 % recall, 0 false calls of 129 good |
| end to end | 12/12 defects caught, 0 missed, 0 false calls, all types correctly labelled |
| torch ↔ onnx parity | max delta 1.2e-07, 100 % argmax agreement |
| throughput | 0.60 ms/component on ONNX → 0.37 s model time per 500-component board |

These are synthetic numbers. They prove the logic and the wiring are right; they
say nothing about false call rate on real boards, which is what §8 of the spec
is actually about.

## The operator station

The spec calls the review screen the product, and it is blunt about why: rules
alone plateau at 3–8 % false calls, and operator labels are what take it under
1 %. None of that can start until the labels are being stored, so the record
matters more than the screen on top of it.

```
inspect ──▶ SQLite ──▶ review screen ──▶ adjudication ──▶ labelled crop
                                                              │
                                          false-call filter ◀──┘  (§6C)
```

`aoi/store.py` is the local record: one SQLite file, no network in the loop, as
§5 requires. Boards, per-component verdicts, the feature vector, and a crop for
everything that failed or could not be tested — not for all 500 good components,
which would be storing the same picture of a correct 0402 several hundred times
a board. `export_labels()` hands back `(patches, labels, packages)` with 1 = real
defect and 0 = false call: naturally balanced, because it only ever contains
components the rule layer already flagged.

`aoi/review.py` is the screen. Native desktop, Tk from the standard library, so
the Pi image stays small. The design follows from the §10 exit criterion — an
operator clears a board's fails in under sixty seconds — which makes it a
keyboard target, not a mouse one:

- the flagged part beside a known-good one of the same package, because deciding
  from the crop alone is guessing
- false colour, low→red mid→green high→blue, the view where a starved joint is
  obvious: it has lost its red
- the measurement that triggered the call, in words
- **D** real defect, **F** false call, **S** skip. Skip means undecided, never
  silently good.

A coverage gap is reported apart from defects and never enters the queue: a
component with no inspection program is a gap in the program, not a fault on the
board, and mixing them is how one unprogrammed package buries the real fails.

Measured on a 54-component board with 12 fails: **12 labelled crops exported,
40 s of operator time**, against the 60 s target. Per package:

```
package      seen  flagged  judged  false  real     FCR   review
0402           18        4       4      1     3   5.56%     3.1s
0603           13        4       4      1     3   7.69%     3.1s
0805            9        2       2      0     2   0.00%     3.6s
ALL            54       12      12      2    10   3.70%
```

Escape rate is deliberately absent. It can only be measured against a
periodically hand-inspected sample, which is an input the station does not have,
and inventing a number for it would be worse than leaving the column empty.

## Packages with more than two terminals

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

## Finding the body

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

## Grading the fillet

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
reported as insufficient solder. Grading excess is a separate test that does not
exist yet.

## Choosing the anomaly model

Three approaches were built and measured. The failures are worth recording,
because two of them look like the obvious thing to do:

| approach | separation | why |
|---|---|---|
| pixel autoencoder | ×1.0 | reconstruction error is dominated by background, which is identical either way — it collapses to predicting the mean image |
| student-teacher, random teacher | ×1.4 | a random projection is too easy a function; the student fits it everywhere, defects included. Real EfficientAD uses a *pretrained* teacher whose function is hard enough that matching it off-distribution fails |
| **per-package subspace (PCA residual)** | **×32** | shipped |

Scoring globally instead of per package family costs an order of magnitude on
its own — ×1.1 versus ×4–6 even for a crude distance-to-mean. A 0402 and a
tantalum share no appearance, so one model asked to call both normal ends up
calling almost anything normal.

The shipped model is also the most deployable: a mean vector and a 16 × 2304
matrix, ~150 kB per package, exporting to ONNX as plain matmuls that run on any
accelerator. Training is one SVD, not an epoch loop.

## Non-obvious things that had to be right

Each of these silently broke the pipeline before it was fixed. They are the
reason to develop on synthetic data first.

- **The match template must be the body, not the footprint.** Lands are fixed to
  the board and never move, so including them anchors the correlation peak at
  zero displacement — a 250 µm offset measured as 8 µm.
- **Skew needs a masked correlation.** Rotating the region drags the lands round
  with it. Rotating an unmasked template rotates the lands inside the template.
  Either way the score argues for 0°. Mask to the body and the angle recovers
  exactly (the peak is the negative of the part's rotation).
- **Rotation search must extend past the tolerance it tests.** A sweep stopping
  at ±12° reports a 20° skew as exactly 12.0° and never trips a 12° limit.
- **Body and solder need genuinely different albedo.** Set them close and the
  diffuse channel becomes a near-uniform blob with no structure to match.
- **Rules are tested in precedence order, not by severity.** A part that is
  absent, bridged or displaced makes its fillet windows meaningless, and the
  fillet test fires hard on all of them — so scoring by severity labels every
  defect "insufficient solder". Gross faults gate joint tests.
- **Report alignment at the board centre.** The raw affine translation is
  measured from the pixel origin and is dominated by the rotation lever arm.
- **Draw the defect map on the registered stack.** Boxing CAD positions over the
  raw capture puts every box off its component by exactly the misregistration
  the pipeline just corrected — which reads to an operator as a broken machine.
- **Anomaly models are per package family, never global.** See above; this one
  costs an order of magnitude.
- **A package library cannot be validated by synthetic data.** The renderer and
  the inspector read the same numbers, so a package that disagrees with its
  datasheet is perfectly self-consistent and every test passes. `SOIC-8` and
  `SOT-23` both shipped here with body length and width swapped, which put the
  leads on the short sides — geometrically valid, fillet run still positive, and
  describing a part that does not exist. It cost more time than any other single
  error. Two guards now: nominal dimensions pinned against the standards, and
  the rule that a dual-row package spreads its leads along its *long* axis.
- **Sanity-check package geometry, because a bad one disables a test silently.**
  If the body reaches past its own land there is nothing for solder to climb:
  the fillet window degenerates, every joint reads permanently starved, and the
  fillet test is dead for that package. No exception, nothing that looks wrong
  in the library — it surfaces only as a false-call rate. `Package.validate()`
  now refuses it.
- **A layout margin cannot be one number once parts differ in size.** A 0402
  needs 1.1 mm of clearance and a QFP-32 needs 4.9 mm. Two failure modes, both
  of which present as a board-wide false-call storm rather than an error: a
  component whose metal reaches into a fiducial's ±2.5 mm search window can win
  the alignment correlation and hand back a bogus transform that displaces every
  crop on the board; and a component whose crop runs off the image gets
  `BORDER_REPLICATE` fill, so the lands landing in that strip read as bare. Both
  were found as a 64 % false-call rate on the first mixed board, and neither was
  a pipeline bug.
- **An unknown must not be representable as a bad reading.** `measure` returned
  early for a package teach never saw, leaving `ncc` at its 0.0 default — and
  0.0 is a real measurement, the one a totally wrong part scores. Every
  threshold downstream read "not measured" as "grossly defective", so a single
  unprogrammed package flagged every instance of itself. On a real board with
  fifteen package families against a library of five, that is a false call on a
  large fraction of the board. Unmeasurable fields are NaN now, and the verdict
  has a third state. The temptation is to suppress the noise by passing those
  parts instead; that trades a false-call problem for an escape problem, which
  is strictly worse.
- **Measure the thing you are deciding about.** Presence was a correlation
  score for three rounds of threshold tuning, and each round moved a number that
  could not answer the question being asked of it: NCC says how well a crop
  looks like a good part, never where the body went, so "absent" and "somewhere
  else" are the same reading. Two label bugs survived every threshold because
  the populations genuinely overlap. Segmenting the body and measuring its size
  and shape settled both immediately and deleted the tuned gate. When a
  threshold keeps needing care, check that the measurement under it is the right
  one before tuning it again.
- **Place a threshold at the midpoint of the window it has to sit in, and
  measure both edges.** The presence gate ran a fixed 0.85 of golden, which gave
  12 % headroom against false calls and 25 % against missed detections: margin
  nobody needed, paid for with margin that was running out. The window is
  0.638 (the best-scoring defect whose *label* depends on the gate firing) to
  0.961 (the worst a good part scored over fresh boards), so the midpoint is
  0.78 and both sides get ~18 %. Measure the upper edge on boards the teach set
  never saw: teach understates the spread by up to 2.2x, and the worst good 0402
  sits 7.4 teach-sigma below the golden mean.
- **Only count constraints that change a verdict.** SOD-123 and TANT-A
  tombstones score 0.81 and 0.77 and look like a far tighter lower bound than
  the 0.638 above. They are not: they fall through to the same label whether the
  gate fires or not, so honouring them would have shrunk the window by half in
  exchange for no behaviour at all. Check what a threshold actually decides
  before letting it constrain you.
- **The presence threshold is a fraction of golden, never an absolute number.**
  Golden NCC is a property of the package, not the machine: an 0603 matches its
  own template at 0.92, a SOD-123 at 0.99. One global constant therefore lands
  inside one package's defect distribution while being unreachable for another.
  At the old `ncc < 0.62`, a tombstoned 0402 scored 0.599 when one land survived
  and 0.647 when the other did, so which pad happened to survive decided whether
  the fault was found — and the pad that cleared the gate came back "skew" off a
  spurious rotation peak. Gate on a fraction of what teach measured for that
  package instead.
- **Grade the fillet on all three bands jointly, never one at a time.** The
  bands correlate at r = 0.5–0.85 on good joints, and that is exactly what makes
  a joint test possible: common-mode brightness changes move all three together
  and must stay cheap, while a starved joint moves across that axis and must be
  expensive. A z-score on the low band alone cannot tell "this joint is starved"
  from "this corner of the board is dimmer", and it was the only source of false
  calls in the rule layer. Regularise the covariance — teach gives you 5–20
  samples for a 3×3 matrix and near-collinear bands, so a plain inverse
  manufactures the false calls you were trying to remove.
- **Failing the presence gate does not say which fault it is.** It only says the
  body is not on its footprint. A part that walked off its lands has a low NCC
  and a dead joint too, exactly like a tombstone. What separates them is that a
  tombstone's surviving joint is *fully formed* — the part really is soldered,
  at one end — while a displaced part has both joints compromised. Name the
  fault from the joints; let anything else fall through to the placement tests,
  or a placement fault gets reported as a reflow fault.
- **Backfill the anomaly column before fitting fusion.** `dataset` builds feature
  vectors before any model exists, so that column is zeros on disk. Fit on it
  as-is and the anomaly score contributes literally nothing — its reported
  importance was 0.000 until the backfill, then became the top feature at 0.374.

## Porting to the Pi

```bash
.venv/bin/python cli.py train        # workstation
.venv/bin/python cli.py export       # -> out/models/*.onnx
scp -r out/models pi@station:~/aoi/out/    # includes the .onnx.data sidecars
# on the Pi:
python cli.py demo --backend onnx
```

Weights over ~1 MB export as an external `.onnx.data` file beside the `.onnx`.
Both must be copied together or the session loads with empty tensors.

`models.Backend` is the only seam. `TorchBackend` and `OnnxBackend` implement
the same two methods, so `pipeline.py` never learns which one it is talking to.
A `HailoBackend` slots in the same way.

Pi constraints carried over from the spec: USB3 camera rather than GigE
(~1 GB per board), keep 96×96 crops rather than full frames (~5 MB vs 1 GB per
board), accelerator on the single PCIe lane with storage on USB3, and training
stays on the workstation.

## Known limits

- Two-terminal packages only. Multi-lead parts need per-lead ROI generation.
- An 0402 displaced 500 µm or more reads "bridge", not "offset". The body and
  its solder land squarely over the inter-pad gap, and the specular reading
  there is what the bridge test measures; the bridge test runs first by design
  because a bridge wrecks every other measurement. Only the smallest package
  does this. The part still fails the board; the operator is sent to look for a
  short that is not there.
- Skew on a tall polarised package reads "offset". Rotating a long body moves
  the correlation peak past the 150 µm placement limit and the offset test runs
  first. Body localisation does not help: a rotated body is still lying flat and
  still spans its own length.
- A package teach never saw is reported as **not inspected**, not as a defect
  and not as a pass. It cannot be judged: every test compares against a golden
  statistic for that package. The board report names the coverage gap once
  rather than flagging every instance, and the defect map crosses those
  components through in grey. A genuine defect on an unprogrammed package will
  not be caught, which is why it is never reported as good.
- The subspace anomaly model assumes patches stay well canonicalised. Real
  boards vary more than synthetic ones; if separation degrades, EfficientAD with
  a pretrained teacher is the upgrade path, not a bigger autoencoder.
- Synthetic data validates logic and wiring, not real-world false call rate.
  The 3–8 % starting FCR in the spec is a claim about production boards.

## Layout

```
aoi/config.py     optics + channel constants from the spec
aoi/packages.py   package library (body/land geometry)
aoi/board.py      Component/Board, centroid CSV import
aoi/roi.py        CAD-derived regions; patch and region extraction
aoi/synth.py      four-channel renderer + defect injection
aoi/rules.py      measurements + golden reference
aoi/models.py     TinyCNN, subspace anomaly, Torch/ONNX backends
aoi/fusion.py     precedence rules + gradient-boosted fusion
aoi/pipeline.py   align, inspect, defect map
cli.py            demo / dataset / train / export / bench
```

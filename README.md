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
- Tombstones on tall polarised packages (TANT-A, SOD-123) are mislabelled, not
  merely confused with "missing" as this used to claim. The body nearly covers
  the lands, leaving a fillet window ~0.2 mm wide, so the surviving joint reads
  only ~43 % of golden and never looks like a formed joint. Pad 1 additionally
  flips the polarity margin and comes back "reversed". Both still fail the
  board, so nothing escapes; the operator is sent to the wrong process fix.
  Separating them needs a measurement that tells a standing body from a
  displaced one, not another threshold. Chips — where tombstoning actually
  happens — discriminate correctly on both pads.
- NCC cannot separate "no body at all" from "body walked off its lands". The
  two populations overlap across packages: a missing part reaches 0.572 of
  golden (SOD-123) while a part displaced 500 um falls to 0.505 (0603), so no
  threshold sits between them and a gross displacement reads "missing" on chips.
  Separating them needs a measurement that finds the body somewhere else on the
  crop. Both still fail the board; only the label is wrong.
- A package the golden set never covered flags every instance of it as
  "missing", because `measure` returns before scoring NCC and the untaught
  branch reads the 0.0 default as an absent part. On a real board that is a
  false-call storm for any package the teach boards happened not to carry. The
  fix is a verdict state meaning "untestable", which the rule layer does not
  have yet.
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

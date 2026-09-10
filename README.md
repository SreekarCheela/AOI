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
| `cli.py shots --dir photos/` | turn real photographs into a channel stack — the physics gate |

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
| lens perspective | 48 µm error on a tall part off-axis, corrected to 0 µm |
| board registration | 0.42 mm / 0.60° misregistration recovered to ~6 µm |
| anomaly separation | ×27.8 mean across package families |
| classifier | 98.9 % validation accuracy, 100 % recall |
| fusion tree | 100 % recall, 0 false calls of 112 good (0.00 %), anomaly column cross-fitted |
| rule layer, false calls | 0 over 1000 component inspections — 6 boards × 4 lighting conditions, misregistered |
| excess solder | good and starved ≤ 0.038 outside the land, balled 0.216–0.671; threshold 0.12 |
| wrong part | good ×0.802–×1.013 of golden body area, substitutions ×0.19–×4.81; bounds ×0.72 / ×1.12 |
| end to end | 12/12 defects caught, 0 missed, 0 false calls, all types correctly labelled |
| torch ↔ onnx parity | max delta 6.0e-07, 100 % argmax agreement |
| throughput | 0.16 ms/component on ONNX → 0.07 s model time per 500-component board |

Anomaly separation and classifier accuracy both moved slightly when the
renderer began modelling lens perspective (×29.6 → ×27.8, 100 % → 98.9 %).
That is the task getting harder rather than the models getting worse: a
body's apparent position now depends on where it sits in the frame, which
is true of the real machine and was not true of the renderer before.

These are synthetic numbers. They prove the logic and the wiring are right; they
say nothing about false call rate on real boards, which is what §8 of the spec
is actually about.

## The physics gate

Everything in this README and in the design notes is validated against a
renderer built from the same equations the tests measure. That proves the logic
and the wiring; it says nothing about whether real solder behaves this way.
`aoi/capture.py` exists to answer that and has never been run on a real
photograph.

**[`docs/physics-gate-shoot.md`](docs/physics-gate-shoot.md) is the checklist**
— a camera on a tripod, one lamp, a scrap reflowed board with one hand-starved
joint, four frames, and `python cli.py shots --dir photos/`. A weekend, and the
last cheap chance to find out the optical design needs rethinking before the
optics are bought.

## Known limits

What the code gets wrong today is written down rather than left to be
discovered: [`docs/design-notes/known-limits.md`](docs/design-notes/known-limits.md).
Each entry is pinned by a test asserting the current behaviour, so it is a
recorded limitation rather than an unknown. In every case the board still fails
— the operator is sent to look at the wrong thing, never sent away.

The two that matter most: a package teach never saw is reported as **not
inspected**, never as good; and synthetic data validates logic and wiring, not
real-world false call rate.

## Design notes

Why each part of the pipeline works the way it does, and what was tried first.
Almost every one of these records a measurement that overturned an obvious
approach — including two places where the spec's own suggested method did not
survive contact with a real land pattern.

**[`docs/design-notes/`](docs/design-notes/README.md)** — indexed there, in
full:

| | |
|---|---|
| [Finding the body](docs/design-notes/finding-the-body.md) | why presence is segmented, not correlated |
| [Grading the fillet](docs/design-notes/grading-the-fillet.md) | why all three bands are scored jointly |
| [Grading excess](docs/design-notes/grading-excess.md) | why excess is measured outside the land |
| [Telling one part from another](docs/design-notes/telling-parts-apart.md) | why body area, never the bounding box |
| [Multi-terminal packages](docs/design-notes/multi-terminal-packages.md) | what changed to inspect more than two pads |
| [Choosing the anomaly model](docs/design-notes/choosing-the-anomaly-model.md) | three built, two of them separate by ×1.0 and ×1.4 |
| [The operator station](docs/design-notes/operator-station.md) | the review screen, and how review time is measured |
| [Porting to the Pi](docs/design-notes/porting-to-the-pi.md) | the one seam that keeps it a deployment, not a rewrite |
| [Frame registration](docs/design-notes/frame-registration.md) | aligning gate photographs that do not look alike |
| [Non-obvious things that had to be right](docs/design-notes/traps.md) | twenty-odd ways this broke silently |
| [Known limits](docs/design-notes/known-limits.md) | what it gets wrong today |

## Tests

```bash
.venv/bin/python -m pytest        # 320 passing, ~100 s
```

Every design note above has tests behind it, named for what they pin rather
than for the module they exercise: `test_fillet.py` holds the illumination-drift
measurements, `test_excess.py` pins the strip-inside-the-crop relationship,
`test_traps.py` and `test_known_limits.py` assert the behaviours the two notes
of the same name describe, and `test_silent_failures.py` covers five failures a
review found in paths the rest of the suite never entered — each verified to
fail against the unfixed code before being kept.

## Layout

```
docs/
  aoi-station-one.html   the design spec, Rev A — source of truth
  physics-gate-shoot.md  checklist for the real-photograph gate
  design-notes/          why the pipeline works the way it does

aoi/
  config.py     optics + channel constants from the spec
  packages.py   package library (body/land geometry)
  board.py      Component/Board, centroid CSV import
  roi.py        CAD-derived regions; patch and region extraction
  tiling.py     scan raster and which tile owns which component
  synth.py      four-channel renderer + defect injection
  capture.py    real photographs -> the same four-channel stack
  rules.py      measurements + golden reference
  models.py     TinyCNN, subspace anomaly, Torch/ONNX backends
  fusion.py     precedence rules + gradient-boosted fusion
  pipeline.py   align, inspect, defect map
  store.py      local SQLite station record
  review.py     operator review screen

cli.py          demo / dataset / train / export / bench / review / yield / shots
tests/          276 tests; see Tests above
examples/       output from `cli.py demo`, no hardware involved
out/            generated — models, datasets, defect maps (gitignored)
```

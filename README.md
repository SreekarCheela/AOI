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
| `cli.py calibrate <image>` | measure µm/px and lens distortion from a dot-grid target |
| `cli.py filter` | fit the false-call filter (§6C); `--demo` simulates production |
| `cli.py audit` | random sample of what the filter suppressed, for hand-checking |
| `inspect_photo.py` | missing and misplaced parts on ordinary photos — standalone, see below |

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

## Photo AOI: placement inspection from ordinary photos

`python -m aoi_photo` opens an offline desktop app (Tk, spec section 5: no web
server, no network, nothing leaves the machine). It checks placed boards from
photographs, separate from the station pipeline, which needs channel stacks,
fiducials and a centroid file. It reports **missing**, **offset**, **skew**,
**polarity**, **tombstone** and **wrong part**, with a box on every flagged part.
Solder quality needs the rings and is out of scope.

Nothing has to be drawn or named. Nine clicks from launch to a box on a defect:

1. **Set up board**: click **Upload golden photos…**. The part detector boxes
   every part as P1, P2, … on its own. Delete any box that is not a part, and
   for the few polarized parts pick *Diode/LED* or *IC/transistor*, which is
   what turns the polarity check on. Renaming (R12 instead of P12) and the
   board length in mm are optional: without a length, offsets are judged
   against the board's own noise and reported in pixels. Save. A checklist
   beside Save says what, if anything, is still missing.
   - One golden works, and every result says "single golden". For real
     thresholds use **3 goldens from 3 different good boards**. Shots of one
     board teach nothing about how solder and flux vary; the same shot twice
     is refused.
2. **Inspect boards**: pick the program, click **Upload test photos…** (one
   per board) and **Run**. Click a board to see its photo with the defect
   boxes; **Review calls →** opens it in Results.
3. **Results**: boxes on flagged parts, golden | test crops, and buttons (or
   keys) for real defect D, false call F, skip S, next N, undo U, and reshoot R
   (board calls only). **False-call stats** is the second page: false calls by
   package, which spec section 8 says decides whether an AOI gets used.
   **Advanced** menu: parts.csv import/export and Retrain detector.

**Whole-board difference.** Every test photo is also compared with golden 1
everywhere, not just inside part boxes (`aoi/photo/diff.py`). A difference
away from every taught part is reported as **UNEXPECTED DIFFERENCE** (DIFF1,
DIFF2, …, REVIEW): a part the detector never boxed that is now missing, or
something on the board that should not be there. On the fixture, a board
taught without a C7 box still gets a box on the missing C7. Differences on
bare tinned metal are ignored: two boards' solder finish differs, and no part
can go missing from a bare trace.

Photos: taken straight down. The board must fill at least 20% of the frame and
be at least 1500 px long, and the smallest part at least 40 px long. Use
diffuse light, and the same camera and distance for golden and test photos
(within ±15%). The app rejects photos that break these rules and says why.
HEIC opens directly on macOS (through the built-in `sips`); elsewhere it needs
`pip install pillow-heif`. JPEG needs nothing.

Headless, for scripts:

```bash
python -m aoi_photo teach --golden photos/golden/ --parts parts.csv --name board   # --board-width-mm W optional
python -m aoi_photo inspect --program board photos/test/        # exit 0 pass, 1 review, 2 fail
python -m aoi_photo stats
python -m aoi_photo eval --program board photos/planted/ --json score.json   # needs photos/planted/truth.json
```

`eval` scores a program on boards whose defects are known. The folder holds one
photo per board and a `truth.json` listing each planted defect
(`{"image": "board3", "refdes": "R7", "defect": "skew"}`; defects are `missing`,
`offset`, `skew`, `wrong_part`, `tombstone`, `reversed`, `marking_differs`).
A photo with no entry is a good board, so anything flagged on it is a false call.
It prints, for each defect type, how many were caught, caught under another
label, missed, or on a part the program never taught. It then lists every false
call and the board escapes. It writes nothing to the operator record. Its rates
mean something only on boards that were not used to set a threshold.

`parts.csv` has these columns: `refdes,x,y,w,h,type,package,polarized,marked_end`,
in golden-1 pixels. Reports go to `out/photo_runs/<program>/<time>/`: an
annotated JPEG, a CSV, and a one-file HTML with no external assets. Operator
labels go to `out/photo_aoi.db`.

**Measured on the only real data so far.** The 2026-09-25 iPhone shoot of
PT-PCB-004-03 has one golden and three test boards; the fixture is in
`photos/fixtures/2026-09-25`, and tests skip when it is absent.

| Defect | Truth | Found |
|---|---|---|
| Missing | 14 | 12 in and out of sample, 0 false calls |
| Reversed diode (test1 D4) | 1 | POLARITY REVIEW |
| Offset transistor (test1 T1, 0.42 mm) | 1 | OFFSET |
| Wrong value by marking (R19) | 2 boards | not found |

- **Missing parts:** the two misses are bare lands of dull solder that read as
  a body on every feature tried.
- **Polarity:** REVERSED needs 3 or more goldens.
- **Marking:** a one-digit marking change is below what marking correlation
  sees at ~45 px/mm.

Scores are shown as `score`, not as probabilities, until a check has 30
operator labels (10 of each kind); after that they are Platt-calibrated and
shown as `P=`.

**Unvalidated:** tombstone and wrong-part checks, until a real example exists.
Neither colour class nor body size from a single golden was stable enough to
use, so wrong-part needs 3 or more goldens.

The same fixture is in single-golden mode (`--single-golden`, CLI and tests
only), with fixed thresholds and every result stamped. Its missing-part model
was fitted on these same photos.

**Part detector (Propose boxes).** It's a small CNN, written in plain PyTorch,
that marks every part body. Colour thresholds couldn't do this: tinned solder,
leads and through-hole pads are the same colour as ceramic capacitors, so they
were proposed as parts. Cutouts are excluded because the board mask now treats
anything connected to the frame edge as background.

The detector trains offline on:

- every taught program;
- the labelled shoots of PT-PCB-004-03, with the fixture golden's body boxes
  carried onto each photo by registration;
- the CC0 "PCB Component Detection" set, at low weight.

No FPIC: its licence is non-commercial, no-derivatives. Retrain it with
`python -m aoi_photo train-detector`, or with **Retrain detector** in the app,
after teaching new boards. That takes about 45–85 minutes on an Apple-silicon
Mac. The weights go to `out/models/part_detector.pt` (gitignored); without them,
Propose boxes falls back to colour and says so.

Measured on PT-PCB-004-03 so far. Trained without any iPhone photo and scored
on the four iPhone shots:

| | tight boxes (IoU ≥ 0.5) | false boxes |
|---|---|---|
| golden | 27 / 37 | 1 |
| all four shots | 74% | 7 (precision 93%) |
| colour method, golden | 18 / 37 | 39 |

The shipped model, trained on those shots too, boxes all 37 golden parts with no
false box, but that is in-sample. A board type it has never seen will score
lower until it is taught.

`inspect_photo.py` is the earlier one-script prototype. Its registration now
lives in `aoi/photo/register.py`.

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
.venv/bin/python -m pytest        # 414 passing, ~120 s
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
  physics-gate-shoot.md  the one experiment still outstanding
  design-notes/          why the pipeline works the way it does

aoi/                     what the machine is told
  config.py       optics, channels, and every tuned threshold
  packages.py     package library: body, land and height geometry
  board.py        Component/Board, and the placement-file entry point
  centroid.py     reading the placement files real tools write
  calibrate.py    sampling pitch and lens distortion, from a glass target
  tiling.py       the scan raster, and which position sees which part

                         what it sees
  synth.py        four-channel renderer + defect injection
  capture.py      real photographs -> the same four-channel stack
  roi.py          CAD-derived regions; patch and region extraction

                         what it decides
  rules.py        measurements — numbers, never verdicts
  models.py       classifier, anomaly subspace, Torch/ONNX backends
  fusion.py       precedence rules + gradient-boosted fusion
  pipeline.py     align, inspect, defect map
  falsecall.py    the false-call filter, and the gate that bounds it

                         where it lands
  store.py        the local station record, one SQLite file
  review.py       the operator screen

cli.py            demo / dataset / train / export / bench / review / yield
                  / shots / calibrate / filter / audit
aoi/photo/        photo mode: io, register, features, judge, teach, inspect, record, report,
                  detect / detect_train (the part-box detector)
aoi_photo/        the offline desktop app (Tk) and its CLI
inspect_photo.py  earlier photo prototype, kept for its synthetic tests
tests/            522 tests; see Tests above
examples/         output from `cli.py demo`, no hardware involved
out/              generated — models, datasets, defect maps (gitignored)
```

`aoi/__init__.py` carries the same map, so it is there when you are reading the
code rather than the README.

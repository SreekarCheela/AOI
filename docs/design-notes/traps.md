# Non-obvious things that had to be right

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
- **Anomaly models are per package family, never global.** See
  [choosing-the-anomaly-model.md](choosing-the-anomaly-model.md); this one costs
  an order of magnitude.
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

---

[← design notes index](README.md) · [repository README](../../README.md)

# Known limits

- A substituted part whose body length matches the *expected* part's width reads
  "tombstone", not "wrong part". An 0603 body is 1.60 mm long and a SOD-123 is
  1.60 mm wide, so on a SOD-123 footprint it spans exactly 1.00× the expected
  width along the pad axis — which is the definition of standing on one land. On
  body geometry alone the two are not distinguishable. The part is caught and
  the board fails; the operator is sent to look at the joint rather than at the
  stores. Fixing it means making the tombstone test require what a tombstone
  implies — one land wetted and the other bare, which the spill measurement now
  gives per land — rather than body shape alone.
- A SOD-123 fitted on a TANT-A footprint reads "insufficient". The two share a
  width and differ only in length, so the segmented area lands at ×0.86 — inside
  the good population, whose low tail reaches ×0.80 under an 8 % dim. No
  threshold separates them. It is caught anyway, by the joint: a shorter body
  leaves a longer run from land to body edge and the stretched fillet reads
  starved.
- Excess solder is a threshold on defect *severity* as much as on the
  measurement. A mild bulge that never spills past its land is not caught by an
  area test and never will be.
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

- **The lens hides part of a tall part's joint, and no software recovers it.**
  A component standing above the board is viewed off the optical axis, so its
  own body image overhangs the joint at the near end. On a 72 mm board a TANT-A
  loses 142 µm of a 200 µm fillet run at the extremes — 71% of the joint simply
  not visible to the camera. This is geometry: it is why the spec chose a 75 mm
  lens over a 35 mm one (194 µm of displacement down to 91 µm), and the residual
  cannot be corrected away because the pixels do not exist.

  Measured on a good TANT-A over three boards, the joint distance ran 1.6 / 3.0 /
  2.4 with the lens modelled flat and 4.8 / 5.3 / 6.7 with it modelled properly,
  against a fail threshold of 6.0. What it costs in practice is teach data:
  worst-case distance over 30 boards fell from 10.5 at five golden boards to 5.6
  at ten and 5.3 at twenty. `cli.py demo` now names the affected packages at
  teach time rather than leaving it to be discovered as a false-call rate.

  If it ever needs to be better than that, the levers are optical and not
  algorithmic — a longer lens, a smaller board, or accepting that the tallest
  packages are graded only near the axis.

---

[← design notes index](README.md) · [repository README](../../README.md)

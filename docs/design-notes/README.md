# Design notes

Why each part of the pipeline works the way it does, and what was tried first.

These were the back half of the repository README until it reached 690 lines.
They are kept because almost every one of them records a measurement that
overturned an obvious approach — including two places where the design spec's
own suggested method did not survive contact with a real land pattern. Deleting
them would mean rediscovering the same things.

The spec itself is [`../aoi-station-one.html`](../aoi-station-one.html); it is
the source of truth for every design decision, and these notes never override
it, only record what implementing it turned up.

## The inspection tests

| note | what it settles |
|---|---|
| [Finding the body](finding-the-body.md) | Why presence is segmented and measured rather than correlated. Correlation cannot say *where* the body went, and two label bugs came from exactly that. |
| [Grading the fillet](grading-the-fillet.md) | Why joint quality is scored on all three slope bands jointly. The only source of false calls in the rule layer. |
| [Grading excess](grading-excess.md) | Why excess solder is measured *outside* the land. Both measurements the spec suggested were tried and neither works. |
| [Telling one part from another](telling-parts-apart.md) | Why wrong-part is scored on body area, never on the bounding box. |
| [Multi-terminal packages](multi-terminal-packages.md) | What had to change to inspect anything with more than two pads — and why the tombstone test has to be gated. |

## The learned layer

| note | what it settles |
|---|---|
| [Choosing the anomaly model](choosing-the-anomaly-model.md) | Three approaches built and measured. Two of them look like the obvious thing to do and separate by ×1.0 and ×1.4. |

## The station

| note | what it settles |
|---|---|
| [The operator station](operator-station.md) | The review screen and the local record. Why review time is measured by the rate sustained rather than by wall clock. |
| [Porting to the Pi](porting-to-the-pi.md) | The one seam that makes the edge deployment a deployment step rather than a rewrite. |

## Real-world readiness

| note | what it settles |
|---|---|
| [Registering frames that do not look alike](frame-registration.md) | Why the physics-gate frames are aligned on what they share rather than on the highlights. It fails silently, which is the dangerous part. |

## Read these two before trusting a number

| note | what it settles |
|---|---|
| [Non-obvious things that had to be right](traps.md) | Twenty-odd ways the pipeline broke silently before it was fixed. The reason to develop against synthetic data first. |
| [Known limits](known-limits.md) | What the code gets wrong today, each one pinned by a test asserting the current behaviour. |

---

[repository README](../../README.md) · [design spec](../aoi-station-one.html) · [physics gate checklist](../physics-gate-shoot.md)

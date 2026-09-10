# Documentation

Three things live here, and they answer different questions.

| | |
|---|---|
| [**aoi-station-one.html**](aoi-station-one.html) | **The design specification, Rev A.** Eleven sections: the optics, the illumination, the mechanics, the software architecture, a costed bill of materials, a six-phase build plan with pass/fail exit criteria, and an honest list of what the machine will never do. It is the source of truth for every design decision, and nothing in this repository overrides it. Open it in a browser. |
| [**physics-gate-shoot.md**](physics-gate-shoot.md) | **The one experiment still outstanding.** A camera on a tripod — or a phone — one lamp, a scrap reflowed board with one deliberately starved joint, four frames. Written to be used standing over a board rather than at a desk. Everything else here is validated against a renderer built from the same equations the tests measure; this is what checks that real solder behaves that way. |
| [**design-notes/**](design-notes/README.md) | **Why the pipeline works the way it does**, and what was tried first. Eleven pages, indexed. Almost every one records a measurement that overturned an obvious approach — including two places where the specification's own suggested method did not survive contact with a real land pattern. |

## Reading order

Coming to this cold, in this order:

1. The specification's §1 and §2 — the reflection physics the whole machine rests on, and why the optics follow from it. Nothing else makes sense first.
2. The repository [README](../README.md) — what runs today, and the measured results.
3. [`design-notes/traps.md`](design-notes/traps.md) and [`design-notes/known-limits.md`](design-notes/known-limits.md) — the two that stop you trusting a number you should not. Read them before believing anything in the results table.
4. [`physics-gate-shoot.md`](physics-gate-shoot.md) — because until it has been run, everything above proves the logic and the wiring, and nothing about the physics.

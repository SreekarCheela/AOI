# Registering frames that do not look alike

The prototype shoot is a camera on a tripod and one lamp moved between frames,
so the frames need registering to each other. That is harder than it sounds and
it fails silently, which is the dangerous part.

Phase correlation assumes the two images look alike. These do not, by design:
each angled frame lights a *different slope band*, so a fillet blazing in the
low frame is dark in the high one. The specular highlights are the brightest
thing in the picture and therefore dominate any gradient — and they are exactly
the content that moves between frames rather than staying put. Correlating on
raw image gradient locks onto a wrong peak:

| frame pair | injected | recovered | error |
|---|---|---|---|
| low → mid | +5.00 | +9.65 | **+4.7 px** |
| low → high | +5.00 | −5.67 | **−10.7 px** |
| low → diffuse | +5.00 | +5.43 | +0.4 px |

The correlation *response* does not help: the wrong answers above scored 0.20,
higher than some correct ones at 0.11. And the symptom is only a smeared
false-colour image, whose natural reading is "the optical design does not work"
rather than "the frames do not line up" — an expensive conclusion to draw
wrongly.

What every frame does share is the **diffuse** part of the scene: soldermask,
silkscreen, component bodies, lit at every lamp elevation and only faintly. Two
steps bring them out, both load-bearing:

- **Normalise each frame by its own 99.5th percentile.** The lamp moves, so
  frames differ three- to four-fold in overall brightness, and a log is not
  scale invariant. Without this the same scene compresses to a different shape
  in each frame and the error returns (measured 4.6 px).
- **Compress with a log, then take the gradient.** This squashes the specular
  highlights into a narrow band at the top and stretches the faint diffuse
  detail underneath, so the gradient is driven by structure present in every
  frame.

Channel spread after registration drops from **16.4 px to 0.04 px** in x. A
shift too large to be a tripod nudge (>64 px) is refused rather than applied —
warping by a meaningless correlation peak turns a nearly-right frame into an
unusable one — and `cli.py shots` prints what registration did to every frame,
because this is the failure that does not announce itself.

None of this survives into the machine: a fixed camera strobing rings gives
inherently registered frames. It exists for the handheld gate.

---

[← design notes index](README.md) · [repository README](../../README.md)

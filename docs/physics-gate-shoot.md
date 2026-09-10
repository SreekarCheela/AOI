# The physics gate — shoot checklist

**What this answers:** does real solder produce the slope-banded response the
whole machine is designed around? Every result in this repo so far comes from a
renderer built on the same equations the tests measure, so it proves the logic
and the wiring, not the physics. `aoi/capture.py` was written for this and has
never seen a real photograph.

**Why now:** it costs a weekend and a scrap board, and it is the last cheap
chance to find out the optical design needs rethinking before roughly $8.8k of
optics is ordered against it.

**What good looks like:** in `out/real_false_colour.png`, a properly formed
fillet sweeps **blue → green → red** from the pad toe out to the component
termination. A starved joint never develops the steep face and simply has no
red.

---

## 1 · Gather

- [ ] Camera with full manual control — DSLR, mirrorless, or a phone with a
      pro/manual mode — **a phone works, but see §2b: it needs more
      turning off than a camera does**
- [ ] Tripod, or any rigid mount. The camera must not move at all between frames
- [ ] Remote release, or the camera's 2-second timer
- [ ] One small lamp — desk LED, bike light, phone torch. Smaller is better; a
      broad softbox smears the slope bands together
- [ ] A white wall, ceiling or card to bounce off for the diffuse frame
- [ ] A scrap reflowed board with **at least one visibly starved joint next to
      good ones**. The comparison is the experiment; a board of uniformly good
      joints proves much less
- [ ] Tape measure or ruler
- [ ] A ruler or caliper laid in the frame for one extra shot (see §6)

> **Making a starved joint:** reflow one board normally, then reheat one joint
> and wick most of the solder off with braid. You want an obviously thin,
> concave joint beside untouched ones on the same board under the same light.

---

## 2 · Lock the camera down

These are the settings that ruin the experiment silently. The method compares
brightness **between** frames, so anything the camera changes automatically
between shots destroys the measurement, and the resulting picture still looks
like a photograph.

- [ ] **Manual exposure.** Fix shutter, aperture and ISO. Write the values down.
      **Do not change them between frames.** ← the single most important line
      on this page
- [ ] **Manual focus**, set on the joints, then locked. Autofocus will hunt as
      the light changes
- [ ] **Manual white balance**, fixed to any preset. Auto WB re-weights the
      colour channels per frame, which changes the greyscale conversion
- [ ] **RAW** if the camera offers it (already linear — run with `--linear`).
      Otherwise JPEG at maximum quality
- [ ] HDR **off**, flash **off**, any scene/auto mode **off**, noise reduction
      and "smart" processing as low as it goes
- [ ] Electronic shutter or mirror lock-up if available, to remove vibration
- [ ] Camera looking **straight down** at the board, sensor plane parallel to
      it. The reflection model assumes a vertical camera; a tilted one shifts
      every slope band

**Exposure level:** the angled frames are *supposed* to look almost black with
small bright highlights — that is the signal. Expose so the brightest solder
highlights sit near the top of the histogram **without a large clipped area**.
If unsure, bracket two or three stops and keep the darkest set in which the
joints are still clearly visible.

---

## 2b · If you are shooting on a phone

A phone works, and the torch on a second phone makes a better lamp than a desk
lamp — small, bright and close to a point source, which is what keeps the slope
bands from smearing together.

What is different is that **a phone is engineered to defeat this experiment.**
HDR, Night mode, multi-frame stacking and local tone mapping all exist to make
a dark scene look bright and evenly lit. The angled frames here are *supposed*
to be almost black with small bright highlights, and the whole method rests on
brightness being comparable **between** frames. Left on auto, a phone will
rescue each frame independently and hand back four photographs that look good
and mean nothing — with no error, and no sign on the image that anything
happened.

So: **full manual, and every computational feature off.**

### An app that can actually hold the settings

- **Android** — the built-in Pro mode on most phones. Free fallback that works
  everywhere: **Open Camera** (manual ISO, shutter, focus, WB, DNG)
- **iOS** — the stock Camera app cannot set a numeric shutter or ISO. Use
  **Halide**, **ProCamera** or **Obscura**

App and OS behaviour move between versions, so confirm the values actually
stick across four shots before trusting the set.

### §2, translated

| §2 says | on a phone |
|---|---|
| manual exposure, fixed | Pro mode: lock **ISO and shutter as numbers**. Not "AE lock" — actual values, written down. The single most important line here too |
| manual white balance | fixed Kelvin or a preset, never auto |
| manual focus | focus slider, or tap-to-focus then lock |
| RAW if offered | DNG is fine — but `rawpy` is not installed by default, so `cli.py shots` will tell you to `pip install rawpy` or shoot JPEG. JPEG needs nothing extra |
| HDR off | **and also** Night mode, Deep Fusion, Scene Optimiser, "AI enhance", Photographic Styles → off or neutral. This row is the one that differs from a camera |
| no vibration | 2-second timer or voice shutter. Never tap the screen |

- [ ] **Mount it and do not touch it.** A phone tripod clamp, or a stack of
      books and tape. The pull to hand-hold is much stronger than with a camera,
      and hand-holding is what creates the registration problem in §5

### Do not try to gate this on an 0402

A phone's main camera will not focus closer than roughly 7–10 cm, which leaves a
0.5 mm chip a few pixels across. That is fine, because **the gate is judged by
eye** — you need to see the fillet *shape*, not measure it.

- [ ] Shoot the **largest joints on the board**: a tantalum, an electrolytic,
      connector or header pins, SOIC leads, through-hole. Starve one of those

If the sweep is there on a big joint the physics holds, and whether it can be
resolved at 0402 is a question the 12 µm/px design already answers. If it is
*not* there on a big joint, no amount of resolution rescues it — and that is the
gate returning its answer.

### One extra check, because the phone may still be cheating

- [ ] Before the real set: shoot the **mid frame twice, without moving the
      lamp**, and run both through `cli.py shots`. It prints per-channel mean
      and p99. Two shots of an identical scene that differ meaningfully in mean
      brightness mean something is still auto-adjusting, and every comparison
      after that is invalid

Fix that before shooting the set, or the weekend goes on interpreting a
lighting artefact as physics.

> **The one thing that is easier on a phone.** The §7 fallback — four azimuths
> per elevation, combined by per-pixel maximum, which is what a ring actually
> does — is twelve frames. With the phone clamped and a torch in your hand that
> is about five minutes of walking round the board. If the first result is
> ambiguous, go straight there rather than re-reading the single-lamp result.

---

## 3 · Place the lamp

Elevation is measured from the **board plane**, not from vertical. With the lamp
a horizontal distance `d` from the board centre and a height `h` above it:

```
elevation = atan(h / d)          h = d · tan(elevation)
```

| target | ring | d = 20 cm | d = 30 cm |
|---|---|---|---|
| 20° | `low` | h = 7 cm | h = 11 cm |
| 45° | `mid` | h = 20 cm | h = 30 cm |
| 72° | `high` | h = 62 cm | h = 92 cm |

- [ ] Keep the lamp at the **same azimuth** (same compass direction) for all
      three angled frames. Only the elevation changes
- [ ] Keep `d` the same for all three. Only the height changes
- [ ] Rough is fine — ±5° will not hide the effect. Consistency between frames
      matters more than hitting the numbers

> **One lamp is not a ring.** The machine lights from every azimuth at once; a
> single lamp lights from one direction, so only fillets **facing the lamp**
> light up. Their partners at the other end of the same chip will look starved
> and are not. Judge the sweep on joints oriented toward the lamp. To do better,
> see §7.

---

## 4 · Shoot

Four frames. **Touch only the lamp between them.**

- [ ] `low` — lamp at ~20°
- [ ] `mid` — lamp at ~45°
- [ ] `high` — lamp at ~72°
- [ ] `diffuse` — no direct light on the board: bounce the lamp off a white wall
      or ceiling, or shoot outdoors under overcast sky. The goal is no
      directional highlight anywhere

- [ ] Note the camera settings and the lamp distances/heights so the shoot is
      repeatable

---

## 5 · Process

- [ ] Copy the four files into `photos/`, named so each contains its channel
      name: `low.jpg`, `mid.jpg`, `high.jpg`, `diffuse.jpg`

```bash
python cli.py shots --dir photos/            # JPEG: undoes the 2.2 display gamma
python cli.py shots --dir photos/ --linear   # RAW: already linear
python cli.py shots --dir photos/ --no-align # camera definitely never moved
```

- [ ] **Read the registration block it prints.** Frames are registered onto the
      first one, and a few pixels is a normal tripod nudge. If a frame reports a
      large shift, or says `REFUSED`, the camera moved — reshoot, or pass
      `--no-align` if you are certain it did not
- [ ] Open **`out/real_false_colour.png`** ← this is the answer
- [ ] Also written: `out/real_channels.png` (the four frames side by side) and
      `out/real_stack.npy`

If the camera genuinely did not move, `--no-align` is the safest option:
nothing to align beats aligning well.

---

## 6 · Read the result

| what you see | what it means |
|---|---|
| Good joints sweep **blue → green → red**, starved joints have **no red** | The design holds. Proceed to optics |
| Sweep is there but weak or noisy | Probably workable. Tighten exposure, use a smaller lamp, reshoot before concluding anything |
| Every joint looks the same colour | Exposure or white balance was not locked, or the lamp did not really change elevation. Check §2 before blaming the physics |
| Angled frames almost entirely black | Underexposed. The highlights are small; do not meter for the whole frame |
| Large blown-white areas | Highlights clipped, slope information destroyed. Expose down |
| Colours smeared or fringed across every joint | Frames are misregistered. Check the registration block from §5 |
| Good and starved joints genuinely indistinguishable, with settings verified | **This is the finding.** The optical design needs rethinking before any money is spent. That is the gate doing its job |

- [ ] One extra frame: lay a ruler or caliper jaws in the field of view and
      shoot it at the same focus and distance. Pixel scale is currently
      hardcoded (`config.UM_PER_PX = 12.0`) and nothing calibrates it, so this
      shot is what a later scale calibration will need. It is not required for
      the gate and costs ten seconds

---

## 7 · If the first result is ambiguous

Approximate a ring light instead of a single lamp. For each elevation, shoot
four frames at 90° azimuths (N/E/S/W) and combine them per elevation by taking
the **per-pixel maximum** — that is what a ring does. 12 frames plus a diffuse.

`cli.py shots` expects one file per channel, so combine first:

```bash
python - <<'PY'
import cv2, numpy as np, pathlib
for ch in ("low", "mid", "high"):
    fs = sorted(pathlib.Path("photos_azimuth").glob(f"{ch}_*.jpg"))
    m = None
    for f in fs:
        a = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        m = a if m is None else np.maximum(m, a)
    cv2.imwrite(f"photos/{ch}.jpg", m.astype(np.uint8))
    print(ch, "<-", len(fs), "azimuths")
PY
```

This assumes the camera did not move across all 12 frames. If it did, register
them before combining.

---

## What this gate does not test

Deliberately out of scope — none of it blocks the shoot:

- **Pixel scale.** Hardcoded at 12 µm/px with no calibration path, so any
  measurement in microns from a handheld shot would be wrong by whatever the
  real ratio is. The gate is judged by eye, so it does not matter here
- **Running the inspector on the photographs.** That additionally needs a
  centroid file for the physical board, fiducials for registration, and teach on
  real known-good boards
- **Parallax and tiling.** Both unimplemented, both irrelevant to a single
  close-up frame

---

## Related

- `docs/aoi-station-one.html` §1–2 — the reflection model and why each ring
  reports on one slope band
- `README.md` → *Registering frames that do not look alike* — why alignment is
  done the way it is, and how it failed before
- `aoi/capture.py` — the code this drives

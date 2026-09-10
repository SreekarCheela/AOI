"""AOI Station One — the inspection package.

Sixteen modules, in four groups. They are kept flat on purpose: the dependency
order below is short enough to hold in your head, and nesting it would mostly
add path noise to imports that are already unambiguous. What the flat layout
does not do on its own is say which module belongs to what, so it says it here.

WHAT THE MACHINE IS TOLD
    config      optics, channels and every tuned threshold, with the
                measurement that set each one recorded beside it
    packages    the package library: body, land and height geometry, with the
                guards that catch an entry which would silently disable a test
    board       Board and Component, and the placement-file entry point
    centroid    reading the placement files real tools write -- different
                column names, units, delimiters, a title block above the header
    calibrate   sampling pitch and lens distortion, measured from a glass
                target rather than assumed
    tiling      the scan raster, and which camera position owns which component

WHAT IT SEES
    synth       the four-channel renderer, and defect injection. Everything in
                this repository is validated against it, which is exactly why
                `docs/physics-gate-shoot.md` exists
    capture     real photographs -> the same four-channel stack
    roi         the regions each test measures, computed from CAD, never found
                by searching the image

WHAT IT DECIDES
    rules       measurements: band fractions, offsets in microns, body geometry.
                Numbers, never verdicts
    models      the learned layer -- classifier, per-package anomaly subspace,
                and the Torch/ONNX backends that are the only seam between the
                workstation and the Pi
    fusion      rules in precedence order, plus the gradient-boosted tree that
                weighs them together
    pipeline    align the board, measure every component, issue verdicts
    falsecall   the false-call filter and, more importantly, the gate that
                stops it being trusted

WHERE IT ALL LANDS
    store       the station's local record. One SQLite file, no network in the
                loop. Deliberately depends on nothing else in this package
    review      the operator screen. Section 5 of the spec calls this the
                product, because operator judgements are the only training data
                that will ever exist for the false-call filter

The one rule worth stating up front: nothing here searches an image for a
component. Every region comes from the placement file, which is what keeps the
models small enough to train and fast enough to run on a Pi.

Nothing is imported here. `store` in particular is meant to keep working if the
inspection code changes shape, and an eager package import would quietly couple
the two.
"""

__version__ = "0.1.0"

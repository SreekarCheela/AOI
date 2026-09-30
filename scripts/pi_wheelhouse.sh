#!/usr/bin/env bash
# Download every package the Pi needs, on a machine WITH network (a Mac is fine),
# so the Pi can be set up with no network at all:
#
#     scripts/pi_wheelhouse.sh [DIR]          # default DIR: pi-bundle/
#
# DIR ends up holding:
#   wheels/   Linux aarch64 wheels for CPython 3.13 (Raspberry Pi OS Trixie)
#   ocr/      the four pinned OCR model files, hash-checked
#   part_detector.pt, if this machine has a trained detector
# Copy DIR to the Pi (USB stick), then on the Pi:
#     scripts/pi_setup.sh --bundle /media/.../pi-bundle
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-pi-bundle}"
PY="${PYTHON:-.venv/bin/python}"
mkdir -p "$OUT/wheels"

# antlr4-python3-runtime 4.9.3 (rapidocr -> omegaconf) is published as source
# only, so it cannot be fetched --only-binary for another platform. It is pure
# Python: build its wheel here first, and let the download find it there.
"$PY" -m pip wheel --quiet --no-deps --wheel-dir "$OUT/wheels" \
    "$(grep '^antlr4-python3-runtime' requirements-pi.txt)"
"$PY" -m pip download --quiet --dest "$OUT/wheels" --find-links "$OUT/wheels" --only-binary=:all: \
    --implementation cp --python-version 3.13 --abi cp313 --abi abi3 --abi none \
    --platform manylinux_2_28_aarch64 --platform manylinux_2_26_aarch64 \
    --platform manylinux_2_17_aarch64 --platform manylinux2014_aarch64 --platform any \
    -r requirements-pi.txt
"$PY" -m pip download --quiet --dest "$OUT/wheels" --no-deps --only-binary=:all: \
    --implementation cp --python-version 3.13 --platform any rapidocr==3.9.2

mkdir -p "$OUT/ocr"
if "$PY" -m aoi.photo.names verify-models >/dev/null 2>&1; then
    "$PY" -m aoi.photo.names install-models --from models/ocr --to "$OUT/ocr"
else
    "$PY" models/ocr/fetch.py "$OUT/ocr"
fi
if [ -f out/models/part_detector.pt ]; then
    cp out/models/part_detector.pt "$OUT/"
else
    echo "note: no out/models/part_detector.pt here; teach on the Pi will use colour proposals"
fi
echo "bundle ready: $OUT ($(du -sh "$OUT" | cut -f1)). Copy it to the Pi and run scripts/pi_setup.sh --bundle DIR"

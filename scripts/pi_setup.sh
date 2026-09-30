#!/usr/bin/env bash
# Set up the photo AOI on a Raspberry Pi 5 running Raspberry Pi OS (64-bit,
# Trixie-based, Python 3.13). Run from the repository on the Pi:
#
#     scripts/pi_setup.sh                       # with network: apt + pip from PyPI
#     scripts/pi_setup.sh --bundle DIR          # offline: DIR from scripts/pi_wheelhouse.sh
#
# Options: --bundle DIR   wheels/, ocr/ and part_detector.pt made on a networked machine
#          --no-apt       skip apt (python3-venv and python3-tk already installed)
# Safe to run again; it ends with `python -m aoi_photo doctor`.
set -euo pipefail
cd "$(dirname "$0")/.."
BUNDLE=""; APT=1
while [ $# -gt 0 ]; do
    case "$1" in
        --bundle) BUNDLE="$(cd "$2" && pwd)"; shift 2 ;;
        --no-apt) APT=0; shift ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

python3 - <<'EOF'
import platform, sys
if sys.version_info < (3, 12):
    sys.exit(f"Python {platform.python_version()}: numpy and scipy need 3.12+. Install the Trixie-based "
             "Raspberry Pi OS (Python 3.13); Bookworm ships 3.11.")
if platform.machine() not in ("aarch64", "x86_64"):
    sys.exit(f"{platform.machine()}: install the 64-bit Raspberry Pi OS")
EOF

if [ "$APT" = 1 ]; then
    missing=""
    for p in python3-venv python3-tk; do dpkg -s "$p" >/dev/null 2>&1 || missing="$missing $p"; done
    if [ -n "$missing" ]; then
        echo "installing$missing (needs network, or pass --no-apt if they are already there)"
        sudo apt-get install -y $missing
    fi
fi

[ -x .venv/bin/python ] || python3 -m venv .venv
PIP=(.venv/bin/python -m pip install --quiet)
if [ -n "$BUNDLE" ]; then
    PIP+=(--no-index --find-links "$BUNDLE/wheels")
fi
"${PIP[@]}" -r requirements-pi.txt
# rapidocr's metadata asks for the GUI opencv-python, which would replace the
# headless build: its real dependencies are in requirements-pi.txt
"${PIP[@]}" --no-deps rapidocr==3.9.2

if [ -n "$BUNDLE" ] && [ -d "$BUNDLE/ocr" ]; then
    .venv/bin/python -m aoi.photo.names install-models --from "$BUNDLE/ocr"
fi
if [ -n "$BUNDLE" ] && [ -f "$BUNDLE/part_detector.pt" ]; then
    mkdir -p out/models && cp "$BUNDLE/part_detector.pt" out/models/
fi

echo
.venv/bin/python -m aoi_photo doctor || true
echo
echo "start the app:   .venv/bin/python -m aoi_photo"
echo "time this Pi:    .venv/bin/python -m aoi_photo bench --program NAME PHOTO_FOLDER"

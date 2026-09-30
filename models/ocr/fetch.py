"""Download the pinned OCR model files, on a machine with network.

    python models/ocr/fetch.py [DIR]

Downloads into DIR (default: this folder), checks every file against the
SHA-256 pinned in aoi/photo/names.py, and installs it. For the offline machine,
copy the resulting folder over and run there:

    python -m aoi.photo.names install-models --from /path/to/copied/folder

This is the only place the model files are fetched from; the app never is.
"""
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aoi.photo import names  # noqa: E402

HUB = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/"


def main() -> int:
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    missing = names.verify_models(dest)
    if not missing:
        print(f"{dest}: all model files present and verified")
        return 0
    with tempfile.TemporaryDirectory() as d:
        for name in missing:
            print(f"downloading {name} ...")
            with urllib.request.urlopen(HUB + names.MODELS[name][2], timeout=120) as r, \
                    open(Path(d) / name, "wb") as fh:
                shutil.copyfileobj(r, fh)
        return names.main(["install-models", "--from", d, "--to", str(dest)])


if __name__ == "__main__":
    sys.exit(main())

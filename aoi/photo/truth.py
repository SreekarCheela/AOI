"""truth.json: which defects were planted on which board photo.

The file `eval` scores against (aoi/photo/evaluate.py), written by the labelling
window (`python -m aoi_photo label`). The format is the fixtures':

    {"goldens": ["golden1.heic"],                       photos in the folder that are not boards
     "truth": [{"image": "test1", "refdes": "C6", "defect": "missing"}, ...],
     ...}                                               any other key is kept as it was

A board photo with no entry is a good board.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .evaluate import EXPECTED
from .io import photo_files

DEFECTS: tuple[str, ...] = tuple(EXPECTED)          # missing, offset, skew, wrong_part, tombstone, reversed, marking_differs


class Truth:
    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.path = self.folder / "truth.json"
        self.data: dict = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.data.setdefault("truth", [])

    # ------------------------------------------------------------------ read
    @property
    def goldens(self) -> set[str]:
        """Stems of the photos marked as goldens (not boards)."""
        names = list(self.data.get("goldens", [])) + ([self.data["golden"]] if self.data.get("golden") else [])
        return {Path(n).stem for n in names}

    def photos(self) -> list[Path]:
        """Every photo in the folder, goldens included, by name."""
        return photo_files(self.folder)

    def boards(self) -> list[Path]:
        return [p for p in self.photos() if p.stem not in self.goldens]

    def on(self, image: str) -> dict[str, str]:
        """refdes -> defect planted on this board photo (by stem)."""
        return {t["refdes"]: t["defect"] for t in self.data["truth"] if t["image"] == image}

    # ----------------------------------------------------------------- write
    def set(self, image: str, refdes: str, defect: str | None) -> None:
        """Plant (or with None, clear) one part's defect on one board, and save."""
        if defect is not None and defect not in DEFECTS:
            raise ValueError(f"unknown defect {defect!r}; one of {', '.join(DEFECTS)}")
        refdes = refdes.strip().upper()
        if not refdes:
            raise ValueError("a part needs a name")
        rows = [t for t in self.data["truth"] if not (t["image"] == image and t["refdes"] == refdes)]
        if defect is not None:
            rows.append({"image": image, "refdes": refdes, "defect": defect})
        self.data["truth"] = sorted(rows, key=lambda t: (t["image"], t["refdes"]))
        self.save()

    def set_golden(self, photo: Path, golden: bool) -> None:
        """Mark a photo in the folder as a golden (not a board) or back."""
        if golden and self.on(photo.stem):
            raise ValueError(f"{photo.name} has planted defects; clear them before calling it a golden")
        names = [n for n in self.data.get("goldens", []) if Path(n).stem != photo.stem]
        if self.data.get("golden") and Path(self.data["golden"]).stem == photo.stem:
            del self.data["golden"]
            if golden:
                names.append(photo.name)
        elif golden:
            names.append(photo.name)
        self.data["goldens"] = sorted(names)
        self.save()

    def save(self) -> None:
        """Atomically: a crash mid-write never leaves half a truth file."""
        self.folder.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.folder, prefix=".truth-", suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(self.data, fh, indent=1)
            fh.write("\n")
        os.replace(tmp, self.path)

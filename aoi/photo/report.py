"""What a photo-mode run leaves behind: an annotated PNG, a CSV, one HTML file.

The HTML is one file with everything inside it -- CSS inline, images as base64
data URIs, no fonts, scripts or stylesheets fetched from anywhere. It gets
mailed, dropped on a share, opened on a line PC with no internet, and opened
again next year; a report that needs a network to render is one that shows up
as a page of broken images on the day somebody needs it.

Two things every export says out loud, because a reader would otherwise assume
the opposite:

* A score is not a probability until it has been calibrated on operator labels.
  Uncalibrated scores print as "score 0.97", calibrated ones as "P=0.97"
  (Call.score_text), and the HTML notes which is which.
* TOMBSTONE and WRONG PART have never been shown to work on a real board. Every
  export marks them unvalidated so nobody reads one as an established check.
"""
from __future__ import annotations

import base64
import csv
import html
from pathlib import Path

import cv2
import numpy as np

from .model import UNVALIDATED_NOTE, BoardResult, Call, Part, PartResult

# BGR, per severity.
COLOURS: dict[str, tuple[int, int, int]] = {
    "FAIL": (0, 0, 255),
    "REVIEW": (0, 191, 255),
    "INFO": (160, 160, 160),
    "OK": (0, 180, 0),
}
VERDICT_SEVERITY = {"PASS": "OK", "FAIL": "FAIL", "REVIEW": "REVIEW"}

HTML_BOARD_SIDE = 2000      # long side of the board image embedded in the HTML
HTML_CROP_SIDE = 240        # long side of each crop thumbnail
JPEG_QUALITY = 85

CSV_FIELDS = ("image", "refdes", "type", "package", "check", "state", "severity",
              "score", "calibrated", "unvalidated", "detail", "dx_px", "dy_px",
              "dtheta_deg")


# ------------------------------------------------------------------ helpers
def _worst(calls: list[Call]) -> Call:
    """The call a flagged part is drawn and labelled by: first FAIL, else REVIEW."""
    for sev in ("FAIL", "REVIEW"):
        for c in calls:
            if c.severity == sev:
                return c
    return calls[0]


def _label(pr: PartResult) -> str:
    """"R12 MISSING score 0.97", plus the unvalidated tag and any further calls."""
    c = _worst(pr.flagged)
    text = " ".join(s for s in (pr.refdes, c.state, c.score_text()) if s)
    if c.unvalidated:
        text += " (unvalidated)"
    if len(pr.flagged) > 1:
        text += f" +{len(pr.flagged) - 1}"
    return text


def found_at(pr: PartResult, result: BoardResult) -> tuple[float, float]:
    """Top-left of where the part's body was found, in aligned golden-1 pixels.

    dx_px, dy_px are net of the board-wide common shift and were measured in a
    window already moved by the part's local re-registration, so both go back
    on: taught box + local shift + common shift + offset."""
    m = pr.measures or {}
    sx, sy = result.common_shift_px
    return (pr.part.x + m.get("local_dx", 0.0) + sx + pr.dx_px,
            pr.part.y + m.get("local_dy", 0.0) + sy + pr.dy_px)


def _scale(img: np.ndarray) -> tuple[int, float]:
    """Line thickness and font scale for an image this size (as inspect_photo.draw)."""
    t = max(1, int(round(max(img.shape[:2]) / 1500)))
    return t, 0.4 * t + 0.1


def _text(img: np.ndarray, text: str, org: tuple[int, int], col, fs: float, t: int) -> None:
    """Text on a dark backing, so a label stays readable over bright solder."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
    x, y = org
    cv2.rectangle(img, (x, y - h - base), (x + w, y + base), (20, 20, 20), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, col, t, cv2.LINE_AA)


def _jpeg_uri(img: np.ndarray, side: int) -> str:
    """Downscale to `side` on the long edge (never up) and encode as a data URI."""
    s = side / max(img.shape[:2])
    if s < 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("could not encode image as JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# ------------------------------------------------------------------ images
def annotate(image_bgr: np.ndarray, result: BoardResult) -> np.ndarray:
    """The aligned test photo with every part outlined and every call labelled.

    The image is already in golden-1 pixels, so each Part's box is where the
    part should be. That is drawn faint for every part; a flagged part gets a
    thick box in its worst severity's colour where it was actually found
    (`found_at`: the taught box moved by the local, common and part shifts).
    The verdict and the stamp go in a banner drawn over the top of the photo,
    so the output is the input's shape.
    """
    out = image_bgr.copy()
    H, W = out.shape[:2]
    t, fs = _scale(out)

    # Taught positions, blended in at a third so they read as context and do
    # not compete with the calls.
    faint = out.copy()
    for pr in result.parts:
        p = pr.part
        cv2.rectangle(faint, (p.x, p.y), (p.x + p.w, p.y + p.h), (220, 220, 220), 1)
    out = cv2.addWeighted(faint, 0.35, out, 0.65, 0.0)

    # Banner first, so the part labels can be kept below it rather than
    # hidden under it: a dark strip over the top rows, light enough that a
    # part underneath is not blacked out.
    head = result.verdict + (f"   {result.stamp}" if result.stamp else "")
    (_, th), base = cv2.getTextSize(head, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.4, t + 1)
    bh = min(H, th + base + 12 * t)
    dark = np.full_like(out[:bh], 20)
    out[:bh] = cv2.addWeighted(dark, 0.8, out[:bh], 0.2, 0.0)
    col = COLOURS[VERDICT_SEVERITY.get(result.verdict, "INFO")]
    cv2.putText(out, head, (6 * t, 6 * t + th), cv2.FONT_HERSHEY_SIMPLEX,
                fs * 1.4, col, t + 1, cv2.LINE_AA)

    (_, lh), lbase = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, fs, t)
    flagged = [pr for pr in result.parts if pr.flagged]
    boxes = []
    for pr in flagged:
        p = pr.part
        col = COLOURS[_worst(pr.flagged).severity]
        fx, fy = found_at(pr, result)
        x, y = int(round(fx)), int(round(fy))
        cv2.rectangle(out, (x, y), (x + p.w, y + p.h), col, 2 * t + 1)
        boxes.append((pr, col, x, y))
    # Labels after every box, each where it hides no earlier label: above its
    # box, else below it, else stepped further up. A cluster of missing 0603s
    # otherwise prints its labels on top of each other.
    placed: list[tuple[int, int, int, int]] = []
    step = lh + 2 * lbase + 2
    for pr, col, x, y in boxes:
        text = _label(pr)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        above = max(y - 4 * t - lbase, bh + lh + 2)
        below = min(y + pr.part.h + lh + 4 * t, H - lbase - 1)
        tries = [above, below] + [max(above - k * step, bh + lh + 2) for k in range(1, 8)]
        for ty in tries:
            r = (x, ty - lh - lbase, x + tw, ty + lbase)
            if not any(r[0] < q[2] and q[0] < r[2] and r[1] < q[3] and q[1] < r[3] for q in placed):
                break
        placed.append((x, ty - lh - lbase, x + tw, ty + lbase))
        _text(out, text, (x, ty), col, fs, t)
    return out


def crop_pair(golden_bgr: np.ndarray, test_bgr: np.ndarray, part: Part,
              margin_frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """The same window from golden and test, around one part.

    The margin is a fraction of the part's long side, so an offset part is
    still inside its own crop. The window is clipped to the smaller of the two
    images, and can be empty if the part lies wholly outside both.
    """
    m = int(round(margin_frac * max(part.w, part.h)))
    H = min(golden_bgr.shape[0], test_bgr.shape[0])
    W = min(golden_bgr.shape[1], test_bgr.shape[1])
    x0, y0 = max(0, part.x - m), max(0, part.y - m)
    x1, y1 = min(W, part.x + part.w + m), min(H, part.y + part.h + m)
    x1, y1 = max(x0, x1), max(y0, y1)
    return golden_bgr[y0:y1, x0:x1].copy(), test_bgr[y0:y1, x0:x1].copy()


# ------------------------------------------------------------------ exports
def _csv_row(result: BoardResult, pr: PartResult | None, c: Call) -> dict:
    p = pr.part if pr is not None else None
    return {
        "image": result.image,
        "refdes": p.refdes if p else "",
        "type": p.type if p else "",
        "package": p.package if p else "",
        "check": c.check,
        "state": c.state,
        "severity": c.severity,
        "score": "" if c.score is None else f"{c.score:.4f}",
        "calibrated": int(c.calibrated),
        "unvalidated": int(c.unvalidated),
        "detail": c.detail,
        "dx_px": f"{pr.dx_px:.2f}" if pr else "",
        "dy_px": f"{pr.dy_px:.2f}" if pr else "",
        "dtheta_deg": f"{pr.dtheta_deg:.2f}" if pr else "",
    }


def write_csv(result: BoardResult, path: str | Path) -> None:
    """One row per call, part-level then board-level (empty refdes)."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for pr in result.parts:
            for c in pr.calls:
                w.writerow(_csv_row(result, pr, c))
        for c in result.board_calls:
            w.writerow(_csv_row(result, None, c))


_CSS = """
:root { --fail: #d62828; --review: #e09a00; --info: #808080; --ok: #2a9d3a;
        --ink: #1b1b1b; --muted: #666; --line: #ddd; --bg: #fafafa; }
* { box-sizing: border-box; }
body { margin: 0; padding: 16px; background: var(--bg); color: var(--ink);
       font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 24px 0 8px; }
.verdict { display: inline-block; padding: 2px 10px; border-radius: 4px;
           color: #fff; font-weight: 700; }
.FAIL { background: var(--fail); } .REVIEW { background: var(--review); }
.INFO { background: var(--info); } .OK, .PASS { background: var(--ok); }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 2px 12px; margin: 8px 0; }
dt { color: var(--muted); } dd { margin: 0; }
.board img { max-width: 100%; height: auto; border: 1px solid var(--line); }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; background: #fff; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: left;
         vertical-align: top; }
th { font-weight: 600; background: #f0f0f0; }
.sev { display: inline-block; padding: 0 6px; border-radius: 3px; color: #fff;
       font-size: 12px; font-weight: 600; }
.crops img { width: 120px; height: auto; margin-right: 4px; border: 1px solid var(--line); }
.tag { color: var(--fail); font-weight: 600; }
.note { color: var(--muted); font-size: 13px; }
"""


def _e(x) -> str:
    return html.escape(str(x))


def _score_cell(c: Call) -> str:
    if c.score is None:
        return ""
    if c.calibrated:
        return _e(c.score_text())
    return f'{_e(c.score_text())} <span class="note">(not calibrated)</span>'


def _crop_cell(pair: tuple[np.ndarray, np.ndarray] | None) -> str:
    if pair is None or min(pair[0].shape[:2] + pair[1].shape[:2]) == 0:
        return '<span class="note">no crop</span>'
    g, t = (_jpeg_uri(a, HTML_CROP_SIDE) for a in pair)
    return (f'<img src="{g}" alt="golden" title="golden">'
            f'<img src="{t}" alt="test" title="test">')


def _call_cells(c: Call) -> str:
    unval = f'<span class="tag">{_e(UNVALIDATED_NOTE)}</span>' if c.unvalidated else ""
    return (f'<td><span class="sev {_e(c.severity)}">{_e(c.state)}</span></td>'
            f"<td>{_e(c.check)}</td><td>{_score_cell(c)}</td>"
            f"<td>{unval}</td><td>{_e(c.detail)}</td>")


def write_html(result: BoardResult, annotated_bgr: np.ndarray,
               crops: dict[str, tuple[np.ndarray, np.ndarray]], path: str | Path) -> None:
    """One self-contained HTML file: header, board image, flagged calls with crops."""
    r = result
    sx, sy = r.common_shift_px
    reasons = "".join(f"<li>{_e(x)}</li>" for x in r.reasons) or "<li>none</li>"
    header = (
        f"<h1>{_e(Path(r.image).name)} "
        f'<span class="verdict {_e(r.verdict)}">{_e(r.verdict)}</span></h1>'
        f"<dl>"
        f"<dt>program</dt><dd>{_e(r.program)} v{_e(r.program_version)}"
        f"{' (single golden)' if r.single_golden else ''}</dd>"
        f"<dt>stamp</dt><dd>{_e(r.stamp) or '&mdash;'}</dd>"
        f"<dt>scale</dt><dd>{f'{r.px_per_mm:.2f} px/mm' if r.px_per_mm > 0 else 'none (offsets in px)'}</dd>"
        f"<dt>common shift</dt><dd>{sx:+.1f}, {sy:+.1f} px</dd>"
        f"<dt>noise</dt><dd>{r.noise_px:.2f} px</dd>"
        f"<dt>offset limit</dt><dd>{r.offset_limit_px:.1f} px</dd>"
        f"<dt>reasons</dt><dd><ul>{reasons}</ul></dd>"
        f"</dl>")

    rows = []
    for pr in r.parts:
        flagged = pr.flagged
        if not flagged:
            continue
        crop = _crop_cell(crops.get(pr.refdes))
        for i, c in enumerate(flagged):
            lead = (f'<td rowspan="{len(flagged)}">{_e(pr.refdes)}<br>'
                    f'<span class="note">{_e(pr.part.group)}</span></td>'
                    f'<td rowspan="{len(flagged)}" class="crops">{crop}</td>'
                    if i == 0 else "")
            rows.append(f"<tr>{lead}{_call_cells(c)}</tr>")
    for c in r.board_calls:
        if c.severity in ("FAIL", "REVIEW"):
            rows.append(f'<tr><td>board</td><td class="crops"></td>{_call_cells(c)}</tr>')
    if rows:
        table = ('<div class="wrap"><table><thead><tr><th>part</th>'
                 "<th>golden | test</th><th>state</th><th>check</th><th>score</th>"
                 "<th>unvalidated</th><th>detail</th></tr></thead><tbody>"
                 + "".join(rows) + "</tbody></table></div>")
    else:
        table = '<p class="note">No flagged calls.</p>'

    calls = [c for pr in r.parts for c in pr.flagged] + list(r.board_calls)
    notes = []
    if any(c.score is not None and not c.calibrated for c in calls):
        notes.append("Scores marked not calibrated are rankings, not probabilities: "
                     "they become P= only after calibration on operator labels.")
    if any(c.unvalidated for c in calls):
        notes.append("TOMBSTONE and WRONG PART are " + UNVALIDATED_NOTE + ".")
    note_html = "".join(f'<p class="note">{_e(n)}</p>' for n in notes)

    n_flagged = sum(1 for pr in r.parts if pr.flagged)
    doc = (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_e(Path(r.image).stem)} {_e(r.verdict)}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"{header}"
        f'<div class="board"><img src="{_jpeg_uri(annotated_bgr, HTML_BOARD_SIDE)}" '
        f'alt="annotated board"></div>'
        f"<h2>Flagged calls ({n_flagged} of {len(r.parts)} parts)</h2>"
        f"{table}{note_html}"
        "</body></html>\n")
    Path(path).write_text(doc, encoding="utf-8")


def write_all(result: BoardResult, golden_bgr: np.ndarray, aligned_test_bgr: np.ndarray,
              out_dir: str | Path) -> dict[str, Path]:
    """Every export for one run, named after the photo: PNG, CSV and HTML."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.image).stem
    paths = {"annotated": out_dir / f"{stem}_annotated.png",
             "csv": out_dir / f"{stem}.csv",
             "html": out_dir / f"{stem}.html"}

    annotated = annotate(aligned_test_bgr, result)
    if not cv2.imwrite(str(paths["annotated"]), annotated):
        raise OSError(f"could not write {paths['annotated']}")
    # Crops come from the photo, not the annotated copy: an operator judging a
    # part needs to see the part, not the box drawn over it.
    crops = {pr.refdes: crop_pair(golden_bgr, aligned_test_bgr, pr.part)
             for pr in result.parts if pr.flagged}
    write_csv(result, paths["csv"])
    write_html(result, annotated, crops, paths["html"])
    return paths

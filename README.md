# AOI — ESP32 Vision Replacement for Solder-Joint AOI

Replacing a commercial PCB solder-joint AOI (Automated Optical Inspection) machine
with a DIY system: ESP32 camera capture + edge-device inference running a custom
vision model. Goal: avoid buying a second commercial AOI unit.

**Full spec:** [Epic #1](https://github.com/SreekarCheela/AOI/issues/1)

## Current Phase: Shadow Mode (Phase 1)

Running the new system alongside the existing AOI, comparing results, with
**zero line authority** — no PLC integration, no live gating. Cutover to live
gating is a future phase, contingent on shadow-mode results.

## Status

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| [#1](https://github.com/SreekarCheela/AOI/issues/1) | Epic | Filed | Full spec, pre-flight checklist, acceptance criteria |
| [#2](https://github.com/SreekarCheela/AOI/issues/2) | Hardware rig: design & procurement | Blocked | Needs pre-flight checklist items 4-6 (BOM owner, safety sign-off, IT network check) |
| [#3](https://github.com/SreekarCheela/AOI/issues/3) | Physical install + pre-flight checklist | Blocked | Depends on #2; needs plant-side answers (see Pre-Flight Checklist below) |
| [#4](https://github.com/SreekarCheela/AOI/issues/4) | Data pipeline | **In progress** | Agent running in `../worktrees/data-pipeline-issue-3-20894` — dataset prep + logging/correlation/labeling scaffolding |
| [#5](https://github.com/SreekarCheela/AOI/issues/5) | Model training | **In progress** | Agent running in `../worktrees/model-training-issue-4-21625` — architecture + training pipeline + baseline transfer-learning run on public data. Fine-tuning stubbed until real line data exists |
| [#6](https://github.com/SreekarCheela/AOI/issues/6) | Shadow-mode run + analysis | Not started | Depends on #3, #5 |

*(Note: GitHub issue numbers above may not exactly match the informal #2-#6 labels used in early planning discussion — always check the linked issue for the authoritative title/scope.)*

## Pre-Flight Checklist (blocking hardware work — needs answers from whoever owns the line/AOI)

1. [ ] Do boards carry a barcode/QR/serial for reliable cross-system correlation? (fallback: sequence + timestamp)
2. [ ] Existing AOI's PLC signal protocol (not needed for shadow mode; needed before any future live-gating cutover)
3. [ ] Does the existing AOI's report/export include per-joint location data? (determines whether per-joint defect localization is feasible as a fast-follow)
4. [ ] Finest component/joint size on the board, from whoever owns the BOM (drives camera/lens spec — placeholder assumes down to 0402/0201 passives, ~0.3-0.5mm features)
5. [ ] Line safety/guarding sign-off for mounting new hardware near the conveyor
6. [ ] Plant IT/network capacity check if image traffic touches shared infrastructure

## Architecture (locked decisions)

- **ESP32** — capture-only (image sensor + trigger logic), no on-device inference
- **Trigger sensor** — new IR/limit-switch sensor added at the station
- **Custom lighting rig** — dedicated LED lighting, not ambient
- **Edge compute** — Jetson Orin Nano-class device runs inference locally (on-prem, IP/confidentiality)
- **Network** — ESP32 + edge device isolated from the plant's OT/PLC network
- **Power** — independent supply, not tapped from line control power
- **Mounting** — immediately downstream of the existing AOI station, same conveyor path

## Model Plan

- **MVP scope:** whole-board classification (pass/fail + defect type), not per-joint localization
- **Pretraining:** transfer learning from public datasets — [SolDef_AI](https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection), [PCB-AoI](https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi)
- **Fine-tuning ground truth:** existing AOI's own pass/fail verdicts during shadow mode (accuracy ceiling = existing AOI's accuracy)
- **Correlation:** sequence + timestamp by default, upgraded to board ID if checklist item 1 confirms one exists

## Shadow-Mode Acceptance Criteria

- ≥98% pass/fail agreement with the existing AOI
- ≤0.5% escape rate (model says pass, actually defective)
- ≤3% false-reject rate (model says fail, actually good)
- Over ≥2 weeks or ≥10,000 boards
- Weekly manual spot-checks for camera/lighting alignment drift

## Out of Scope (this phase)

- Other inspection stations on the line
- PCB redesign
- Automated reject/rework handling
- Live human review during shadow mode (batch comparison only)
- PLC/line integration and live-gating cutover
- Per-joint defect localization (fast-follow, contingent on checklist item 3)

## Rollback

Shadow mode has no line authority — rollback is unmounting/powering off the rig; production is unaffected. Main risk is silently-bad data from alignment/lighting drift, mitigated by weekly spot-checks.

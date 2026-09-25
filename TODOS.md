# TODOS

Deferred from the Photo AOI plan (~/.gstack/projects/SreekarCheela-AOI/main-design-photo-aoi-app.md).

- **Shoot 2+ more goldens of PT-PCB-004-03** (P1). The fixture has one golden, so
  it runs in stamped single-golden mode; REVERSED, wrong-part and golden-spread
  thresholds can't be exercised on real photos until it has three.
- **Missing parts on dull solder lands** (P1). test3 R11/R14 read as a body on
  every feature tried. Try a raking-light or second-angle shot of the same board
  before more features.
- **E7 centroid import + two-fiducial click** (P2, M). Teach from the placement
  file instead of boxes; aoi/centroid.py already reads the files.
- **E8 marking OCR** (P3, L). Wrong value by marking (R19 3303 vs 3203) is not
  detected; needs more resolution or OCR. Revisit if a wrong-value escape shows up.
- **E10 trend / SPC view** (P3, M). After 4+ weeks of runs in photo_aoi.db.
- **E11 multi-photo test fusion** (P2, M). Up to 3 shots per board, least-glary
  wins; after single-photo false-call rate is measured.
- **E12 escape tracking via pass audits** (P2, M). Stats counts false calls only;
  a missed defect never reaches the operator.
- **Part detector on new board types** (P2, M). It has seen one board type
  (PT-PCB-004-03); held out across cameras it found 74% of parts tight. Teach 2+
  more board types, retrain, and measure it on a board it has not seen.
- **Detector training speed** (P3, S). About 1.3 s per iteration on MPS; a
  from-scratch run is about 85 minutes. Try a smaller tile or fewer iterations
  when fine-tuning from the last model.
- **autoplan phase dispatch blocked** (tooling). gstack's phase-publication-hook
  refuses native reviewer Agent dispatch in this Claude Code build; the Design,
  DX and Eng review phases of this plan did not run.

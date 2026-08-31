# Shadow-Mode Comparison Report — SYNTHETIC DEMO — not a real run

**Run window:** 2026-08-01 to 2026-08-31

## Correlation Summary

- Ground-truth (existing AOI) records: 2000
- Model (new system) records: 2000
- Matched pairs: 2000
- Unmatched ground-truth records: 0
- Unmatched model records: 0

## Acceptance Criteria

| Metric | Observed | Threshold | Result |
|---|---|---|---|
| Pass/fail agreement | 98.70% | >= 98.00% | PASS |
| Escape rate | 0.05% | <= 0.50% | PASS |
| False-reject rate | 1.25% | <= 3.00% | PASS |

Agree: 1974 / Escapes: 1 / False-rejects: 25 (of 2000 matched boards)

## Per-Defect-Type Breakdown

Catch rate = fraction of ground-truth boards with this defect type that the model also failed.

| Defect Type | GT Count | Caught | Escaped | Catch Rate |
|---|---|---|---|---|
| INSUFFICIENT_SOLDER | 47 | 46 | 1 | 97.87% |
| MISSING_COMPONENT | 32 | 32 | 0 | 100.00% |
| MISALIGNMENT | 28 | 28 | 0 | 100.00% |
| TOMBSTONE | 22 | 22 | 0 | 100.00% |
| SOLDER_BRIDGE | 22 | 22 | 0 | 100.00% |

## Weekly Spot-Checks (camera/lighting alignment drift)

- This is placeholder synthetic data generated for demo purposes; no real shadow-mode run has occurred yet.

## Recommendation

**Proceed Toward Live Gating Cutover**

All acceptance criteria are met with no significant per-defect-type weak spots. The system appears ready for live-gating cutover planning (actual PLC/line integration is out of scope for this epic).

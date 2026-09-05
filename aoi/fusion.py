"""Verdict fusion.

Two paths. `rule_verdict` needs no training and works the day the machine is
built. `FusionModel` fits a gradient-boosted tree over the same measurements
plus the anomaly score -- it trains on hundreds of samples rather than
thousands, reports feature importances, and gives calibrated confidence
(spec section 6). Rules and model do not vote; the rule *measurements* are
features the tree consumes.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .rules import FEATURE_NAMES, Golden, Measure


@dataclass
class Verdict:
    refdes: str
    package: str
    failed: bool
    score: float           # 0..1, higher = more likely a real defect
    reason: str
    detail: str = ""


def rule_verdict(m: Measure, golden: Golden) -> Verdict:
    """Threshold logic over the measurements. The pre-training baseline.

    Tested in precedence order, not by severity. Gross faults have to be settled
    before joint quality is even asked about: a part that is absent, bridged or
    badly displaced makes its fillet windows meaningless, and scoring by
    severity lets the fillet test -- which fires hard on every one of those --
    mislabel them all as insufficient solder. Real AOI gates the same way.
    """
    g = golden.packages.get(m.package)

    def V(reason: str, sev: float, detail: str) -> Verdict:
        return Verdict(m.refdes, m.package, True, float(min(1.0, 0.5 + sev / 2.0)),
                       reason, detail)

    # 1. bridge -- unambiguous, and it wrecks every other measurement
    if g is not None and m.gap_spec > g.gap_mean + 5.0 * g.gap_std:
        sev = (m.gap_spec - g.gap_mean) / max(g.gap_std, 1e-3) / 5.0
        return V("bridge", sev, f"gap specular {m.gap_spec:.3f} vs {g.gap_mean:.3f}")

    # 2. reversed -- the part is present, so this must precede the presence test
    if m.pol_margin < -0.05:
        return V("reversed", 1.0 + abs(m.pol_margin),
                 f"180 deg template wins by {abs(m.pol_margin):.3f}")

    # 3. presence. Both joints starved means nothing was ever placed; one joint
    #    still wetted means a part is there but stood up.
    if m.ncc < 0.62:
        collapsed = 2
        if g is not None:
            collapsed = int(sum(m.bands[i, 0] < 0.30 * g.band_mean[i, 0] for i in range(2)))
        kind = "missing" if collapsed >= 2 else "tombstone"
        return V(kind, 1.0 - m.ncc, f"NCC {m.ncc:.2f}, {collapsed}/2 joints unwetted")

    # 4. placement
    off = float(np.hypot(m.dx_um, m.dy_um))
    if off > config.MAX_OFFSET_UM:
        return V("offset", off / config.MAX_OFFSET_UM, f"{off:.0f} um off nominal")
    if abs(m.dtheta_deg) > config.MAX_SKEW_DEG:
        return V("skew", abs(m.dtheta_deg) / config.MAX_SKEW_DEG, f"{m.dtheta_deg:+.1f} deg")

    # 5. joint quality -- only now is the fillet window trustworthy
    if g is not None:
        for pad in range(2):
            obs, ref = m.bands[pad, 0], g.band_mean[pad, 0]
            std = max(g.band_std[pad, 0], 1e-3)
            if obs < ref - 3.0 * std:
                return V("insufficient", (ref - obs) / std / 3.0,
                         f"pad {pad} low-angle band {obs:.2f} vs {ref:.2f}")

    return Verdict(m.refdes, m.package, False, 0.0, "ok")


class FusionModel:
    """Gradient-boosted tree over the measurement vector + anomaly score."""

    def __init__(self) -> None:
        self.clf = None
        self.importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FusionModel":
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.clf = HistGradientBoostingClassifier(
            max_iter=220, learning_rate=0.08, max_depth=4,
            l2_regularization=1.0, random_state=0)
        self.clf.fit(np.nan_to_num(X), y)
        from sklearn.inspection import permutation_importance
        r = permutation_importance(self.clf, np.nan_to_num(X), y, n_repeats=5,
                                   random_state=0, n_jobs=1)
        self.importances_ = r.importances_mean
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("fusion model not fitted")
        return self.clf.predict_proba(np.nan_to_num(X))[:, 1]

    def top_features(self, k: int = 6) -> list[tuple[str, float]]:
        if self.importances_ is None:
            return []
        idx = np.argsort(self.importances_)[::-1][:k]
        return [(FEATURE_NAMES[i], float(self.importances_[i])) for i in idx]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> "FusionModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)

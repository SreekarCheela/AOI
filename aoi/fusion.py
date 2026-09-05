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
from .rules import FEATURE_NAMES, Golden, Measure, band_distance


@dataclass
class Verdict:
    refdes: str
    package: str
    failed: bool
    score: float           # 0..1, higher = more likely a real defect
    reason: str
    detail: str = ""
    testable: bool = True  # False = no inspection program for this package, so
                           # this component was not judged either way

    @property
    def passed(self) -> bool:
        """Inspected and found good. An untestable component is neither."""
        return self.testable and not self.failed


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

    # 2. presence, decided on WHERE THE BODY IS rather than how well it
    #    correlates.
    #
    #    NCC answers one question -- does this look like a good part -- and
    #    collapses every way of not being one into a single low number. It
    #    cannot tell a body that is absent from a body that is somewhere else,
    #    which is why missing-vs-displaced and tombstone-vs-displaced both
    #    resisted every threshold tried on it: the populations genuinely overlap
    #    in NCC. Segmenting the body and measuring it answers directly.
    #
    #      no blob                     nothing was placed
    #      spans its own width         standing on one land
    #      spans its own length        lying flat; offset/skew will name it
    #
    #    The aspect test normalises by the body's WIDTH, not its golden length,
    #    because a part standing on end spans exactly its width along the pad
    #    axis whatever the package. Every tombstone measures 1.00; anything
    #    lying flat measures its package aspect ratio, 1.5 to 2.2 here.
    if g is not None:
        body_gone = m.body_area_ratio < config.BODY_PRESENT_AREA

        # Both signals have to agree before calling a part absent. Segmentation
        # is the sharper instrument but it is also the one that can fail wholesale
        # if a real board's body and substrate turn out closer in tone than the
        # golden boards suggested -- and a segmentation that finds nothing would
        # otherwise flag every component as missing. If correlation still sees a
        # body, believe it and fall through.
        if body_gone and m.ncc < g.ncc_absent:
            return V("missing", 1.0 - m.ncc / max(g.ncc_mean, 1e-6),
                     f"no body found (area {m.body_area_ratio:.2f} of golden, "
                     f"NCC {m.ncc:.2f} vs {g.ncc_mean:.2f})")

        if not body_gone and m.body_aspect < config.BODY_STANDING_ASPECT:
            return V("tombstone",
                     1.0 - m.body_aspect / config.BODY_STANDING_ASPECT,
                     f"body spans {m.body_aspect:.2f}x its own width, standing "
                     f"on one land")
    else:
        # No golden statistics for this package, so every test above and below
        # has nothing to compare against -- correlation, band distances, the
        # segmentation thresholds, all of it is learned at teach. This component
        # is not good and not bad, it is UNINSPECTED, and saying so is the whole
        # point: calling it a defect turns one unprogrammed package into a false
        # call on every instance of it, which is the failure mode section 8 says
        # decides the project. It is reported once at board level as a coverage
        # gap instead of N times as a defect.
        return Verdict(m.refdes, m.package, False, 0.0, "untaught",
                       f"no golden model for {m.package}; component not inspected",
                       testable=False)

    # 3. reversed. Safe here now: a part standing on end was already caught
    #    above, and it was exactly that case which used to flip the 180 degree
    #    template margin and come back as a polarity fault.
    if m.pol_margin < -0.05:
        return V("reversed", 1.0 + abs(m.pol_margin),
                 f"180 deg template wins by {abs(m.pol_margin):.3f}")

    # 4. placement
    off = float(np.hypot(m.dx_um, m.dy_um))
    if off > config.MAX_OFFSET_UM:
        return V("offset", off / config.MAX_OFFSET_UM, f"{off:.0f} um off nominal")
    if abs(m.dtheta_deg) > config.MAX_SKEW_DEG:
        return V("skew", abs(m.dtheta_deg) / config.MAX_SKEW_DEG, f"{m.dtheta_deg:+.1f} deg")

    # 5. joint quality -- only now is the fillet window trustworthy.
    #
    #    Scored as a Mahalanobis distance of the pad's three band fractions from
    #    the golden distribution, not as a z-score on the low band alone. The
    #    bands are correlated at r = 0.5 to 0.85 on good joints, and that is what
    #    makes the joint test work: anything changing how much light reaches the
    #    part -- vignetting, a dim corner, LED drift, a different board finish --
    #    moves all three together along the high-covariance axis and costs almost
    #    no distance, while a starved joint loses the steep face but keeps the
    #    shallow one and moves across that axis, where it costs a lot. A
    #    univariate test cannot separate those two, which is why illumination
    #    non-uniformity is the largest single source of false calls in a rule
    #    layer. It was also the only source of them here: 3 of 3, measured over
    #    1080 good-component inspections.
    if g is not None:
        dist = band_distance(m.bands, g)
        for pad in range(2):
            # One-sided on purpose. Mahalanobis distance is symmetric, so a joint
            # carrying MORE solder than golden sits just as far from the mean as
            # a starved one. Without this guard a heavy fillet gets reported as
            # insufficient solder -- the opposite fault. Grading excess is a
            # separate test that does not exist yet (spec section 5).
            if dist[pad] > config.BAND_MAHALANOBIS_FAIL \
                    and m.bands[pad, 0] < g.band_mean[pad, 0]:
                return V("insufficient", dist[pad] / config.BAND_MAHALANOBIS_FAIL,
                         f"pad {pad} fillet d={dist[pad]:.1f} from golden, "
                         f"low-angle band {m.bands[pad, 0]:.2f} vs "
                         f"{g.band_mean[pad, 0]:.2f}")

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

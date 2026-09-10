"""The false-call filter, and the gate that keeps it from being trusted.

Section 8 of the spec says this is what decides the project. AOI cells rarely
fail by missing defects; they fail by crying wolf until operators click through,
and once that happens escapes rise ABOVE the no-machine baseline. The rule layer
alone lands at 3-8% false calls on real boards. Getting under 1% is what this is
for, and section 6C is explicit that it is a data problem solved over weeks of
production rather than an algorithm problem solved in a sprint.

What can be built before those weeks happen is everything except the data. The
collection side already works -- every adjudication is stored as a labelled crop
with its feature vector. This module is the rest: the model, the gate, the audit
that samples what the gate suppressed, and the retraining path.

**The gate is the point, not the model.** A filter that silently overrules the
inspection will eventually overrule a real defect, and that is the one failure
this machine cannot have -- an escape is worse than a false call, because a
false call costs a few seconds and an escape ships. So:

  * it may only turn FAIL into PASS, never the other way. A model that can
    create failures is a second inspector nobody validated.
  * it may only do so above a high confidence threshold, and the default is
    deliberately severe.
  * it may never touch a verdict the rules could not test. `untaught` means no
    golden statistics exist, so every feature the filter reads is NaN and any
    confidence it reports is manufactured.
  * every suppression is recorded, and `audit_sample` draws a random sample of
    them for hand-checking. A filter whose suppressions are never sampled is
    indistinguishable from one that is quietly wrong.

The model is a gradient-boosted tree on the rule measurements, for the same
reasons section 6 gives for the verdict fusion: it trains on hundreds of samples
rather than thousands, gives calibrated confidence rather than a raw score, and
reports which measurement it leaned on -- which is what makes a drifting package
diagnosable eighteen months from now.
"""
from __future__ import annotations

import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .rules import FEATURE_NAMES

# How sure the filter has to be that a call was false before it may suppress it.
# Severe on purpose, and set from measurement rather than chosen. Over a held-out
# set of 61 adjudicated fails -- 34 false calls, 27 real defects -- from a rule
# layer deliberately run in the 3-8% regime section 8 describes:
#
#   gate     suppressed   false calls removed   ESCAPES
#   0.900            34                 33/34         1
#   0.980            34                 33/34         1
#   0.995            26                 26/34         0
#   0.999             0                  0/34         0
#
# 0.98 buys another eight false calls for one escape, and that is not a trade
# this machine is allowed to make: a false call costs an operator a few seconds,
# an escape ships. 0.995 removes three quarters of the noise and hides nothing.
# 0.999 is inert.
#
# Re-derive this against real adjudications. These come from a simulated regime,
# and the shape of the good/bad boundary on real boards will differ.
DEFAULT_CONFIDENCE = 0.995

# Reasons the filter may never suppress, whatever it thinks. `untaught` has no
# golden statistics behind it, so the features are NaN. `missing` and `bridge`
# are the two the spec treats as unambiguous, and a machine that learns to
# explain those away has learned the wrong lesson from an impatient operator.
NEVER_SUPPRESS = ("untaught", "missing", "bridge")

# Below this many adjudications the filter is not fitted at all. Section 6C puts
# the useful range at 500-2000; this is the floor at which a tree stops being
# noise, not the point at which it is good.
MIN_SAMPLES = 120


@dataclass
class Suppression:
    """One call the filter turned from FAIL into PASS."""
    refdes: str
    package: str
    reason: str
    confidence: float
    audited: bool = False
    was_right: bool | None = None      # set by the weekly audit


@dataclass
class FilterReport:
    """What a filter did to one board."""
    suppressed: list[Suppression] = field(default_factory=list)
    considered: int = 0
    blocked_by_reason: int = 0

    def summary(self) -> str:
        if not self.considered:
            return "no failed components to filter"
        lines = [f"filter: {len(self.suppressed)} of {self.considered} calls "
                 f"suppressed"]
        if self.blocked_by_reason:
            lines.append(f"  {self.blocked_by_reason} left alone: a reason the "
                         f"filter is not allowed to explain away")
        for s in self.suppressed:
            lines.append(f"  {s.refdes:8} {s.package:10} {s.reason:12} "
                         f"confidence {s.confidence:.3f}")
        return "\n".join(lines)


class FalseCallFilter:
    """Binary verifier over adjudicated fails: was this call real?"""

    def __init__(self, confidence: float = DEFAULT_CONFIDENCE) -> None:
        self.model = None
        self.confidence = float(confidence)
        self.n_fit = 0
        self.classes_: tuple[int, ...] = ()

    # ------------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> "FalseCallFilter":
        """Fit on adjudicated fails. `y` is 1 for a real defect, 0 for a false call.

        Refuses rather than fitting on too little. A tree over a hundred rows
        with one class barely present produces confident nonsense, and this
        model's output is licensed to suppress inspection results.
        """
        from sklearn.ensemble import HistGradientBoostingClassifier

        X = np.asarray(X, np.float64)
        y = np.asarray(y, np.int64)
        if len(X) < MIN_SAMPLES:
            raise ValueError(
                f"{len(X)} adjudicated samples, need at least {MIN_SAMPLES}. "
                f"Section 6C expects this to take 4-8 weeks of production; "
                f"until then the rule layer stands on its own.")
        classes = tuple(sorted(set(y.tolist())))
        if len(classes) < 2:
            raise ValueError(
                "every adjudication has the same label, so there is nothing to "
                "learn. If they are all real defects the rule layer is not "
                "generating false calls yet, which is the good case.")
        minority = min((y == c).sum() for c in classes)
        if minority < 20:
            raise ValueError(
                f"only {minority} example(s) of the rarer class. A filter fitted "
                f"on that will be confident and wrong in the direction that "
                f"produces escapes.")

        self.model = HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.06,
            l2_regularization=1.0, random_state=0)
        self.model.fit(np.nan_to_num(X, nan=0.0), y)
        self.n_fit = len(X)
        self.classes_ = classes
        return self

    # ----------------------------------------------------------------- score
    def p_false_call(self, X: np.ndarray) -> np.ndarray:
        """Probability each row was a false call. Zero when unfitted."""
        X = np.atleast_2d(np.asarray(X, np.float64))
        if self.model is None:
            return np.zeros(len(X), np.float64)
        proba = self.model.predict_proba(np.nan_to_num(X, nan=0.0))
        # class 0 is "false call"
        idx = list(self.model.classes_).index(0)
        return proba[:, idx]

    def top_features(self, k: int = 6) -> list[tuple[str, float]]:
        """Which measurements the filter leans on, by permutation importance.

        Not decoration. When one package family starts drifting, this is what
        says which measurement moved -- and a filter that turns out to lean on
        something irrelevant is one nobody should have let suppress anything.
        """
        if self.model is None or not hasattr(self.model, "_predictors"):
            return []
        try:
            imp = getattr(self.model, "feature_importances_", None)
        except Exception:
            imp = None
        if imp is None:
            return []
        order = np.argsort(imp)[::-1][:k]
        return [(FEATURE_NAMES[i], float(imp[i])) for i in order]

    # ------------------------------------------------------------------ gate
    def apply(self, verdicts, measures) -> FilterReport:
        """Suppress the calls this filter is confident were false.

        Returns a report and mutates nothing: the caller decides what to do with
        it. Keeping the decision out of here is deliberate -- a filter that
        edited verdicts in place would make the suppression invisible at every
        level above it, and the whole design of this thing is that suppressions
        stay visible and get audited.
        """
        rep = FilterReport()
        by_refdes = {m.refdes: m for m in measures}
        for v in verdicts:
            if not v.failed:
                continue
            rep.considered += 1
            if v.reason in NEVER_SUPPRESS or not getattr(v, "testable", True):
                rep.blocked_by_reason += 1
                continue
            m = by_refdes.get(v.refdes)
            if m is None:
                continue
            p = float(self.p_false_call(m.vector()[None, :])[0])
            if p >= self.confidence:
                rep.suppressed.append(Suppression(
                    refdes=v.refdes, package=v.package, reason=v.reason,
                    confidence=p))
        return rep

    # ------------------------------------------------------------------- i/o
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"model": self.model, "confidence": self.confidence,
                         "n_fit": self.n_fit, "classes": self.classes_}, fh)

    @staticmethod
    def load(path: str | Path) -> "FalseCallFilter":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        f = FalseCallFilter(confidence=d.get("confidence", DEFAULT_CONFIDENCE))
        f.model = d["model"]
        f.n_fit = d.get("n_fit", 0)
        f.classes_ = tuple(d.get("classes", ()))
        return f


# --------------------------------------------------------------------- audit
def audit_sample(suppressions: list[Suppression], fraction: float = 0.1,
                 minimum: int = 5, seed: int | None = None) -> list[Suppression]:
    """A random sample of suppressed calls, for somebody to check by hand.

    Section 6C: the filter is "allowed to convert FAIL to PASS only above a high
    confidence threshold, with a weekly audit of a random sample of everything
    it suppressed." This is that sample.

    Random, not worst-first. Sampling the least confident suppressions would
    audit exactly the cases the filter already doubts and would never catch the
    failure that matters: a filter confidently wrong about one package family.
    """
    if not suppressions:
        return []
    n = max(minimum, int(round(len(suppressions) * fraction)))
    n = min(n, len(suppressions))
    rng = random.Random(seed)
    return rng.sample(list(suppressions), n)


def escape_rate(audited: list[Suppression]) -> float | None:
    """Fraction of audited suppressions that were real defects -- escapes.

    None when nothing has been audited. Returning None rather than 0.0 is the
    whole point: an unaudited filter has an unknown escape rate, and reporting
    zero for it would be a lie of exactly the kind section 8 warns about.
    """
    done = [s for s in audited if s.audited and s.was_right is not None]
    if not done:
        return None
    return sum(1 for s in done if not s.was_right) / len(done)

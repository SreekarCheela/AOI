"""The false-call filter, and the gate that keeps it from being trusted.

Section 8 says this is what decides the project: AOI cells fail by crying wolf
until operators click through, and once that happens escapes rise ABOVE the
no-machine baseline. So the tests that matter here are not the ones about
accuracy. They are the ones about what the filter is FORBIDDEN to do.

An escape is worse than a false call. A false call costs an operator a few
seconds; an escape ships. Every constraint below exists because a model licensed
to overrule an inspection will eventually overrule a real defect, and none of
these can be relaxed for a better suppression rate.
"""
from __future__ import annotations

import numpy as np
import pytest

from aoi import falsecall, rules, store
from aoi.falsecall import FalseCallFilter, Suppression


N_FEAT = len(rules.FEATURE_NAMES)


def _training_set(n=400, seed=0):
    """Separable-ish adjudications: false calls sit near the threshold, real
    defects sit well past it. Shaped like the real thing without pretending to
    be it."""
    rng = np.random.RandomState(seed)
    X = rng.normal(0.0, 1.0, (n, N_FEAT))
    y = (np.arange(n) % 2).astype(np.int64)          # balanced
    maha = rules.FEATURE_NAMES.index("band_maha")
    X[y == 1, maha] += 6.0                           # real defects: far out
    X[y == 0, maha] += 0.5                           # false calls: marginal
    return X, y


class _V:
    """Minimal stand-in for a Verdict."""
    def __init__(self, refdes, reason, failed=True, package="0402", testable=True):
        self.refdes, self.reason, self.failed = refdes, reason, failed
        self.package, self.testable = package, testable


class _M:
    def __init__(self, refdes, vec):
        self.refdes, self._v = refdes, vec

    def vector(self):
        return self._v


# ------------------------------------------------------------ what it refuses
def test_it_will_not_fit_on_too_little_data():
    """Section 6C puts the useful range at 500-2000 adjudications and expects
    4-8 weeks of production to get there. A tree fitted on a handful is
    confident nonsense, and this model's output is licensed to suppress
    inspection results."""
    X, y = _training_set(40)
    with pytest.raises(ValueError) as e:
        FalseCallFilter().fit(X, y)
    assert "at least" in str(e.value)


def test_it_will_not_fit_when_every_adjudication_agrees():
    X, y = _training_set(400)
    with pytest.raises(ValueError) as e:
        FalseCallFilter().fit(X, np.ones_like(y))
    assert "same label" in str(e.value)


def test_it_will_not_fit_when_one_class_is_nearly_absent():
    """Fitted on five false calls, the filter is confident in the direction that
    produces escapes."""
    X, y = _training_set(400)
    y = y.copy()
    y[y == 0] = 1
    y[:5] = 0
    with pytest.raises(ValueError) as e:
        FalseCallFilter().fit(X, y)
    assert "rarer class" in str(e.value)


def test_an_unfitted_filter_suppresses_nothing():
    """The safe default. A filter that has never been trained must behave as
    though it does not exist, not as though everything is a false call."""
    f = FalseCallFilter()
    assert float(f.p_false_call(np.zeros((1, N_FEAT)))[0]) == 0.0
    rep = f.apply([_V("R1", "insufficient")], [_M("R1", np.zeros(N_FEAT))])
    assert rep.suppressed == []


# ------------------------------------------------------------------- the gate
@pytest.fixture(scope="module")
def fitted():
    X, y = _training_set(400)
    return FalseCallFilter().fit(X, y)


def test_it_only_ever_turns_a_fail_into_a_pass(fitted):
    """It may not create failures. A model that can is a second inspector
    nobody validated, and nothing downstream would know."""
    passing = [_V("R1", "ok", failed=False), _V("R2", "ok", failed=False)]
    measures = [_M("R1", np.zeros(N_FEAT)), _M("R2", np.zeros(N_FEAT))]
    rep = fitted.apply(passing, measures)
    assert rep.considered == 0 and rep.suppressed == []


@pytest.mark.parametrize("reason", falsecall.NEVER_SUPPRESS)
def test_some_reasons_can_never_be_explained_away(fitted, reason):
    """`untaught` has no golden statistics behind it, so every feature the
    filter reads is NaN and any confidence it reports is manufactured.
    `missing` and `bridge` are unambiguous, and a machine that learns to explain
    those away has learned the wrong lesson from an impatient operator."""
    X, _ = _training_set(400)
    false_looking = X[1]                                   # a confident false call
    rep = fitted.apply([_V("R1", reason)], [_M("R1", false_looking)])
    assert rep.suppressed == [], f"{reason} was suppressed"
    assert rep.blocked_by_reason == 1


def test_an_untestable_verdict_is_never_suppressed(fitted):
    X, _ = _training_set(400)
    v = _V("R1", "insufficient", testable=False)
    rep = fitted.apply([v], [_M("R1", X[1])])
    assert rep.suppressed == []


def test_confidence_below_the_gate_suppresses_nothing(fitted):
    """The gate is the whole safety argument. Raising it must monotonically
    reduce what gets suppressed, never leave it unchanged by accident."""
    X, y = _training_set(400, seed=3)
    verdicts = [_V(f"R{i}", "insufficient") for i in range(len(X))]
    measures = [_M(f"R{i}", X[i]) for i in range(len(X))]

    counts = []
    for c in (0.5, 0.9, 0.995, 0.99999):
        fitted.confidence = c
        counts.append(len(fitted.apply(verdicts, measures).suppressed))
    fitted.confidence = falsecall.DEFAULT_CONFIDENCE
    assert counts == sorted(counts, reverse=True), (
        f"suppression count {counts} does not fall as the gate rises")
    assert counts[-1] == 0, "an almost-certain gate still suppressed something"


def test_the_default_gate_is_severe():
    """Measured, not chosen: at 0.98 the filter removed eight more false calls
    and hid one real defect. That trade is not available to this machine."""
    assert falsecall.DEFAULT_CONFIDENCE >= 0.99


def test_applying_the_filter_changes_no_verdict(fitted):
    """It reports; the caller decides. A filter that edited verdicts in place
    would make its own suppressions invisible at every level above it, and the
    entire design is that they stay visible and get audited."""
    X, _ = _training_set(400)
    v = _V("R1", "insufficient")
    fitted.apply([v], [_M("R1", X[1])])
    assert v.failed is True and v.reason == "insufficient"


# ---------------------------------------------------------------- the audit
def test_the_audit_sample_is_random_not_worst_first():
    """Sampling the least confident suppressions would audit exactly the cases
    the filter already doubts, and would never catch the failure that matters:
    a filter confidently wrong about one package family."""
    sup = [Suppression(f"R{i}", "0402", "insufficient", 0.99 + i * 1e-5)
           for i in range(100)]
    a = falsecall.audit_sample(sup, fraction=0.1, seed=1)
    b = falsecall.audit_sample(sup, fraction=0.1, seed=2)
    assert len(a) == 10
    assert [s.refdes for s in a] != [s.refdes for s in b], "sample is not random"
    conf = [s.confidence for s in a]
    assert conf != sorted(conf), "the sample looks ordered by confidence"


def test_the_audit_sample_has_a_floor():
    """Ten percent of six suppressions is nothing, and auditing nothing is how a
    filter goes unchecked for a quarter."""
    sup = [Suppression(f"R{i}", "0402", "insufficient", 0.99) for i in range(6)]
    assert len(falsecall.audit_sample(sup, fraction=0.1, minimum=5)) == 5


def test_an_unaudited_filter_reports_an_unknown_escape_rate_not_zero():
    """The lie section 8 warns about. Zero escapes and no audit look identical
    in a report that does not distinguish them."""
    sup = [Suppression("R1", "0402", "insufficient", 0.99)]
    assert falsecall.escape_rate(sup) is None

    sup[0].audited, sup[0].was_right = True, True
    assert falsecall.escape_rate(sup) == 0.0

    sup.append(Suppression("R2", "0402", "insufficient", 0.99,
                           audited=True, was_right=False))
    assert falsecall.escape_rate(sup) == pytest.approx(0.5)


# ------------------------------------------------------------- the store path
def test_suppressions_are_recorded_and_auditable(tmp_path):
    """A filter whose suppressions are not recorded cannot be audited, and one
    that is not audited is indistinguishable from one quietly creating escapes.
    """
    db = tmp_path / "s.db"
    with store.Store(db) as st:
        st.db.execute("INSERT INTO board (name, serial, inspected_at, seconds,"
                      " n_components, n_flagged, n_untestable)"
                      " VALUES ('B','1','now',1.0,1,1,0)")
        st.db.execute("INSERT INTO verdict (id, board_id, refdes, package,"
                      " failed, testable, reason, score) "
                      "VALUES (1,1,'R1','0402',1,1,'insufficient',0.9)")
        st.db.commit()

        assert st.suppression_metrics()["escape_rate"] is None
        st.record_suppression(1, "insufficient", 0.997)

        m = st.suppression_metrics()
        assert m["suppressed"] == 1 and m["audited"] == 0
        assert m["escape_rate"] is None, "unaudited must not report zero"
        assert len(st.unaudited_suppressions()) == 1

        st.record_audit(1, was_right=False)          # it hid a real defect
        m = st.suppression_metrics()
        assert m["audited"] == 1 and m["escapes_found"] == 1
        assert m["escape_rate"] == pytest.approx(1.0)
        assert st.unaudited_suppressions() == []


def test_the_training_set_drops_rows_with_no_features(tmp_path):
    """A zero vector is a real reading -- the one a grossly defective part
    produces -- so filling a missing one with zeros would teach the filter that
    a missing measurement means a real defect."""
    db = tmp_path / "s.db"
    with store.Store(db) as st:
        st.db.execute("INSERT INTO board (name, serial, inspected_at, seconds,"
                      " n_components, n_flagged, n_untestable)"
                      " VALUES ('B','1','now',1.0,2,2,0)")
        vec = np.zeros(N_FEAT, np.float32).tobytes()
        st.db.execute("INSERT INTO verdict (id, board_id, refdes, package, failed,"
                      " testable, reason, score, features)"
                      " VALUES (1,1,'R1','0402',1,1,'insufficient',0.9,?)", (vec,))
        st.db.execute("INSERT INTO verdict (id, board_id, refdes, package, failed,"
                      " testable, reason, score, features)"
                      " VALUES (2,1,'R2','0402',1,1,'insufficient',0.9,NULL)")
        st.db.commit()
        st.adjudicate(1, "defect")
        st.adjudicate(2, "false_call")

        X, y, pk, ids = st.export_training_set()
        assert len(X) == 1 and list(ids) == [1]
        assert X.shape[1] == N_FEAT


def test_an_empty_store_returns_a_correctly_shaped_training_set(tmp_path):
    with store.Store(tmp_path / "e.db") as st:
        X, y, pk, ids = st.export_training_set()
        assert X.shape == (0, N_FEAT) and len(y) == 0


# ------------------------------------------------------------------- round trip
def test_a_filter_survives_being_saved_and_loaded(fitted, tmp_path):
    p = tmp_path / "fc.pkl"
    fitted.save(p)
    back = FalseCallFilter.load(p)
    X, _ = _training_set(400, seed=9)
    assert np.allclose(back.p_false_call(X[:20]), fitted.p_false_call(X[:20]))
    assert back.confidence == fitted.confidence
    assert back.n_fit == fitted.n_fit

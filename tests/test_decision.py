import numpy as np

from src.calibration import apply_temperature
from src.decision import (
    decide,
    confidence,
    evaluate_with_abstention,
    PASS,
    FAIL,
    ABSTAIN,
)


def test_threshold_half_never_abstains():
    probs = np.array([0.01, 0.5, 0.5, 0.99, 0.5])
    d = decide(probs, threshold=0.5)
    assert (d != ABSTAIN).all()


def test_decisions_match_probability_side():
    probs = np.array([0.1, 0.9])
    d = decide(probs, threshold=0.5)
    assert d[0] == PASS
    assert d[1] == FAIL


def test_uncertain_parts_abstain_at_high_threshold():
    probs = np.array([0.52, 0.95, 0.48, 0.03])
    d = decide(probs, threshold=0.9)
    # The two near 0.5 parts are uncertain and must abstain.
    assert d[0] == ABSTAIN
    assert d[2] == ABSTAIN
    # The confident parts keep their verdict.
    assert d[1] == FAIL
    assert d[3] == PASS


def test_confidence_is_in_valid_range():
    probs = np.linspace(0.0, 1.0, 11)
    c = confidence(probs)
    assert (c >= 0.5 - 1e-9).all()
    assert (c <= 1.0 + 1e-9).all()


def test_abstention_reduces_error_on_accepted_set(trained):
    z = trained["z_te"]
    y = trained["y_te"]
    T = trained["T"]
    p = apply_temperature(z, T)

    base = evaluate_with_abstention(p, y, threshold=0.5)   # accept everything
    strict = evaluate_with_abstention(p, y, threshold=0.9)

    # Raising the threshold must not increase the error rate on accepted parts,
    # and here it strictly reduces it.
    assert strict.accepted_error < base.accepted_error
    # Abstaining means accepting fewer parts.
    assert strict.coverage < base.coverage
    assert strict.n_accepted > 0


def test_accepted_error_is_monotone_nonincreasing(trained):
    z = trained["z_te"]
    y = trained["y_te"]
    T = trained["T"]
    p = apply_temperature(z, T)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    errors = [
        evaluate_with_abstention(p, y, t).accepted_error for t in thresholds
    ]
    # Confidence based abstention should not make the accepted set worse as the
    # bar rises. Allow a tiny tolerance for ties from finite samples.
    for lo, hi in zip(errors[1:], errors[:-1]):
        assert lo <= hi + 1e-9

import numpy as np

from src.calibration import (
    fit_temperature,
    apply_temperature,
    expected_calibration_error,
    sigmoid,
)


def test_temperature_is_positive_and_finite(trained):
    T = trained["T"]
    assert np.isfinite(T)
    assert T > 0.0


def test_temperature_preserves_decision_boundary(trained):
    z = trained["z_te"]
    T = trained["T"]
    p = apply_temperature(z, T)
    # Dividing by a positive T does not change the sign of any logit, so the
    # hard predictions before and after calibration must agree exactly.
    pred_raw = (z >= 0.0).astype(int)
    pred_cal = (p >= 0.5).astype(int)
    assert np.array_equal(pred_raw, pred_cal)


def test_calibration_reduces_ece(trained):
    z = trained["z_te"]
    y = trained["y_te"]
    T = trained["T"]
    ece_uncal = expected_calibration_error(sigmoid(z), y)
    ece_cal = expected_calibration_error(apply_temperature(z, T), y)
    assert ece_cal < ece_uncal


def test_ece_is_zero_for_perfect_calibration():
    # Construct probabilities whose confidence exactly matches accuracy.
    # 100 samples at confidence 0.8 with 80 correct gives bin acc == conf.
    probs = np.concatenate([np.full(80, 0.8), np.full(20, 0.8)])
    labels = np.concatenate([np.ones(80), np.zeros(20)]).astype(int)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece < 1e-9


def test_fit_temperature_recovers_known_scaling():
    # Build logits whose natural temperature is 3.0, then check we recover it.
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.01, 0.99, size=4000)
    labels = (rng.uniform(size=4000) < true_p).astype(int)
    true_logits = np.log(true_p / (1 - true_p))
    scaled = true_logits * 3.0  # the model is overconfident by a factor of 3
    T = fit_temperature(scaled, labels)
    assert 2.3 < T < 3.7

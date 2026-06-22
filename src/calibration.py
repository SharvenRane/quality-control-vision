"""Temperature scaling for calibrated confidence.

A trained classifier is often sharp but miscalibrated: its softmax or sigmoid
scores do not match the true probability of being correct. Temperature scaling
fits a single positive scalar T that divides the logits before the sigmoid. It
leaves the decision boundary untouched (the sign of the logit does not change)
so accuracy is preserved, while the confidences become better calibrated.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _nll(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    p = sigmoid(logits / T)
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    t_min: float = 0.05,
    t_max: float = 20.0,
    n_grid: int = 200,
    refine_iter: int = 50,
) -> float:
    """Fit a scalar temperature on held out logits by minimizing NLL.

    The negative log likelihood as a function of T is one dimensional and
    smooth, so a coarse grid search over log spaced temperatures followed by a
    local ternary refinement finds the optimum reliably. This avoids the
    instability of running a quasi Newton optimizer on a nearly flat objective,
    which can otherwise drive the temperature toward zero.

    Parameters
    ----------
    logits : raw model logits, shape (n,).
    labels : binary labels in {0, 1}, shape (n,).

    Returns
    -------
    T : the fitted temperature, bounded to ``[t_min, t_max]``.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    grid = np.logspace(np.log10(t_min), np.log10(t_max), n_grid)
    losses = [_nll(logits, labels, t) for t in grid]
    best = int(np.argmin(losses))

    # Ternary search in the log domain around the best grid point.
    lo = grid[max(best - 1, 0)]
    hi = grid[min(best + 1, n_grid - 1)]
    a, b = np.log(lo), np.log(hi)
    for _ in range(refine_iter):
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        if _nll(logits, labels, np.exp(m1)) < _nll(logits, labels, np.exp(m2)):
            b = m2
        else:
            a = m1
    T = float(np.exp((a + b) / 2.0))
    return float(np.clip(T, t_min, t_max))


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Return calibrated probabilities of the fail class after dividing by T."""
    return sigmoid(np.asarray(logits, dtype=np.float64) / T)


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Expected calibration error of fail probabilities against binary labels.

    Confidence here is the probability assigned to the predicted class, so it
    always lies in [0.5, 1.0].
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    pred = (probs >= 0.5).astype(int)
    confidence = np.where(pred == 1, probs, 1.0 - probs)
    correct = (pred == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        acc = correct[in_bin].mean()
        conf = confidence[in_bin].mean()
        ece += (count / n) * abs(acc - conf)
    return float(ece)

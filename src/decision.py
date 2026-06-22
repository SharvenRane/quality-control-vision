"""Pass, fail, or abstain decisions with a confidence threshold.

In a real quality line you would rather route an uncertain part to a human than
guess. Given calibrated fail probabilities, the system accepts a decision only
when its confidence clears a threshold; otherwise it abstains. Raising the
threshold trades coverage (how many parts get an automatic verdict) for a lower
error rate on the parts that are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PASS = 0
FAIL = 1
ABSTAIN = -1


def confidence(probs: np.ndarray) -> np.ndarray:
    """Confidence of the predicted class, in [0.5, 1.0]."""
    probs = np.asarray(probs, dtype=np.float64)
    return np.maximum(probs, 1.0 - probs)


def decide(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Map fail probabilities to PASS, FAIL, or ABSTAIN.

    A part is accepted only if its confidence is at least ``threshold``. With a
    threshold of 0.5 nothing abstains and every part gets a verdict.
    """
    probs = np.asarray(probs, dtype=np.float64)
    pred = np.where(probs >= 0.5, FAIL, PASS)
    conf = confidence(probs)
    return np.where(conf >= threshold, pred, ABSTAIN)


@dataclass
class AcceptedMetrics:
    coverage: float          # fraction of parts that received a verdict
    accepted_error: float    # error rate among accepted parts
    n_accepted: int


def evaluate_with_abstention(
    probs: np.ndarray, labels: np.ndarray, threshold: float
) -> AcceptedMetrics:
    """Coverage and error rate on the accepted set at a given threshold."""
    labels = np.asarray(labels)
    decisions = decide(probs, threshold)
    accepted = decisions != ABSTAIN
    n_accepted = int(accepted.sum())
    if n_accepted == 0:
        return AcceptedMetrics(coverage=0.0, accepted_error=0.0, n_accepted=0)
    errors = (decisions[accepted] != labels[accepted]).mean()
    return AcceptedMetrics(
        coverage=n_accepted / len(labels),
        accepted_error=float(errors),
        n_accepted=n_accepted,
    )

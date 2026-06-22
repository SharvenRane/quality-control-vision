"""End to end demo of the quality control pipeline.

Run it with::

    python -m src.demo

It generates synthetic parts, trains the classifier, fits a temperature on a
held out split, and prints test accuracy, the calibration error before and
after scaling, and a coverage versus error table for the abstain option.
"""

from __future__ import annotations

from src.data import make_dataset
from src.model import train_model, predict_logits
from src.calibration import (
    fit_temperature,
    apply_temperature,
    expected_calibration_error,
    sigmoid,
)
from src.decision import evaluate_with_abstention


def main() -> None:
    X_train, y_train = make_dataset(800, seed=1)
    X_cal, y_cal = make_dataset(400, seed=2)
    X_test, y_test = make_dataset(400, seed=3)

    model = train_model(X_train, y_train, epochs=12, seed=0)

    z_test = predict_logits(model, X_test)
    pred = (z_test >= 0.0).astype(int)
    acc = (pred == y_test).mean()
    print(f"test accuracy        : {acc:.3f}  (chance is 0.500)")

    z_cal = predict_logits(model, X_cal)
    T = fit_temperature(z_cal, y_cal)
    print(f"fitted temperature   : {T:.4f}")

    p_uncal = sigmoid(z_test)
    p_cal = apply_temperature(z_test, T)
    ece_uncal = expected_calibration_error(p_uncal, y_test)
    ece_cal = expected_calibration_error(p_cal, y_test)
    print(f"ECE before scaling   : {ece_uncal:.4f}")
    print(f"ECE after scaling    : {ece_cal:.4f}")

    print("\nabstain option on the calibrated test set:")
    print(f"{'threshold':>10}  {'coverage':>9}  {'accepted error':>15}")
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        m = evaluate_with_abstention(p_cal, y_test, thr)
        print(f"{thr:>10.2f}  {m.coverage:>9.3f}  {m.accepted_error:>15.4f}")


if __name__ == "__main__":
    main()

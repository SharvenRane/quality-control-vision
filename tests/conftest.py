"""Shared fixtures.

Training a tiny model and fitting a temperature is cheap, but several tests
reuse the same artifacts. Building them once per module keeps the suite fast
while staying fully on CPU with synthetic data.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import make_dataset
from src.model import train_model, predict_logits
from src.calibration import fit_temperature


@pytest.fixture(scope="module")
def trained():
    Xtr, ytr = make_dataset(800, seed=1)
    Xcal, ycal = make_dataset(400, seed=2)
    Xte, yte = make_dataset(400, seed=3)

    model = train_model(Xtr, ytr, epochs=12, seed=0)

    z_cal = predict_logits(model, Xcal)
    z_te = predict_logits(model, Xte)
    T = fit_temperature(z_cal, ycal)

    return {
        "model": model,
        "z_cal": z_cal, "y_cal": ycal,
        "z_te": z_te, "y_te": yte,
        "T": T,
    }

import numpy as np

from src.model import QCNet, predict_logits


def test_forward_output_shape():
    import torch
    model = QCNet(in_channels=1)
    x = torch.randn(5, 1, 32, 32)
    out = model(x)
    assert out.shape == (5,)


def test_accuracy_beats_chance(trained):
    z = trained["z_te"]
    y = trained["y_te"]
    pred = (z >= 0.0).astype(int)
    acc = (pred == y).mean()
    # Chance on a balanced set is 0.5. Demand a clear margin above it.
    assert acc > 0.7


def test_logits_separate_the_classes(trained):
    z = trained["z_te"]
    y = trained["y_te"]
    # Mean logit for fail parts should sit above the mean for good parts.
    assert z[y == 1].mean() > z[y == 0].mean()


def test_prediction_is_deterministic_in_eval_mode(trained):
    model = trained["model"]
    X = np.random.default_rng(0).standard_normal((8, 1, 32, 32)).astype(np.float32)
    a = predict_logits(model, X)
    b = predict_logits(model, X)
    assert np.array_equal(a, b)

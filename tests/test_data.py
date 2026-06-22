import numpy as np

from src.data import make_dataset, make_part, IMG_SIZE


def test_dataset_shapes_and_dtypes():
    X, y = make_dataset(50, seed=0)
    assert X.shape == (50, 1, IMG_SIZE, IMG_SIZE)
    assert y.shape == (50,)
    assert X.dtype == np.float32
    assert y.dtype == np.int64


def test_pixel_values_in_unit_range():
    X, _ = make_dataset(60, seed=7)
    assert X.min() >= 0.0
    assert X.max() <= 1.0


def test_labels_are_binary_and_both_present():
    _, y = make_dataset(200, seed=3, defect_rate=0.5)
    assert set(np.unique(y)).issubset({0, 1})
    # With a balanced rate over 200 samples both classes must appear.
    assert (y == 0).sum() > 0
    assert (y == 1).sum() > 0


def test_generation_is_deterministic_given_seed():
    a, ya = make_dataset(30, seed=11)
    b, yb = make_dataset(30, seed=11)
    assert np.array_equal(a, b)
    assert np.array_equal(ya, yb)


def test_defect_rate_is_respected_roughly():
    _, y = make_dataset(2000, seed=5, defect_rate=0.3)
    frac = y.mean()
    assert 0.25 < frac < 0.35


def test_defects_change_the_image():
    # A defective part drawn from the same rng state as a clean part must differ.
    rng_clean = np.random.default_rng(42)
    rng_bad = np.random.default_rng(42)
    clean = make_part(rng_clean, defective=False)
    bad = make_part(rng_bad, defective=True)
    assert not np.allclose(clean, bad)

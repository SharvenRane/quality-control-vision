"""Synthetic good versus bad part image generator.

Each part is a small grayscale image of a textured circular workpiece on a
neutral background. Good parts have only mild texture and lighting variation.
Bad parts carry one or more visible defects: scratches, holes, blobs, or
missing chunks. The signal is real and learnable but noisy, so a classifier
has to do actual work rather than memorize a constant.
"""

from __future__ import annotations

import numpy as np


IMG_SIZE = 32


def _disk_mask(size: int, cx: float, cy: float, radius: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def _base_part(rng: np.random.Generator, size: int) -> np.ndarray:
    """A clean textured circular part centered in the frame."""
    img = rng.normal(0.15, 0.03, size=(size, size)).astype(np.float32)
    cx = size / 2 + rng.normal(0, 0.6)
    cy = size / 2 + rng.normal(0, 0.6)
    radius = size * 0.38 * (1.0 + rng.normal(0, 0.03))
    mask = _disk_mask(size, cx, cy, radius)

    # Base body brightness plus smooth shading and fine machined texture.
    body = rng.uniform(0.55, 0.75)
    yy, xx = np.mgrid[0:size, 0:size]
    shading = 0.08 * ((xx - cx) / size) + 0.06 * ((yy - cy) / size)
    texture = rng.normal(0, 0.025, size=(size, size)).astype(np.float32)
    img[mask] = body + shading[mask] + texture[mask]
    return img, (cx, cy, radius, mask)


def _add_scratch(img, rng, cx, cy, radius):
    size = img.shape[0]
    angle = rng.uniform(0, np.pi)
    length = radius * rng.uniform(0.8, 1.6)
    dx, dy = np.cos(angle), np.sin(angle)
    n = int(length * 2)
    t = np.linspace(-length / 2, length / 2, max(n, 4))
    xs = np.clip((cx + t * dx).astype(int), 0, size - 1)
    ys = np.clip((cy + t * dy).astype(int), 0, size - 1)
    val = rng.uniform(0.0, 0.2)
    img[ys, xs] = val
    # Give the scratch a little width.
    img[np.clip(ys + 1, 0, size - 1), xs] = val
    return img


def _add_hole(img, rng, cx, cy, radius):
    size = img.shape[0]
    hx = cx + rng.uniform(-0.6, 0.6) * radius
    hy = cy + rng.uniform(-0.6, 0.6) * radius
    hr = radius * rng.uniform(0.12, 0.28)
    mask = _disk_mask(size, hx, hy, hr)
    img[mask] = rng.uniform(0.0, 0.12)
    return img


def _add_blob(img, rng, cx, cy, radius):
    size = img.shape[0]
    bx = cx + rng.uniform(-0.6, 0.6) * radius
    by = cy + rng.uniform(-0.6, 0.6) * radius
    br = radius * rng.uniform(0.15, 0.30)
    mask = _disk_mask(size, bx, by, br)
    img[mask] = np.clip(img[mask] + rng.uniform(0.3, 0.5), 0, 1)
    return img


_DEFECTS = [_add_scratch, _add_hole, _add_blob]


def make_part(rng: np.random.Generator, defective: bool, size: int = IMG_SIZE):
    img, (cx, cy, radius, _) = _base_part(rng, size)
    if defective:
        n_defects = rng.integers(1, 3)
        for _ in range(n_defects):
            defect = _DEFECTS[rng.integers(0, len(_DEFECTS))]
            img = defect(img, rng, cx, cy, radius)
    img = np.clip(img, 0.0, 1.0)
    return img.astype(np.float32)


def make_dataset(n: int, seed: int = 0, defect_rate: float = 0.5,
                 size: int = IMG_SIZE):
    """Generate ``n`` labeled parts.

    Returns
    -------
    X : float32 array of shape (n, 1, size, size), values in [0, 1].
    y : int64 array of shape (n,). 1 means defective (fail), 0 means good (pass).
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n, 1, size, size), dtype=np.float32)
    y = np.empty((n,), dtype=np.int64)
    for i in range(n):
        defective = rng.random() < defect_rate
        X[i, 0] = make_part(rng, defective, size)
        y[i] = 1 if defective else 0
    return X, y

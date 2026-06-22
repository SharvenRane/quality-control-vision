"""A small convolutional classifier for pass versus fail decisions.

The network is intentionally tiny so it trains in a second or two on CPU. It
outputs a single logit per image. A positive logit leans toward fail
(defective), a negative logit leans toward pass (good).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class QCNet(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 32 -> 16
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 16 -> 8
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),   # 8 -> 1
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = z.flatten(1)
        return self.head(z).squeeze(1)  # shape (batch,)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
) -> QCNet:
    """Train a QCNet on labeled parts and return the fitted model in eval mode."""
    torch.manual_seed(seed)
    model = QCNet(in_channels=X.shape[1]).to(device)
    model.train()

    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()

    model.eval()
    return model


@torch.no_grad()
def predict_logits(model: QCNet, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    Xt = torch.from_numpy(X).to(device)
    return model(Xt).cpu().numpy()

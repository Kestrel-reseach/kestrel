"""Numpy logistic head for real hidden-state features.

The stdlib reference head in verifier.py is fine for tests but too slow for
1536-dimensional Qwen features. This head standardises inputs and uses batch
gradient descent with L2 regularisation.
"""

from __future__ import annotations

from collections.abc import Sequence

from kestrel.verifier import VerifierExample

__all__ = ["NumpyLogisticHead"]


class NumpyLogisticHead:
    """Standardised logistic regression over last-token hidden states."""

    def __init__(
        self, learning_rate: float = 0.1, epochs: int = 500, l2: float = 1e-4
    ) -> None:
        if learning_rate <= 0 or epochs < 1 or l2 < 0:
            raise ValueError(
                "learning_rate and epochs must be positive; l2 non-negative"
            )
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights = None
        self.bias = 0.0
        self._mean = None
        self._std = None

    def fit(self, examples: Sequence[VerifierExample]) -> None:
        """Fit on labelled examples."""
        import numpy as np

        if not examples:
            raise ValueError("cannot fit without examples")
        x = np.asarray(
            [example.features.last_token_hidden for example in examples],
            dtype=np.float64,
        )
        y = np.asarray([1.0 if example.correct else 0.0 for example in examples])
        if x.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("feature vectors must have a consistent length")

        self._mean = x.mean(axis=0)
        self._std = x.std(axis=0) + 1e-8
        x = (x - self._mean) / self._std

        weights = np.zeros(x.shape[1], dtype=np.float64)
        bias = 0.0
        for _ in range(self.epochs):
            probability = self._sigmoid(x @ weights + bias)
            error = probability - y
            weights -= self.learning_rate * (x.T @ error / len(y) + self.l2 * weights)
            bias -= self.learning_rate * float(error.mean())
        self.weights = weights
        self.bias = bias

    def predict(self, features) -> float:
        """Return a correctness probability in [0, 1]."""
        import numpy as np

        if self.weights is None or self._mean is None or self._std is None:
            raise RuntimeError("fit the head before calling predict")
        x = (
            np.asarray(features.last_token_hidden, dtype=np.float64) - self._mean
        ) / self._std
        return float(self._sigmoid(float(x @ self.weights) + self.bias))

    @staticmethod
    def _sigmoid(values):
        import numpy as np

        return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))

"""Common interface for every model in the study.

Every model consumes sequences of shape ``(n, T, F)`` and produces class
probabilities of shape ``(n, C)``. Models that cannot consume sequences pool
them internally, so the experiment driver never needs to know which kind it
is holding - which is what keeps the comparison table fair.
"""

from __future__ import annotations

import abc
import time

import numpy as np


class SequenceClassifier(abc.ABC):
    """Base class for all classifiers in the study."""

    name: str = "base"
    kind: str = "abstract"      # "classical" | "neural"

    def __init__(self, n_classes: int, seed: int = 0, **kwargs) -> None:
        self.n_classes = n_classes
        self.seed = seed
        self.params = kwargs
        self.fit_seconds: float = 0.0

    # ------------------------------------------------------------------ api

    @abc.abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray,
             X_val: np.ndarray | None = None,
             y_val: np.ndarray | None = None) -> None: ...

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "SequenceClassifier":
        """Fit the model.

        ``X_val``/``y_val`` are supplied by the experiment driver and are
        carved out of the *training* fold under the same protocol as the
        outer split. Models must never touch the test fold for early
        stopping or model selection - doing so is exactly the leak this
        codebase exists to measure.
        """
        t0 = time.perf_counter()
        self._fit(X, y, X_val, y_val)
        self.fit_seconds = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    # ------------------------------------------------------------- metadata

    def n_parameters(self) -> int:
        """Trainable parameter count, for the cost column of the results table."""
        return -1

    def model_size_bytes(self) -> int:
        return -1

    def describe(self) -> dict:
        return {
            "model": self.name,
            "kind": self.kind,
            "n_params": self.n_parameters(),
            "size_bytes": self.model_size_bytes(),
            "fit_seconds": round(self.fit_seconds, 3),
            **{f"hp_{k}": v for k, v in self.params.items()},
        }

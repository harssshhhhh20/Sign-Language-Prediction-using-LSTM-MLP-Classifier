"""Model registry."""

from __future__ import annotations

from .base import SequenceClassifier
from .classical import CLASSICAL

CLASSICAL_MODELS = list(CLASSICAL)

# TensorFlow is optional. `neural.py` imports it lazily so that the module
# itself always imports cleanly - which means importing it proves nothing
# about whether TensorFlow is usable. Check for the package itself, or the
# models advertise themselves as available and then fail at fit time.
def _tensorflow_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("tensorflow") is not None


if _tensorflow_available():
    try:
        from .neural import NEURAL
        NEURAL_MODELS = list(NEURAL)
    except Exception:                   # pragma: no cover
        NEURAL = {}
        NEURAL_MODELS = []
else:
    NEURAL = {}
    NEURAL_MODELS = []

MODEL_REGISTRY: dict[str, type[SequenceClassifier]] = {**CLASSICAL, **NEURAL}


def build_model(name: str, n_classes: int, seed: int = 0, **kwargs) -> SequenceClassifier:
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}"
            + ("" if NEURAL_MODELS else
               "  (TensorFlow not importable - neural models unavailable)")
        )
    return MODEL_REGISTRY[name](n_classes=n_classes, seed=seed, **kwargs)


def available_models(kind: str | None = None) -> list[str]:
    if kind == "classical":
        return sorted(CLASSICAL_MODELS)
    if kind == "neural":
        return sorted(NEURAL_MODELS)
    return sorted(MODEL_REGISTRY)

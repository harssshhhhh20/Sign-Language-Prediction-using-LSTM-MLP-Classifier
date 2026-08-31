"""Model registry: classical baselines and neural sequence models."""

from .registry import (
    MODEL_REGISTRY,
    CLASSICAL_MODELS,
    NEURAL_MODELS,
    build_model,
    available_models,
)

__all__ = [
    "MODEL_REGISTRY",
    "CLASSICAL_MODELS",
    "NEURAL_MODELS",
    "build_model",
    "available_models",
]

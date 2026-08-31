"""Deployable model bundle.

A trained model on its own is not deployable. Inference has to reproduce the
exact preprocessing used at training time - the same feature group, the same
normalisation, the same sequence length - and it needs the decision
thresholds that were calibrated on held-out data. Getting any of those wrong
silently degrades live accuracy in a way that is very hard to debug from the
webcam.

So all of it travels together in one directory:

    bundle/
      config.json     feature group, normalisation, window, thresholds, labels
      model.keras     or model.pkl for classical estimators
      calibration.json  held-out performance and the chosen thresholds
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np


@dataclass
class BundleConfig:
    """Everything inference needs to reproduce training-time preprocessing."""

    labels: list[str]
    feature_group: str = "pose_hands"
    normalisation: str = "body_centred"
    sequence_length: int = 30
    dominant_hand: str = "right"
    # Static pathway only: how many of the 69 hand-descriptor dimensions this
    # model was trained on. Older datasets used the first 63 (wrist-relative
    # coordinates) without the fingertip-distance terms, and the live
    # extractor always produces all 69 - so the runtime truncates to match.
    static_feature_dim: int = 69
    model_name: str = "gru"
    model_kind: str = "neural"           # "neural" | "classical"

    # Live decision thresholds, calibrated on held-out data.
    confidence_threshold: float = 0.75
    margin_threshold: float = 0.20       # top1 - top2 probability
    smoothing_window: int = 7            # frames of prediction history
    min_agreement: float = 0.6           # fraction of window agreeing

    # Provenance
    n_signers: int = 1
    n_sequences: int = 0
    trained_at: str = ""
    slr_version: str = ""
    notes: str = ""

    idle_label: str = "__idle__"

    def to_json(self) -> dict:
        return asdict(self)

    @property
    def has_idle_class(self) -> bool:
        return self.idle_label in self.labels

    @property
    def n_classes(self) -> int:
        return len(self.labels)


@dataclass
class Calibration:
    """Held-out performance, and how the thresholds were chosen."""

    accuracy: float = 0.0
    macro_f1: float = 0.0
    per_class_f1: dict = field(default_factory=dict)
    n_eval: int = 0
    protocol: str = ""
    threshold_sweep: list = field(default_factory=list)
    confusable_pairs: list = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


class ModelBundle:
    """A trained model plus everything needed to run it live."""

    def __init__(self, config: BundleConfig, model, calibration: Calibration | None = None):
        self.config = config
        self.model = model
        self.calibration = calibration or Calibration()

    # ---------------------------------------------------------------- save

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        (path / "config.json").write_text(
            json.dumps(self.config.to_json(), indent=2), encoding="utf-8")
        (path / "calibration.json").write_text(
            json.dumps(self.calibration.to_json(), indent=2), encoding="utf-8")

        inner = getattr(self.model, "model", None)
        if self.config.model_kind == "neural" and inner is not None:
            inner.save(path / "model.keras")
        else:
            with open(path / "model.pkl", "wb") as fh:
                pickle.dump(getattr(self.model, "estimator", self.model), fh)
        return path

    # ---------------------------------------------------------------- load

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        path = Path(path)
        cfg_path = path / "config.json"
        if not cfg_path.exists():
            raise SystemExit(
                f"no bundle at {path} (missing config.json). "
                "Train one first:  python run.py train"
            )
        config = BundleConfig(**json.loads(cfg_path.read_text(encoding="utf-8")))

        calib = Calibration()
        cal_path = path / "calibration.json"
        if cal_path.exists():
            calib = Calibration(**json.loads(cal_path.read_text(encoding="utf-8")))

        keras_path = path / "model.keras"
        pkl_path = path / "model.pkl"
        if keras_path.exists():
            import tensorflow as tf
            raw = tf.keras.models.load_model(keras_path)
            model = _KerasAdapter(raw)
        elif pkl_path.exists():
            with open(pkl_path, "rb") as fh:
                model = _SklearnAdapter(pickle.load(fh))
        else:
            raise SystemExit(f"bundle at {path} has no model file")

        return cls(config, model, calib)

    # ------------------------------------------------------------ predict

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def describe(self) -> str:
        c = self.config
        lines = [
            f"Model      : {c.model_name} ({c.model_kind})",
            f"Vocabulary : {c.n_classes} classes"
            + (" (incl. idle)" if c.has_idle_class else ""),
            f"Features   : {c.feature_group}, {c.normalisation}, "
            f"window {c.sequence_length}",
            f"Trained on : {c.n_sequences} sequences from {c.n_signers} signer(s)",
            f"Thresholds : conf>{c.confidence_threshold:.2f} "
            f"margin>{c.margin_threshold:.2f} "
            f"agree>{c.min_agreement:.0%} of {c.smoothing_window}",
        ]
        if self.calibration.n_eval:
            lines.append(
                f"Held-out   : acc {self.calibration.accuracy:.1%}, "
                f"macro-F1 {self.calibration.macro_f1:.1%} "
                f"on {self.calibration.n_eval} sequences "
                f"({self.calibration.protocol})"
            )
        if c.n_signers < 2:
            lines.append(
                "WARNING    : trained on a single signer. Expect noticeably "
                "worse accuracy for anyone else."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Adapters so loaded models present the same interface as training-time ones
# ---------------------------------------------------------------------------


class _KerasAdapter:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(X, dtype=np.float32), verbose=0)


class _SklearnAdapter:
    def __init__(self, estimator):
        self.estimator = estimator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        from .features import pyramid_pool
        if X.ndim == 3:
            # Must match PooledClassical._prepare exactly, or live inference
            # feeds the estimator a differently-shaped vector than it trained
            # on and every prediction is garbage.
            X = pyramid_pool(X)
        return self.estimator.predict_proba(X)

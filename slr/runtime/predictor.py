"""Live sliding-window prediction with smoothing, gating and segmentation.

The gap between "the model scores 98% on test sequences" and "it works in
front of a webcam" is almost entirely in this file. A test sequence is
pre-segmented: it starts when the sign starts and ends when it ends. A live
camera provides an unbroken stream in which most frames are not a sign at
all - reaching for the mouse, scratching your nose, sitting still - and the
classifier has an opinion about every one of them.

Four mechanisms turn a stream into words:

* **Rejection.** Low confidence, or a small gap between the top two classes,
  means no output. A classifier trained on N signs will always name one of
  them; refusing to answer is a capability that has to be added.
* **Smoothing.** A vote over recent windows, not a single frame's argmax.
* **Segmentation.** Motion energy rises when a sign starts and falls when it
  ends. Committing a word on that falling edge is what stops one sign from
  emitting the same word fifteen times.
* **Refractory period.** After committing, a short interval during which the
  same label will not be emitted again.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from ..bundle import ModelBundle
from ..features import normalise_sequence, select_features
from ..pipeline.gate import MotionEnergyGate


@dataclass
class Prediction:
    """The predictor's state after one update."""

    label: str | None            # smoothed current best, None if rejected
    confidence: float
    margin: float
    committed: str | None        # a word was finalised on this update
    energy: float
    route: str                   # "idle" | "static" | "dynamic"
    buffer_fill: float           # 0..1, how full the frame window is
    reason: str = ""             # why nothing was emitted, when nothing was


class LivePredictor:
    """Consumes holistic frames, emits words.

    Parameters
    ----------
    bundle
        Trained model plus its preprocessing config and calibrated thresholds.
    stride
        Run inference every N frames. Landmark extraction dominates the frame
        budget, so a stride of 2-3 costs almost no responsiveness and leaves
        headroom on slower machines.
    commit_on_rest
        Finalise a word when motion falls after a sign. Turn off to commit on
        prediction stability alone.
    refractory_frames
        How many frames must pass before the *same* label may be emitted
        again. Counted in frames rather than seconds so behaviour does not
        change with the camera's frame rate, and so replay tests reproduce
        what happens live.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        stride: int = 3,
        commit_on_rest: bool = True,
        refractory_frames: int = 20,
        stable_commit_frames: int = 12,
        gate: MotionEnergyGate | None = None,
    ) -> None:
        self.bundle = bundle
        self.cfg = bundle.config
        self.stride = max(1, stride)
        self.commit_on_rest = commit_on_rest
        self.refractory_frames = refractory_frames
        self.stable_commit_frames = stable_commit_frames

        self.gate = gate or MotionEnergyGate()
        self.frames: deque[np.ndarray] = deque(maxlen=self.cfg.sequence_length)
        self.history: deque[tuple[str, float, float]] = deque(
            maxlen=self.cfg.smoothing_window)

        self._frame_count = 0
        self._last_commit_label: str | None = None
        self._last_commit_frame = -10**9
        self._stable_count = 0
        self._was_signing = False
        self._last_proba: np.ndarray | None = None
        # Best accepted prediction seen since the current sign started. The
        # commit happens on the falling edge of motion, by which point the
        # route has already gone idle and the live prediction is rejected -
        # so the answer has to be held from while the sign was happening.
        self._pending: tuple[str, float] | None = None

    # ------------------------------------------------------------ helpers

    def reset(self) -> None:
        self.frames.clear()
        self.history.clear()
        self.gate.reset()
        self._stable_count = 0
        self._was_signing = False
        self._last_commit_label = None
        self._pending = None

    def _prepare_window(self) -> np.ndarray:
        """Frame buffer -> exactly the tensor the model was trained on."""
        window = np.stack(self.frames)[None, ...]          # (1, T, 1662)
        window = normalise_sequence(window, self.cfg.normalisation)
        return select_features(window, self.cfg.feature_group,
                               self.cfg.dominant_hand)

    def _smoothed(self) -> tuple[str | None, float, float, float]:
        """Majority label over history, with mean confidence and agreement."""
        if not self.history:
            return None, 0.0, 0.0, 0.0
        counts = Counter(lab for lab, _, _ in self.history)
        label, n = counts.most_common(1)[0]
        agreement = n / len(self.history)
        confs = [c for lab, c, _ in self.history if lab == label]
        margins = [m for lab, _, m in self.history if lab == label]
        return label, float(np.mean(confs)), float(np.mean(margins)), agreement

    # -------------------------------------------------------------- update

    def update(self, keypoints: np.ndarray) -> Prediction:
        """Feed one holistic frame."""
        self.frames.append(np.asarray(keypoints, dtype=np.float32))
        self._frame_count += 1
        fill = len(self.frames) / self.cfg.sequence_length

        if len(self.frames) < self.cfg.sequence_length:
            return Prediction(None, 0.0, 0.0, None, 0.0, "idle", fill,
                              reason="filling buffer")

        window_raw = np.stack(self.frames)
        decision = self.gate.update(window_raw)
        signing = decision.route == "dynamic"

        # Idle costs nothing: skip the classifier entirely when the gate says
        # there is no hand in frame. On a laptop this is most of the runtime.
        if decision.route == "idle":
            self.history.clear()
            self._stable_count = 0
            committed = self._maybe_commit(rest_edge=self._was_signing)
            self._was_signing = False
            return Prediction(None, 0.0, 0.0, committed, decision.energy,
                              decision.route, fill,
                              reason="no hands detected")

        # Only run the classifier every `stride` frames.
        if self._frame_count % self.stride == 0 or self._last_proba is None:
            proba = self.bundle.predict_proba(self._prepare_window())[0]
            self._last_proba = proba
        else:
            proba = self._last_proba

        order = np.argsort(proba)[::-1]
        top_label = self.cfg.labels[order[0]]
        top_conf = float(proba[order[0]])
        margin = float(proba[order[0]] - proba[order[1]]) if len(order) > 1 else 1.0

        self.history.append((top_label, top_conf, margin))
        label, conf, mean_margin, agreement = self._smoothed()

        # ---- rejection -------------------------------------------------
        reason = ""
        accepted = label
        if label == self.cfg.idle_label:
            accepted, reason = None, "idle class"
        elif conf < self.cfg.confidence_threshold:
            accepted, reason = None, f"low confidence ({conf:.2f})"
        elif mean_margin < self.cfg.margin_threshold:
            accepted, reason = None, f"ambiguous (margin {mean_margin:.2f})"
        elif agreement < self.cfg.min_agreement:
            accepted, reason = None, f"unstable ({agreement:.0%} agreement)"

        # ---- hold the best answer seen during this sign -----------------
        if accepted is not None:
            self._stable_count += 1
            if self._pending is None or conf > self._pending[1]:
                self._pending = (accepted, conf)
        else:
            self._stable_count = 0

        # ---- commit ----------------------------------------------------
        # Falling edge of motion is the natural sign boundary. The stable
        # fallback covers continuous signing, where motion never drops.
        rest_edge = self._was_signing and not signing
        stable_edge = self._stable_count >= self.stable_commit_frames
        committed = self._maybe_commit(
            rest_edge=(rest_edge if self.commit_on_rest else False),
            stable_edge=stable_edge,
        )

        self._was_signing = signing

        return Prediction(
            label=accepted, confidence=conf, margin=mean_margin,
            committed=committed, energy=decision.energy, route=decision.route,
            buffer_fill=fill, reason=reason,
        )

    def _maybe_commit(self, rest_edge: bool = False,
                      stable_edge: bool = False) -> str | None:
        """Finalise the pending word if this is a boundary and it is allowed."""
        if self._pending is None or not (rest_edge or stable_edge):
            return None

        label, _ = self._pending
        if (label == self._last_commit_label
                and self._frames_since_commit() < self.refractory_frames):
            self._pending = None
            return None

        self._last_commit_label = label
        self._last_commit_frame = self._frame_count
        self._pending = None

        # Consume the window. The buffer still holds the sign that was just
        # emitted, so anything predicted from it for the next `sequence_length`
        # frames would be that same sign again - which is where duplicate
        # emissions come from. Starting clean is both simpler and correct:
        # the sign is spent.
        self.frames.clear()
        self.history.clear()
        self.gate.reset()
        self._stable_count = 0
        self._was_signing = False
        self._last_proba = None
        return label

    def _frames_since_commit(self) -> int:
        return self._frame_count - self._last_commit_frame

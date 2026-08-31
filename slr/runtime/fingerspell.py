"""Static pathway: recognise fingerspelled letters from a held handshape.

Runs when the motion gate reports ``static`` - a hand present but not
travelling. Letters are committed only after the same letter has been held
steadily, because a hand moving between two letters passes through a
continuous range of shapes and every intermediate shape is *some* letter as
far as the classifier is concerned.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

from .. import schema
from ..bundle import ModelBundle
from ..features import normalise_static, static_features_from_landmarks


class FingerspellRecogniser:
    """Commits a letter once it has been held steady."""

    def __init__(self, bundle: ModelBundle, hold_frames: int = 8,
                 release_frames: int = 4) -> None:
        self.bundle = bundle
        self.cfg = bundle.config
        self.hold_frames = hold_frames
        self.release_frames = release_frames

        self.history: deque[tuple[str, float]] = deque(
            maxlen=self.cfg.smoothing_window)
        self._held: str | None = None
        self._held_count = 0
        self._last_committed: str | None = None
        self._absent = 0

    def reset(self) -> None:
        self.history.clear()
        self._held = None
        self._held_count = 0
        self._last_committed = None
        self._absent = 0

    @staticmethod
    def hand_from_holistic(frame: np.ndarray) -> np.ndarray | None:
        """Pull the more-visible hand's 21 landmarks out of a holistic frame."""
        right = frame[schema.RIGHT_HAND_SLICE]
        left = frame[schema.LEFT_HAND_SLICE]
        for hand in (right, left):
            if np.abs(hand).sum() > 0:
                return hand.reshape(schema.N_HAND, 3)
        return None

    def update(self, frame: np.ndarray) -> str | None:
        """Feed one holistic frame; returns a letter when one is committed."""
        coords = self.hand_from_holistic(np.asarray(frame, dtype=np.float32))
        if coords is None:
            self._absent += 1
            if self._absent >= self.release_frames:
                self.history.clear()
                self._held, self._held_count = None, 0
                self._last_committed = None
            return None
        self._absent = 0

        feats = static_features_from_landmarks(coords)
        feats = normalise_static(feats[None, :])[:, : self.cfg.static_feature_dim]

        proba = self.bundle.predict_proba(feats)[0]
        order = np.argsort(proba)[::-1]
        label = self.cfg.labels[order[0]]
        conf = float(proba[order[0]])
        margin = float(proba[order[0]] - proba[order[1]]) if len(order) > 1 else 1.0

        if conf < self.cfg.confidence_threshold or margin < self.cfg.margin_threshold:
            self.history.clear()
            self._held, self._held_count = None, 0
            return None

        self.history.append((label, conf))
        counts = Counter(l for l, _ in self.history)
        top, n = counts.most_common(1)[0]
        if n / len(self.history) < self.cfg.min_agreement:
            return None

        if top == self._held:
            self._held_count += 1
        else:
            self._held, self._held_count = top, 1

        if self._held_count == self.hold_frames and top != self._last_committed:
            self._last_committed = top
            return top
        return None

    @property
    def current(self) -> str | None:
        return self._held

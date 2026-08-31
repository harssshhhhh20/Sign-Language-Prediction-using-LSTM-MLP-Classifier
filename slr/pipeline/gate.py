"""Motion-energy gate: route a window to the static or dynamic pathway.

A practical sign interface needs both pathways. Word-level models cover a
fixed vocabulary; anything outside it - a name, a place, a technical term -
has to be fingerspelled. Systems in the literature almost always implement
one or the other, so out-of-vocabulary input either fails silently or is
forced into the nearest known class.

The gate is deliberately cheap: fingerspelling holds a handshape roughly
still while word signs traverse the signing space, so mean frame-to-frame
displacement of the hand landmarks separates them without a learned model
and without adding measurable latency.

Hysteresis matters. A single-threshold gate flaps at the boundary - a signer
pausing mid-sign flips the route to static and emits a spurious letter. Two
thresholds plus a dwell requirement means the route only changes when the
evidence persists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import schema


@dataclass
class GateDecision:
    route: str          # "static" | "dynamic" | "idle"
    energy: float
    confidence: float
    changed: bool = False


class MotionEnergyGate:
    """Hysteretic motion gate over a sliding landmark window.

    Parameters
    ----------
    static_threshold, dynamic_threshold
        Lower and upper motion-energy bounds. Between them the current route
        is retained - that band is the hysteresis.
    dwell_frames
        A new route must hold for this many consecutive frames before it is
        adopted.
    idle_threshold
        Below this, no hand is meaningfully present and nothing is emitted.
    """

    def __init__(
        self,
        static_threshold: float = 0.010,
        dynamic_threshold: float = 0.022,
        dwell_frames: int = 5,
        idle_threshold: float = 0.0015,
        window: int = 10,
    ) -> None:
        if static_threshold >= dynamic_threshold:
            raise ValueError("static_threshold must be below dynamic_threshold")
        self.static_threshold = static_threshold
        self.dynamic_threshold = dynamic_threshold
        self.dwell_frames = dwell_frames
        self.idle_threshold = idle_threshold
        self.window = window

        self.route = "idle"
        self._candidate = "idle"
        self._dwell = 0

    # ------------------------------------------------------------ energy ---

    @staticmethod
    def motion_energy(window: np.ndarray) -> float:
        """Mean per-frame hand displacement across a (T, 1662) window.

        Only hand landmarks contribute. Pose drift from the signer shifting
        their weight is not sign motion, and including it makes the gate
        fire on fidgeting.
        """
        window = np.asarray(window, dtype=np.float32)
        if window.ndim != 2 or window.shape[0] < 2:
            return 0.0

        energies = []
        for sl in (schema.LEFT_HAND_SLICE, schema.RIGHT_HAND_SLICE):
            hand = window[:, sl]
            present = np.abs(hand).sum(axis=1) > 0
            if present.sum() < 2:
                continue
            pts = hand[present].reshape(-1, schema.N_HAND, 3)
            disp = np.linalg.norm(np.diff(pts, axis=0), axis=-1)  # (T-1, 21)
            energies.append(float(disp.mean()))

        return float(np.mean(energies)) if energies else 0.0

    # ------------------------------------------------------------ routing ---

    def update(self, window: np.ndarray) -> GateDecision:
        """Feed the newest window; returns the current route."""
        energy = self.motion_energy(window)

        if energy < self.idle_threshold:
            proposed = "idle"
        elif energy < self.static_threshold:
            proposed = "static"
        elif energy > self.dynamic_threshold:
            proposed = "dynamic"
        else:
            proposed = self.route          # hysteresis band: hold

        if proposed == self._candidate:
            self._dwell += 1
        else:
            self._candidate = proposed
            self._dwell = 1

        changed = False
        if self._candidate != self.route and self._dwell >= self.dwell_frames:
            self.route = self._candidate
            changed = True

        # Distance past the relevant threshold, squashed to [0, 1].
        if self.route == "dynamic":
            conf = min(1.0, max(0.0, (energy - self.dynamic_threshold)
                                / max(self.dynamic_threshold, 1e-6)))
        elif self.route == "static":
            conf = min(1.0, max(0.0, (self.static_threshold - energy)
                                / max(self.static_threshold, 1e-6)))
        else:
            conf = 1.0
        return GateDecision(self.route, energy, conf, changed)

    def reset(self) -> None:
        self.route = "idle"
        self._candidate = "idle"
        self._dwell = 0

    # -------------------------------------------------------- calibration ---

    @classmethod
    def calibrate(
        cls,
        dynamic_windows: np.ndarray,
        static_windows: np.ndarray | None = None,
        percentile: float = 10.0,
        **kwargs,
    ) -> "MotionEnergyGate":
        """Set thresholds from data rather than by guessing.

        ``dynamic_threshold`` is placed at the low tail of word-sign motion
        energy so genuine signs are not misrouted; ``static_threshold`` at the
        high tail of fingerspelling energy when static examples are
        available, otherwise at a fixed fraction below.
        """
        dyn = np.asarray([cls.motion_energy(w) for w in dynamic_windows])
        dyn = dyn[dyn > 0]
        if dyn.size == 0:
            raise ValueError("no usable dynamic windows for calibration")
        dynamic_threshold = float(np.percentile(dyn, percentile))

        if static_windows is not None and len(static_windows):
            sta = np.asarray([cls.motion_energy(w) for w in static_windows])
            sta = sta[sta > 0]
            static_threshold = (float(np.percentile(sta, 100 - percentile))
                                if sta.size else dynamic_threshold * 0.45)
        else:
            static_threshold = dynamic_threshold * 0.45

        if static_threshold >= dynamic_threshold:
            static_threshold = dynamic_threshold * 0.45

        return cls(static_threshold=static_threshold,
                   dynamic_threshold=dynamic_threshold, **kwargs)

    def describe(self) -> str:
        return (f"MotionEnergyGate(static<{self.static_threshold:.5f}, "
                f"dynamic>{self.dynamic_threshold:.5f}, "
                f"dwell={self.dwell_frames}, idle<{self.idle_threshold:.5f})")

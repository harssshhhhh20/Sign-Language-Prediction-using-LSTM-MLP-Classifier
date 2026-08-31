"""Feature selection, normalisation, temporal pooling and augmentation.

MediaPipe returns coordinates normalised to the image frame, not to the
signer. That means a raw holistic vector encodes where the person was
standing and how far from the camera at least as strongly as it encodes the
sign. A model trained on raw coordinates from one signer in one session
learns the room. :func:`normalise_sequence` removes that, and the ablation in
:mod:`slr.experiments.run_study` measures how much it matters.
"""

from __future__ import annotations

import numpy as np

from . import schema


# ---------------------------------------------------------------------------
# Feature-group selection
# ---------------------------------------------------------------------------


def select_features(X: np.ndarray, group: str, dominant: str = "right") -> np.ndarray:
    """Slice a feature group out of holistic sequences.

    Parameters
    ----------
    X : (n_sequences, n_frames, 1662) or (n_frames, 1662) or (n, 1662)
    group : a key of :data:`slr.schema.FEATURE_GROUPS`, or ``dominant_hand``
    """
    idx = schema.get_feature_indices(group, dominant)
    return X[..., idx]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _reshape_part(flat: np.ndarray, n_landmarks: int, dims: int) -> np.ndarray:
    """(..., n_landmarks*dims) -> (..., n_landmarks, dims)."""
    return flat.reshape(*flat.shape[:-1], n_landmarks, dims)


def _flatten_part(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(*arr.shape[:-2], -1)


def normalise_sequence(X: np.ndarray, mode: str = "body_centred") -> np.ndarray:
    """Normalise a holistic sequence of shape (..., n_frames, 1662).

    Modes
    -----
    ``raw``
        Passthrough. What the original code used.
    ``wrist_relative``
        Each hand is expressed relative to its own wrist. Removes hand
        position, keeps hand shape. This is what the static pathway already
        did, applied to the dynamic pathway.
    ``body_centred``
        Pose and hands are translated so the mid-shoulder point is the origin
        and scaled by shoulder width. Removes signer position, camera
        distance and body size. Hands additionally keep a wrist-relative
        shape encoding. This is the setting expected to matter most for
        cross-signer transfer.
    """
    if mode == "raw":
        return X.astype(np.float32, copy=False)

    X = np.asarray(X, dtype=np.float32)
    out = X.copy()

    pose = _reshape_part(X[..., schema.POSE_SLICE], schema.N_POSE, schema.POSE_DIMS)
    lh = _reshape_part(X[..., schema.LEFT_HAND_SLICE], schema.N_HAND, schema.HAND_DIMS)
    rh = _reshape_part(X[..., schema.RIGHT_HAND_SLICE], schema.N_HAND, schema.HAND_DIMS)
    face = _reshape_part(X[..., schema.FACE_SLICE], schema.N_FACE, schema.FACE_DIMS)

    # A landmark block that MediaPipe failed to detect is written as all
    # zeros. Normalising zeros produces spurious structure, so track the mask
    # and restore zeros afterwards.
    lh_missing = np.all(lh == 0, axis=(-1, -2), keepdims=True)
    rh_missing = np.all(rh == 0, axis=(-1, -2), keepdims=True)
    face_missing = np.all(face == 0, axis=(-1, -2), keepdims=True)
    pose_missing = np.all(pose == 0, axis=(-1, -2), keepdims=True)

    if mode == "wrist_relative":
        lh = lh - lh[..., schema.HAND_WRIST : schema.HAND_WRIST + 1, :]
        rh = rh - rh[..., schema.HAND_WRIST : schema.HAND_WRIST + 1, :]

    elif mode == "body_centred":
        ls = pose[..., schema.POSE_LEFT_SHOULDER, :3]
        rs = pose[..., schema.POSE_RIGHT_SHOULDER, :3]
        centre = ((ls + rs) / 2.0)[..., None, :]                  # (...,1,3)
        scale = np.linalg.norm(ls - rs, axis=-1)[..., None, None]  # (...,1,1)
        # Guard against a frame where shoulders were not detected.
        scale = np.where(scale < 1e-6, 1.0, scale)

        pose_xyz = (pose[..., :3] - centre) / scale
        pose = np.concatenate([pose_xyz, pose[..., 3:]], axis=-1)  # keep visibility

        lh = (lh - centre) / scale
        rh = (rh - centre) / scale
        face = (face - centre) / scale
    else:
        raise ValueError(f"unknown normalisation mode {mode!r}")

    pose = np.where(pose_missing, 0.0, pose)
    lh = np.where(lh_missing, 0.0, lh)
    rh = np.where(rh_missing, 0.0, rh)
    face = np.where(face_missing, 0.0, face)

    out[..., schema.POSE_SLICE] = _flatten_part(pose)
    out[..., schema.FACE_SLICE] = _flatten_part(face)
    out[..., schema.LEFT_HAND_SLICE] = _flatten_part(lh)
    out[..., schema.RIGHT_HAND_SLICE] = _flatten_part(rh)
    return out


# ---------------------------------------------------------------------------
# Temporal representation
# ---------------------------------------------------------------------------


def add_velocity(X: np.ndarray) -> np.ndarray:
    """Append first-order temporal differences. (n, T, F) -> (n, T, 2F).

    Sign distinctions are frequently carried by movement rather than posture;
    an explicit velocity channel gives non-recurrent models access to it.
    """
    dx = np.diff(X, axis=-2, prepend=X[..., :1, :])
    return np.concatenate([X, dx], axis=-1)


def pyramid_pool(
    X: np.ndarray,
    levels: tuple[int, ...] = (1, 2, 4),
    stats: tuple[str, ...] = ("mean", "std", "min", "max"),
) -> np.ndarray:
    """Temporal pyramid pooling: (n, T, F) -> (n, F * len(stats) * sum(levels)).

    Plain pooling over the whole sequence throws away order, which matters:
    a sign and the same sign performed backwards pool identically. Pooling
    over progressively finer temporal segments - the whole sequence, then
    halves, then quarters - keeps coarse ordering while still producing a
    fixed-length vector.

    This is what lets non-sequential models compete with recurrent ones on
    short, fixed-length sign windows, and it is why this project does not
    require TensorFlow to work well.
    """
    T = X.shape[-2]
    parts = []
    for n_seg in levels:
        bounds = np.linspace(0, T, n_seg + 1).astype(int)
        for a, b in zip(bounds[:-1], bounds[1:]):
            seg = X[..., max(a, 0):max(b, a + 1), :]
            parts.append(pool_sequence(seg, stats))
    return np.concatenate(parts, axis=-1)


def pool_sequence(X: np.ndarray, stats: tuple[str, ...] = ("mean", "std", "min", "max")) -> np.ndarray:
    """Collapse (n, T, F) to (n, F * len(stats)) for non-sequential models.

    Classical baselines cannot consume sequences. Pooling them this way is
    the standard trick and it makes the comparison honest: if a random forest
    on pooled statistics matches an LSTM, the temporal model is not earning
    its parameters on this dataset.
    """
    parts = []
    for s in stats:
        if s == "mean":
            parts.append(X.mean(axis=-2))
        elif s == "std":
            parts.append(X.std(axis=-2))
        elif s == "min":
            parts.append(X.min(axis=-2))
        elif s == "max":
            parts.append(X.max(axis=-2))
        elif s == "first":
            parts.append(X[..., 0, :])
        elif s == "last":
            parts.append(X[..., -1, :])
        elif s == "range":
            parts.append(X.max(axis=-2) - X.min(axis=-2))
        else:
            raise ValueError(f"unknown pooling statistic {s!r}")
    return np.concatenate(parts, axis=-1)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def augment_sequences(
    X: np.ndarray,
    y: np.ndarray,
    n_copies: int = 4,
    seed: int = 0,
    scale_jitter: float = 0.08,
    shift_jitter: float = 0.04,
    rotation_deg: float = 8.0,
    time_warp: float = 0.15,
    dropout_prob: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Geometric + temporal augmentation for very small landmark datasets.

    With ~100 training sequences and >1000 input dimensions, augmentation is
    not a nicety. Each transform corresponds to a real nuisance factor:
    scale (distance from camera), shift (position in frame), rotation (camera
    tilt / body angle), time warp (signing speed), landmark dropout
    (detector failure).

    Must be applied to the full 1662-d holistic vector, before feature
    selection and before normalisation: the geometric transforms address the
    landmark blocks individually, and applying them after normalisation would
    reintroduce exactly the nuisance factors normalisation just removed.

    Returns the originals followed by ``n_copies`` augmented sets.
    """
    if X.shape[-1] != schema.HOLISTIC_SIZE:
        raise ValueError(
            f"augment_sequences expects the full {schema.HOLISTIC_SIZE}-d "
            f"holistic vector, got {X.shape[-1]}-d. Augment first, then call "
            "normalise_sequence() and select_features()."
        )
    rng = np.random.default_rng(seed)
    outs = [X]
    ys = [y]

    for _ in range(n_copies):
        Z = X.copy()
        n = Z.shape[0]

        # Per-sequence isotropic scale and translation.
        s = rng.normal(1.0, scale_jitter, size=(n, 1, 1)).astype(np.float32)
        t = rng.normal(0.0, shift_jitter, size=(n, 1, 1)).astype(np.float32)
        Z = Z * s + t

        # Rotation about the image z-axis, applied to every (x, y) pair.
        theta = np.deg2rad(rng.normal(0.0, rotation_deg, size=n)).astype(np.float32)
        cos, sin = np.cos(theta)[:, None], np.sin(theta)[:, None]
        for sl, n_lm, dims in (
            (schema.POSE_SLICE, schema.N_POSE, schema.POSE_DIMS),
            (schema.FACE_SLICE, schema.N_FACE, schema.FACE_DIMS),
            (schema.LEFT_HAND_SLICE, schema.N_HAND, schema.HAND_DIMS),
            (schema.RIGHT_HAND_SLICE, schema.N_HAND, schema.HAND_DIMS),
        ):
            part = _reshape_part(Z[..., sl], n_lm, dims)
            x0 = part[..., 0].reshape(n, -1)
            y0 = part[..., 1].reshape(n, -1)
            part[..., 0] = (x0 * cos - y0 * sin).reshape(part.shape[:-1])
            part[..., 1] = (x0 * sin + y0 * cos).reshape(part.shape[:-1])
            Z[..., sl] = _flatten_part(part)

        # Temporal warp: resample the sequence at jittered timestamps.
        T = Z.shape[1]
        for i in range(n):
            w = rng.normal(1.0, time_warp)
            src = np.clip(np.linspace(0, (T - 1) * w, T), 0, T - 1)
            lo = np.floor(src).astype(int)
            hi = np.clip(lo + 1, 0, T - 1)
            frac = (src - lo)[:, None].astype(np.float32)
            Z[i] = Z[i][lo] * (1 - frac) + Z[i][hi] * frac

        # Landmark dropout: simulate detector failure on whole blocks.
        if dropout_prob > 0:
            for sl in (schema.LEFT_HAND_SLICE, schema.RIGHT_HAND_SLICE):
                mask = rng.random((n, T)) < dropout_prob
                Z[..., sl][mask] = 0.0

        outs.append(Z.astype(np.float32))
        ys.append(y)

    return np.concatenate(outs, axis=0), np.concatenate(ys, axis=0)


# ---------------------------------------------------------------------------
# Static (fingerspelling) features
# ---------------------------------------------------------------------------


def static_features_from_landmarks(coords: np.ndarray) -> np.ndarray:
    """69-d fingerspelling descriptor from one hand's 21 (x, y, z) landmarks.

    21 wrist-relative coordinates (63) + 5 fingertip-to-palm-centre distances
    + thumb-to-index distance. The distance terms disambiguate handshapes
    that differ by aperture rather than by joint position - notably C vs O.
    """
    coords = np.asarray(coords, dtype=np.float32).reshape(schema.N_HAND, 3)
    wrist = coords[schema.HAND_WRIST]
    rel = (coords - wrist).reshape(-1)

    palm = coords[schema.PALM_LANDMARKS][:, :2].mean(axis=0)
    tip_dists = np.linalg.norm(coords[schema.FINGER_TIPS][:, :2] - palm, axis=1)

    thumb_index = np.linalg.norm(
        coords[schema.HAND_THUMB_TIP][:2] - coords[schema.HAND_INDEX_TIP][:2]
    )
    return np.concatenate([rel, tip_dists, [thumb_index]]).astype(np.float32)


def normalise_static(F: np.ndarray, scale_invariant: bool = True) -> np.ndarray:
    """Scale-normalise 69-d static descriptors.

    Hand size varies between signers by well over 20%. Dividing by the hand's
    own span makes the descriptor size-invariant, which is the cheapest
    available intervention for cross-signer fingerspelling transfer.
    """
    F = np.asarray(F, dtype=np.float32)
    if not scale_invariant:
        return F
    rel = F[..., : schema.N_HAND * 3].reshape(*F.shape[:-1], schema.N_HAND, 3)
    span = np.linalg.norm(rel, axis=-1).max(axis=-1, keepdims=True)
    span = np.where(span < 1e-6, 1.0, span)
    out = F.copy()
    out[..., : schema.N_HAND * 3] = (rel / span[..., None]).reshape(
        *F.shape[:-1], schema.N_HAND * 3
    )
    out[..., schema.N_HAND * 3 :] = F[..., schema.N_HAND * 3 :] / span
    return out

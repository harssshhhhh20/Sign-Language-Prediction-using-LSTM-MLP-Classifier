"""MediaPipe landmark layout and feature-group definitions.

The 1662-dimensional holistic vector used throughout this project is the
concatenation produced by ``extract_holistic_keypoints``:

    [   0 : 132 ]  pose        33 landmarks x (x, y, z, visibility)
    [ 132 :1536 ]  face       468 landmarks x (x, y, z)
    [1536 :1599 ]  left hand   21 landmarks x (x, y, z)
    [1599 :1662 ]  right hand  21 landmarks x (x, y, z)

Face landmarks account for 1404 / 1662 = 84.5% of the vector. Whether that
mass carries sign information or merely signer identity is one of the
questions this codebase is built to answer, so every slice is named and
addressable rather than hard-coded at the call site.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Landmark counts
# --------------------------------------------------------------------------

N_POSE = 33
N_FACE = 468
N_HAND = 21

POSE_DIMS = 4  # x, y, z, visibility
FACE_DIMS = 3
HAND_DIMS = 3

POSE_SIZE = N_POSE * POSE_DIMS   # 132
FACE_SIZE = N_FACE * FACE_DIMS   # 1404
HAND_SIZE = N_HAND * HAND_DIMS   # 63

HOLISTIC_SIZE = POSE_SIZE + FACE_SIZE + 2 * HAND_SIZE  # 1662

# Slice boundaries into the holistic vector
POSE_SLICE = slice(0, POSE_SIZE)
FACE_SLICE = slice(POSE_SIZE, POSE_SIZE + FACE_SIZE)
LEFT_HAND_SLICE = slice(POSE_SIZE + FACE_SIZE, POSE_SIZE + FACE_SIZE + HAND_SIZE)
RIGHT_HAND_SLICE = slice(POSE_SIZE + FACE_SIZE + HAND_SIZE, HOLISTIC_SIZE)

# --------------------------------------------------------------------------
# Named pose landmarks (MediaPipe Pose topology)
# --------------------------------------------------------------------------

POSE_NOSE = 0
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24

# Landmarks above the waist. Signing happens in the upper signing space; legs
# contribute nothing but do contribute variance, so this subset exists as its
# own ablation condition.
UPPER_BODY_POSE = list(range(0, 25))

# Landmarks that carry articulator information without face-mesh identity:
# shoulders, elbows, wrists, hands.
ARTICULATOR_POSE = [
    POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER,
    POSE_LEFT_ELBOW, POSE_RIGHT_ELBOW,
    POSE_LEFT_WRIST, POSE_RIGHT_WRIST,
]

# --------------------------------------------------------------------------
# Hand landmark topology (MediaPipe Hands)
# --------------------------------------------------------------------------

HAND_WRIST = 0
HAND_THUMB_TIP = 4
HAND_INDEX_TIP = 8
HAND_MIDDLE_TIP = 12
HAND_RING_TIP = 16
HAND_PINKY_TIP = 20

FINGER_TIPS = [HAND_THUMB_TIP, HAND_INDEX_TIP, HAND_MIDDLE_TIP,
               HAND_RING_TIP, HAND_PINKY_TIP]

# Metacarpophalangeal joints, used to estimate the palm centre.
PALM_LANDMARKS = [0, 5, 9, 13, 17]


def _pose_indices(landmarks: list[int]) -> np.ndarray:
    """Flat indices into the holistic vector for the given pose landmarks."""
    out = []
    for lm in landmarks:
        base = POSE_SLICE.start + lm * POSE_DIMS
        out.extend(range(base, base + POSE_DIMS))
    return np.asarray(out, dtype=np.int64)


def _slice_indices(s: slice) -> np.ndarray:
    return np.arange(s.start, s.stop, dtype=np.int64)


# --------------------------------------------------------------------------
# Feature groups
# --------------------------------------------------------------------------
#
# Each entry maps a name to the flat indices it selects from the 1662-d
# holistic vector. These are the ablation conditions: the same model trained
# on each group isolates how much of the input actually carries sign content.

def _build_feature_groups() -> dict[str, np.ndarray]:
    pose = _slice_indices(POSE_SLICE)
    face = _slice_indices(FACE_SLICE)
    lh = _slice_indices(LEFT_HAND_SLICE)
    rh = _slice_indices(RIGHT_HAND_SLICE)
    hands = np.concatenate([lh, rh])

    groups: dict[str, np.ndarray] = {
        # Everything - the configuration used by most published MediaPipe+LSTM
        # sign recognition systems, and the baseline we are interrogating.
        "full": np.arange(HOLISTIC_SIZE, dtype=np.int64),

        # Drop the face mesh (84.5% of the vector).
        "pose_hands": np.concatenate([pose, hands]),

        # Upper body only, no face mesh.
        "upper_pose_hands": np.concatenate([_pose_indices(UPPER_BODY_POSE), hands]),

        # Arms + hands: the articulators, nothing else.
        "articulator_hands": np.concatenate([_pose_indices(ARTICULATOR_POSE), hands]),

        # Hands alone.
        "hands": hands,

        # Single hand. Which hand is dominant is data-dependent; resolve with
        # `dominant_hand_group` rather than assuming.
        "left_hand": lh,
        "right_hand": rh,

        # Pose without hands - how much survives on gross body motion alone?
        "pose": pose,

        # Face alone. This is the control condition. A model that scores well
        # above chance here on manual signs is reading signer identity or
        # recording-session artefacts, not the sign.
        "face": face,
    }
    return groups


FEATURE_GROUPS: dict[str, np.ndarray] = _build_feature_groups()

FEATURE_GROUP_SIZES: dict[str, int] = {k: len(v) for k, v in FEATURE_GROUPS.items()}

# Ordered ablation ladder, coarse to fine. Used as the default sweep.
ABLATION_LADDER = [
    "full",
    "pose_hands",
    "upper_pose_hands",
    "articulator_hands",
    "hands",
    "dominant_hand",
    "face",
]


def get_feature_indices(group: str, dominant: str = "right") -> np.ndarray:
    """Resolve a feature-group name to flat indices.

    ``dominant_hand`` is resolved against the supplied dominant side so that
    the ablation ladder stays meaningful across datasets recorded by
    left- and right-handed signers.
    """
    if group == "dominant_hand":
        group = "right_hand" if dominant == "right" else "left_hand"
    if group not in FEATURE_GROUPS:
        raise KeyError(
            f"unknown feature group {group!r}; "
            f"available: {sorted(FEATURE_GROUPS)} (+ 'dominant_hand')"
        )
    return FEATURE_GROUPS[group]


def group_size(group: str, dominant: str = "right") -> int:
    return len(get_feature_indices(group, dominant))


# --------------------------------------------------------------------------
# Static fingerspelling features
# --------------------------------------------------------------------------
#
# The static pathway consumes a single hand rather than a holistic frame.
# 21 wrist-relative coordinates + 5 fingertip-to-palm distances +
# thumb-index distance = 69 features.

STATIC_FEATURE_SIZE = N_HAND * HAND_DIMS + len(FINGER_TIPS) + 1  # 69

# ASL 'J' and 'Z' are traced with motion and cannot be recovered from a single
# frame. A static classifier that includes them is measuring noise, so the
# default alphabet omits them and the paper reports that exclusion explicitly.
STATIC_ALPHABET_FULL = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["SPACE"]
DYNAMIC_LETTERS = ["J", "Z"]
STATIC_ALPHABET = [c for c in STATIC_ALPHABET_FULL if c not in DYNAMIC_LETTERS]

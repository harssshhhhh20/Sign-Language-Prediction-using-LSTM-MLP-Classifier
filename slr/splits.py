"""Evaluation protocols, ordered by how much independence they enforce.

This module is the scientific core of the project. The claim it exists to
test is simple: **the accuracy a landmark-based sign recogniser reports is
mostly determined by how the test set was constructed, not by the model.**

The protocols form a ladder. Each rung removes one source of leakage that the
rung above it permits:

1. ``random``               - shuffle every sequence. Temporally adjacent,
                              near-duplicate recordings land on both sides.
                              This is what most published MediaPipe+LSTM sign
                              recognition systems report. It is the ceiling,
                              and it is not measuring generalisation.
2. ``temporal_holdout``     - hold out the sequences recorded last within
                              each class. Removes adjacency leakage; the
                              signer and session are still shared.
3. ``block_disjoint``       - partition each class's sequences into
                              contiguous blocks and hold out whole blocks.
                              Approximates session independence when only one
                              session exists.
4. ``session_independent``  - hold out entire recording sessions. Removes
                              lighting, camera placement and clothing leakage.
5. ``signer_independent``   - hold out entire signers (leave-one-signer-out).
                              The only protocol that measures what a deployed
                              system actually faces: a person it has never
                              seen.

A protocol the data cannot express raises :class:`ProtocolUnavailable` rather
than silently degrading to a weaker one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .manifest import DatasetManifest


class ProtocolUnavailable(RuntimeError):
    """Raised when the dataset lacks the provenance a protocol requires."""


@dataclass
class Fold:
    """One train/test partition."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    protocol: str
    fold_id: str
    held_out: str = ""     # what this fold holds out (signer id, session id, ...)

    def __post_init__(self) -> None:
        overlap = np.intersect1d(self.train_idx, self.test_idx)
        if overlap.size:
            raise AssertionError(
                f"{self.protocol}/{self.fold_id}: {overlap.size} indices appear "
                "in both train and test - the split is leaking."
            )

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)


# ---------------------------------------------------------------------------
# Protocol implementations
# ---------------------------------------------------------------------------


def random_split(
    manifest: DatasetManifest,
    test_size: float = 0.2,
    seed: int = 0,
    n_repeats: int = 1,
) -> list[Fold]:
    """Protocol 1. Shuffle everything - the leaky baseline.

    Reproduces ``train_test_split(X, y, test_size=0.2, shuffle=True)``. Kept
    deliberately so the paper can quantify the gap between this and the
    honest protocols rather than merely assert one exists.
    """
    y = manifest.label_array()
    n = len(y)
    folds = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed + rep)
        # Stratified: preserve class proportions, as sklearn's does by default
        # in most sign-recognition code that passes `stratify=y`.
        test_idx = []
        for cls in np.unique(y):
            cls_idx = np.flatnonzero(y == cls)
            rng.shuffle(cls_idx)
            k = max(1, int(round(test_size * len(cls_idx))))
            test_idx.append(cls_idx[:k])
        test_idx = np.sort(np.concatenate(test_idx))
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        folds.append(
            Fold(train_idx, test_idx, "random", f"rep{rep}", held_out="(none)")
        )
    return folds


def temporal_holdout_split(
    manifest: DatasetManifest,
    test_size: float = 0.2,
    **_: object,
) -> list[Fold]:
    """Protocol 2. Hold out the last-recorded sequences of each class.

    Within a recording session a signer drifts: they warm up, tire, shift
    position. Sequences recorded next to each other in time are more similar
    than sequences recorded minutes apart. Testing on the tail of each class
    removes the adjacency that ``random`` exploits.
    """
    y = manifest.label_array()
    order = manifest.order_array()
    if len(np.unique(order)) <= 1:
        raise ProtocolUnavailable(
            "temporal_holdout needs per-sequence recording order; the manifest "
            "records a single order value for every sequence."
        )

    test_idx = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        cls_idx = cls_idx[np.argsort(order[cls_idx])]   # chronological
        k = max(1, int(round(test_size * len(cls_idx))))
        test_idx.append(cls_idx[-k:])                    # newest recordings
    test_idx = np.sort(np.concatenate(test_idx))
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    return [Fold(train_idx, test_idx, "temporal_holdout", "tail",
                 held_out="last recordings per class")]


def block_disjoint_split(
    manifest: DatasetManifest,
    n_blocks: int = 5,
    **_: object,
) -> list[Fold]:
    """Protocol 3. Contiguous-block cross-validation.

    Each class's chronologically ordered sequences are cut into ``n_blocks``
    contiguous blocks; each fold holds out one block per class. This is the
    strongest protocol available from a single-session recording, and it is
    a stand-in for session independence - not a replacement for it.
    """
    y = manifest.label_array()
    order = manifest.order_array()
    n = len(y)

    per_class_blocks: dict[int, list[np.ndarray]] = {}
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        cls_idx = cls_idx[np.argsort(order[cls_idx])]
        if len(cls_idx) < n_blocks:
            raise ProtocolUnavailable(
                f"block_disjoint with n_blocks={n_blocks} needs at least "
                f"{n_blocks} sequences per class; class {cls} has {len(cls_idx)}."
            )
        per_class_blocks[cls] = np.array_split(cls_idx, n_blocks)

    folds = []
    for b in range(n_blocks):
        test_idx = np.sort(np.concatenate([per_class_blocks[c][b]
                                           for c in per_class_blocks]))
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        folds.append(Fold(train_idx, test_idx, "block_disjoint", f"block{b}",
                          held_out=f"block {b} of each class"))
    return folds


def session_independent_split(manifest: DatasetManifest, **_: object) -> list[Fold]:
    """Protocol 4. Leave-one-session-out."""
    sessions = manifest.session_array()
    unique = sorted(set(sessions))
    if len(unique) < 2:
        raise ProtocolUnavailable(
            f"session_independent needs >=2 sessions, dataset has {len(unique)}. "
            "Record a second sitting - different day, lighting and camera "
            "placement - and tag it with a new session_id."
        )
    folds = []
    n = len(sessions)
    for ses in unique:
        test_idx = np.flatnonzero(sessions == ses)
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        if len(train_idx) == 0:
            continue
        folds.append(Fold(train_idx, test_idx, "session_independent",
                          f"held_out_{ses}", held_out=ses))
    return folds


def signer_independent_split(manifest: DatasetManifest, **_: object) -> list[Fold]:
    """Protocol 5. Leave-one-signer-out (LOSO).

    The protocol that matters. Every other number in a sign recognition paper
    is an upper bound on this one.
    """
    signers = manifest.signer_array()
    unique = sorted(set(signers))
    if len(unique) < 2:
        raise ProtocolUnavailable(
            f"signer_independent needs >=2 signers, dataset has {len(unique)}. "
            "This is the protocol that supports a generalisation claim; without "
            "a second signer the paper cannot make one. Use "
            "`python -m slr.data.collect_dynamic --signer S02` to record another."
        )
    folds = []
    n = len(signers)
    for sgn in unique:
        test_idx = np.flatnonzero(signers == sgn)
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        if len(train_idx) == 0:
            continue
        folds.append(Fold(train_idx, test_idx, "signer_independent",
                          f"held_out_{sgn}", held_out=sgn))
    return folds


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PROTOCOLS = {
    "random": random_split,
    "temporal_holdout": temporal_holdout_split,
    "block_disjoint": block_disjoint_split,
    "session_independent": session_independent_split,
    "signer_independent": signer_independent_split,
}

# Ordered weakest (most leakage) to strongest (most independence).
PROTOCOL_LADDER = [
    "random",
    "temporal_holdout",
    "block_disjoint",
    "session_independent",
    "signer_independent",
]

PROTOCOL_DESCRIPTIONS = {
    "random": "Stratified random shuffle (leaky baseline; standard in prior work)",
    "temporal_holdout": "Last-recorded sequences per class held out",
    "block_disjoint": "Contiguous-block CV within each class",
    "session_independent": "Leave-one-session-out",
    "signer_independent": "Leave-one-signer-out (LOSO)",
}


def make_folds(manifest: DatasetManifest, protocol: str, **kwargs) -> list[Fold]:
    """Build folds for a named protocol."""
    if protocol not in PROTOCOLS:
        raise KeyError(f"unknown protocol {protocol!r}; available: {PROTOCOL_LADDER}")
    return PROTOCOLS[protocol](manifest, **kwargs)


def available_protocols(manifest: DatasetManifest) -> list[str]:
    """Protocols this dataset can actually express, in ladder order."""
    ok = []
    for name in PROTOCOL_LADDER:
        try:
            make_folds(manifest, name)
            ok.append(name)
        except ProtocolUnavailable:
            continue
        except Exception:
            continue
    return ok


def iter_all(manifest: DatasetManifest,
             protocols: list[str] | None = None,
             **kwargs) -> Iterator[Fold]:
    """Yield every fold of every requested protocol the data supports."""
    for name in (protocols or PROTOCOL_LADDER):
        try:
            yield from make_folds(manifest, name, **kwargs)
        except ProtocolUnavailable:
            continue

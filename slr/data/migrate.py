"""Migrate the original flat data layout into a provenance-carrying dataset.

The original layout stored sequences as ``MP_Data/<label>/<n>/<frame>.npy``
with no record of signer, session or recording order. This script rebuilds it
as one array per sequence plus a manifest.

Recording order is recovered from the sequence folder index: ``collect.py``
iterated ``for sequence in range(num_sequences)`` inside a single webcam
session, so folder number is chronological order. That recovered ordering is
what makes the ``temporal_holdout`` and ``block_disjoint`` protocols possible
on the existing data.

Usage
-----
    python -m slr.data.migrate --source gesture/MP_Data --out data/dynamic \\
        --signer S01 --session S01_SES01
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from .. import schema
from ..manifest import DatasetManifest, SequenceRecord


def _hand_presence(seq: np.ndarray) -> float:
    """Fraction of frames in which at least one hand was detected."""
    lh = seq[:, schema.LEFT_HAND_SLICE]
    rh = seq[:, schema.RIGHT_HAND_SLICE]
    present = (np.abs(lh).sum(axis=1) > 0) | (np.abs(rh).sum(axis=1) > 0)
    return float(present.mean())


def _dominant_hand(seq: np.ndarray) -> str:
    lh = float((np.abs(seq[:, schema.LEFT_HAND_SLICE]).sum(axis=1) > 0).mean())
    rh = float((np.abs(seq[:, schema.RIGHT_HAND_SLICE]).sum(axis=1) > 0).mean())
    return "right" if rh >= lh else "left"


def migrate_dynamic(
    source: Path,
    out_dir: Path,
    signer: str = "S01",
    session: str = "S01_SES01",
    dataset_name: str = "selfcollected-dynamic",
) -> DatasetManifest:
    """Convert ``MP_Data/<label>/<n>/<frame>.npy`` into sequence arrays."""
    source, out_dir = Path(source), Path(out_dir)
    arrays_dir = out_dir / "sequences"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(p.name for p in source.iterdir() if p.is_dir())
    if not labels:
        raise SystemExit(f"no label folders under {source}")

    records: list[SequenceRecord] = []
    for label in labels:
        seq_dirs = sorted(
            (p for p in (source / label).iterdir() if p.is_dir()),
            key=lambda p: int(p.name) if p.name.isdigit() else p.name,
        )
        for seq_dir in seq_dirs:
            frame_files = sorted(
                seq_dir.glob("*.npy"),
                key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
            )
            if not frame_files:
                continue
            seq = np.stack([np.load(f) for f in frame_files]).astype(np.float32)

            order = int(seq_dir.name) if seq_dir.name.isdigit() else len(records)
            seq_id = f"{signer}_{session}_{label}_{order:03d}"
            rel = f"sequences/{seq_id}.npy"
            np.save(out_dir / rel, seq)

            records.append(
                SequenceRecord(
                    sequence_id=seq_id,
                    label=label,
                    signer_id=signer,
                    session_id=session,
                    order=order,
                    n_frames=int(seq.shape[0]),
                    source=rel,
                    hand_presence=_hand_presence(seq),
                    dominant_hand=_dominant_hand(seq),
                    notes="migrated from original MP_Data layout",
                )
            )

    manifest = DatasetManifest(
        name=dataset_name,
        modality="dynamic",
        feature_layout="holistic_1662",
        records=records,
        description=(
            "Word-level signs recorded with MediaPipe Holistic. Migrated from "
            "the original MP_Data layout; signer and session identifiers were "
            "assigned retrospectively because the original collection script "
            "did not record them."
        ),
    )
    manifest.save(out_dir / "manifest.json")
    return manifest


def migrate_static(
    source: Path,
    out_dir: Path,
    signer: str = "S01",
    session: str = "S01_SES01",
    dataset_name: str = "selfcollected-static",
) -> DatasetManifest:
    """Convert the pickled fingerspelling arrays into the manifest layout."""
    source, out_dir = Path(source), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class _NumpyOnly(pickle.Unpickler):
        # The stored pickle contains only numpy arrays; refuse to import
        # anything else rather than executing arbitrary code on load.
        def find_class(self, module, name):
            if module.startswith("numpy"):
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    with open(source, "rb") as fh:
        payload = _NumpyOnly(fh).load()

    X = np.asarray(payload["X"], dtype=np.float32)
    y = np.asarray(payload["y"]).ravel()

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)

    n_classes = int(y.max()) + 1
    alphabet = (
        schema.STATIC_ALPHABET
        if n_classes == len(schema.STATIC_ALPHABET)
        else schema.STATIC_ALPHABET_FULL[:n_classes]
    )

    records = []
    per_class_seen: dict[int, int] = {}
    for i, cls in enumerate(y.tolist()):
        order = per_class_seen.get(cls, 0)
        per_class_seen[cls] = order + 1
        label = alphabet[cls] if cls < len(alphabet) else str(cls)
        records.append(
            SequenceRecord(
                sequence_id=f"{signer}_{session}_{label}_{order:04d}",
                label=label,
                signer_id=signer,
                session_id=session,
                order=order,
                n_frames=1,
                source=f"X.npy#{i}",
                hand_presence=1.0,
                notes="migrated from sign_data.pkl",
            )
        )

    manifest = DatasetManifest(
        name=dataset_name,
        modality="static",
        feature_layout=f"hand_{X.shape[1]}",
        records=records,
        description=(
            "Static fingerspelling handshapes, one frame per sample. "
            f"{X.shape[1]}-dimensional hand descriptors."
        ),
    )
    manifest.save(out_dir / "manifest.json")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="gesture/MP_Data")
    ap.add_argument("--out", default="data/dynamic")
    ap.add_argument("--signer", default="S01")
    ap.add_argument("--session", default="S01_SES01")
    ap.add_argument("--modality", choices=["dynamic", "static"], default="dynamic")
    args = ap.parse_args()

    fn = migrate_dynamic if args.modality == "dynamic" else migrate_static
    manifest = fn(Path(args.source), Path(args.out), args.signer, args.session)
    print(manifest.describe())
    print(f"\nWrote {args.out}/manifest.json")


if __name__ == "__main__":
    main()

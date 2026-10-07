"""Load a manifest-backed dataset into memory."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..manifest import DatasetManifest


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a sequence to ``target_len`` frames.

    Preferred over zero-padding: padding introduces frames that never
    happened and teaches the model that a sign ends with a blank, which is a
    session artefact rather than a property of the sign.
    """
    T = seq.shape[0]
    if T == target_len:
        return seq
    src = np.linspace(0, T - 1, target_len)
    lo = np.floor(src).astype(int)
    hi = np.clip(lo + 1, 0, T - 1)
    frac = (src - lo)[:, None].astype(seq.dtype)
    return seq[lo] * (1 - frac) + seq[hi] * frac


def load_dynamic(
    root: str | Path,
    sequence_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, DatasetManifest]:
    """Load word-level sequences.

    Returns
    -------
    X : (n_sequences, T, 1662) float32
    y : (n_sequences,) int64  - indices into ``manifest.labels``
    """
    root = Path(root)
    manifest = DatasetManifest.load(root / "manifest.json")
    if not manifest.records:
        raise SystemExit(f"manifest at {root} is empty")

    lengths = {r.n_frames for r in manifest.records}
    target = sequence_length or max(lengths)

    seqs = []
    for rec in manifest.records:
        arr = np.load(root / rec.source).astype(np.float32)
        seqs.append(resample_sequence(arr, target))

    X = np.stack(seqs)
    y = manifest.label_array()
    return X, y, manifest


def load_static(root: str | Path) -> tuple[np.ndarray, np.ndarray, DatasetManifest]:
    """Load fingerspelling descriptors: (n, n_features) float32."""
    root = Path(root)
    manifest = DatasetManifest.load(root / "manifest.json")
    X = np.load(root / "X.npy").astype(np.float32)
    y = manifest.label_array()
    return X, y, manifest


def merge_manifests(roots: list[str | Path], out: str | Path) -> DatasetManifest:
    """Combine per-signer dataset directories into one dataset.

    Each signer records into their own directory; this stitches them together
    so leave-one-signer-out becomes expressible. Source paths are rewritten
    relative to ``out``.
    """
    import shutil

    out = Path(out)
    (out / "sequences").mkdir(parents=True, exist_ok=True)

    merged: list = []
    name_parts, modality, layout = [], None, None

    for root in roots:
        root = Path(root)
        m = DatasetManifest.load(root / "manifest.json")
        modality = modality or m.modality
        layout = layout or m.feature_layout
        if m.modality != modality or m.feature_layout != layout:
            raise SystemExit(
                f"cannot merge {root}: modality/layout mismatch "
                f"({m.modality}/{m.feature_layout} vs {modality}/{layout})"
            )
        name_parts.append(m.name)
        for rec in m.records:
            src = root / rec.source
            dst_rel = f"sequences/{rec.sequence_id}.npy"
            shutil.copy2(src, out / dst_rel)
            rec.source = dst_rel
            merged.append(rec)

    manifest = DatasetManifest(
        name="+".join(sorted(set(name_parts))),
        modality=modality or "dynamic",
        feature_layout=layout or "holistic_1662",
        records=merged,
        description=f"Merged from {len(roots)} source datasets.",
    )
    manifest.save(out / "manifest.json")
    return manifest

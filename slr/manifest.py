"""Dataset manifest: provenance for every recorded sequence.

The single most consequential defect in the original codebase was that the
data carried no provenance. Sequences were folders numbered 0..29 with no
record of who signed them, when, or in what order, so the only split that
could be expressed was ``train_test_split(shuffle=True)`` - which places
temporally adjacent, near-duplicate sequences on both sides of the split and
inflates the reported accuracy.

A manifest fixes that. Every sequence carries:

    signer_id     who signed it            -> signer-independent splits
    session_id    which recording sitting  -> session-disjoint splits
    order         index within the session -> temporal holdout splits
    hand_presence fraction of frames where a hand was detected -> QC

Without these fields the evaluation protocols in :mod:`slr.splits` cannot be
constructed, and the accuracy numbers a paper reports are not meaningful.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np


@dataclass
class SequenceRecord:
    """One recorded sign sequence."""

    sequence_id: str          # globally unique, e.g. "S01_SES01_hello_007"
    label: str                # gloss, e.g. "hello"
    signer_id: str            # e.g. "S01"
    session_id: str           # e.g. "S01_SES01"
    order: int                # recording index within the session
    n_frames: int
    source: str               # relative path to the stored array
    hand_presence: float = 1.0     # fraction of frames with >=1 hand detected
    dominant_hand: str = "right"
    notes: str = ""

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class DatasetManifest:
    """A collection of sequence records plus dataset-level metadata."""

    name: str
    modality: str                       # "dynamic" (sequences) | "static" (frames)
    feature_layout: str                 # "holistic_1662" | "hand_69"
    records: list[SequenceRecord] = field(default_factory=list)
    description: str = ""

    # ---------------------------------------------------------------- io ---

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "modality": self.modality,
            "feature_layout": self.feature_layout,
            "description": self.description,
            "records": [r.to_json() for r in self.records],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            name=payload["name"],
            modality=payload["modality"],
            feature_layout=payload["feature_layout"],
            description=payload.get("description", ""),
            records=[SequenceRecord(**r) for r in payload["records"]],
        )

    # ----------------------------------------------------------- queries ---

    def __len__(self) -> int:
        return len(self.records)

    @property
    def labels(self) -> list[str]:
        return sorted({r.label for r in self.records})

    @property
    def signers(self) -> list[str]:
        return sorted({r.signer_id for r in self.records})

    @property
    def sessions(self) -> list[str]:
        return sorted({r.session_id for r in self.records})

    def label_array(self) -> np.ndarray:
        index = {lab: i for i, lab in enumerate(self.labels)}
        return np.asarray([index[r.label] for r in self.records], dtype=np.int64)

    def signer_array(self) -> np.ndarray:
        return np.asarray([r.signer_id for r in self.records])

    def session_array(self) -> np.ndarray:
        return np.asarray([r.session_id for r in self.records])

    def order_array(self) -> np.ndarray:
        return np.asarray([r.order for r in self.records], dtype=np.int64)

    # -------------------------------------------------------- diagnostics ---

    def summary(self) -> dict:
        """Dataset statistics, including the ones reviewers ask about."""
        counts: dict[str, int] = {}
        for r in self.records:
            counts[r.label] = counts.get(r.label, 0) + 1

        per_signer: dict[str, int] = {}
        for r in self.records:
            per_signer[r.signer_id] = per_signer.get(r.signer_id, 0) + 1

        presence = [r.hand_presence for r in self.records]
        return {
            "name": self.name,
            "modality": self.modality,
            "feature_layout": self.feature_layout,
            "n_sequences": len(self.records),
            "n_classes": len(self.labels),
            "n_signers": len(self.signers),
            "n_sessions": len(self.sessions),
            "sequences_per_class": counts,
            "sequences_per_signer": per_signer,
            "mean_hand_presence": float(np.mean(presence)) if presence else 0.0,
            "class_balance_ratio": (
                max(counts.values()) / min(counts.values()) if counts else 1.0
            ),
        }

    def describe(self) -> str:
        s = self.summary()
        lines = [
            f"Dataset      : {s['name']} ({s['modality']}, {s['feature_layout']})",
            f"Sequences    : {s['n_sequences']}",
            f"Classes      : {s['n_classes']}  {self.labels}",
            f"Signers      : {s['n_signers']}  {self.signers}",
            f"Sessions     : {s['n_sessions']}",
            f"Balance      : {s['class_balance_ratio']:.2f}x (max/min per class)",
            f"Hand present : {s['mean_hand_presence']:.1%} of frames",
        ]
        if s["n_signers"] < 2:
            lines.append(
                "WARNING      : single signer - signer-independent evaluation "
                "is not available and no generalisation claim is supportable."
            )
        if s["n_sessions"] < 2:
            lines.append(
                "WARNING      : single session - session-disjoint evaluation "
                "is not available."
            )
        return "\n".join(lines)


def build_capability_report(manifest: DatasetManifest) -> dict[str, bool]:
    """Which evaluation protocols this dataset can actually support.

    Used by the experiment runner to skip - loudly - protocols the data
    cannot express, instead of silently producing a number that looks fine.
    """
    return {
        "random": True,
        "temporal_holdout": len(manifest) > 0 and len(set(manifest.order_array())) > 1,
        "block_disjoint": len(manifest) >= 10,
        "session_independent": len(manifest.sessions) >= 2,
        "signer_independent": len(manifest.signers) >= 2,
    }

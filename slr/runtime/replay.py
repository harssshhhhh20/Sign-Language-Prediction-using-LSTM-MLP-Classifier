"""Replay recorded sequences through the live path - no webcam needed.

Offline accuracy does not tell you whether the live system works, because
the live system has to decide *when* a sign happened, not just what it was.
This replays real recordings as a continuous stream - sign, rest, sign, rest
- and checks what the predictor actually commits.

It catches the failures that only appear live:

  * a sign that never commits because the motion gate never sees a rest edge
  * one sign committing three times
  * words firing during the rest gaps
  * thresholds so tight that nothing is ever emitted

    python run.py replay --model models/word_model --limit 20
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from ..bundle import ModelBundle
from ..data.loaders import load_dynamic
from ..pipeline.gate import MotionEnergyGate
from .predictor import LivePredictor


def make_rest_frames(reference: np.ndarray, n: int, rng) -> np.ndarray:
    """Synthesise a plausible rest gap: a still pose with hands undetected.

    Hands are zeroed, which is what MediaPipe reports when they leave frame
    or drop below the detection threshold, and the body is held with small
    jitter so the gate sees genuinely low motion energy.
    """
    from .. import schema

    base = reference[0].copy()
    base[schema.LEFT_HAND_SLICE] = 0.0
    base[schema.RIGHT_HAND_SLICE] = 0.0
    frames = np.repeat(base[None, :], n, axis=0)
    jitter = rng.normal(0, 0.0008, frames[:, schema.POSE_SLICE].shape)
    frames[:, schema.POSE_SLICE] += jitter.astype(np.float32)
    return frames.astype(np.float32)


def replay(model_path: str, data_root: str, limit: int = 20,
           rest_frames: int = 18, verbose: bool = False, seed: int = 0) -> dict:
    bundle = ModelBundle.load(model_path)
    X, y, manifest = load_dynamic(data_root, sequence_length=None)
    labels = manifest.labels

    print(bundle.describe())
    print(f"\nReplaying {min(limit, len(X))} sequences with {rest_frames}-frame "
          f"rest gaps ...\n")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))[:limit]

    predictor = LivePredictor(bundle, stride=1, gate=MotionEnergyGate())

    expected: list[str] = []
    committed: list[tuple[str, int]] = []      # (label, sequence position)
    commits_in_rest = 0

    for pos, idx in enumerate(order):
        truth = labels[y[idx]]
        expected.append(truth)

        for frame in X[idx]:
            pred = predictor.update(frame)
            if pred.committed:
                committed.append((pred.committed, pos))
                if verbose:
                    mark = "ok " if pred.committed == truth else "MISS"
                    print(f"  [{mark}] expected {truth:12s} "
                          f"got {pred.committed:12s} conf={pred.confidence:.2f}")

        # A sign is committed on the falling edge of motion, which by
        # construction lands in the first frames of the rest gap. So a commit
        # here is only spurious if this sequence already produced one.
        already_committed = any(p == pos for _, p in committed)
        for frame in make_rest_frames(X[idx], rest_frames, rng):
            pred = predictor.update(frame)
            if pred.committed:
                committed.append((pred.committed, pos))
                if already_committed:
                    commits_in_rest += 1
                    if verbose:
                        print(f"  [rest] spurious commit: {pred.committed}")
                else:
                    already_committed = True
                    if verbose:
                        mark = "ok " if pred.committed == truth else "MISS"
                        print(f"  [{mark}] expected {truth:12s} "
                              f"got {pred.committed:12s} (on rest edge)")

    # ---- score ---------------------------------------------------------
    per_seq: dict[int, list[str]] = {}
    for lab, pos in committed:
        per_seq.setdefault(pos, []).append(lab)

    correct = sum(1 for i, truth in enumerate(expected)
                  if per_seq.get(i) and per_seq[i][0] == truth)
    missed = [expected[i] for i in range(len(expected)) if i not in per_seq]
    duplicated = sum(max(0, len(v) - 1) for v in per_seq.values())

    n = len(expected)
    stats = {
        "sequences": n,
        "committed": len(committed),
        "correct_first_commit": correct,
        "commit_accuracy": correct / n if n else 0.0,
        "missed": len(missed),
        "miss_rate": len(missed) / n if n else 0.0,
        "duplicate_commits": duplicated,
        "commits_during_rest": commits_in_rest,
    }

    print("=" * 62)
    print("LIVE-PATH REPLAY")
    print("=" * 62)
    print(f"  sequences replayed      {stats['sequences']}")
    print(f"  correct first commit    {correct}/{n}  "
          f"({stats['commit_accuracy']:.0%})")
    print(f"  never committed         {stats['missed']}  "
          f"({stats['miss_rate']:.0%})")
    print(f"  duplicate commits       {duplicated}")
    print(f"  spurious rest commits   {commits_in_rest}")

    if missed:
        print(f"\n  signs that never fired: "
              f"{', '.join(f'{k} x{v}' for k, v in Counter(missed).items())}")

    print("\nDiagnosis:")
    if stats["miss_rate"] > 0.25:
        print("  - High miss rate. Thresholds are too tight for live use, or")
        print("    the motion gate is not seeing a rest edge. Lower")
        print("    confidence_threshold in the bundle config, or widen the")
        print("    gate with MotionEnergyGate.calibrate() on your data.")
    if duplicated > n * 0.2:
        print("  - Signs are committing more than once. Raise")
        print("    refractory_seconds in LivePredictor.")
    if commits_in_rest > 0:
        print("  - Words fire while not signing. Record the __idle__ class")
        print("    (python run.py collect --signer S01) and retrain.")
    if (stats["miss_rate"] <= 0.25 and duplicated <= n * 0.2
            and commits_in_rest == 0 and stats["commit_accuracy"] >= 0.7):
        print("  - Live path looks healthy.")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/word_model")
    ap.add_argument("--data", default="data/dynamic")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--rest-frames", type=int, default=18)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    replay(args.model, args.data, args.limit, args.rest_frames, args.verbose)


if __name__ == "__main__":
    main()

"""Record word-level sign sequences with full provenance.

Replaces the original ``collect.py``. The differences are the ones that
decide whether the resulting dataset can support a paper:

* **Signer and session are mandatory.** Without them, leave-one-signer-out
  and leave-one-session-out cannot be constructed, and the strongest claim
  a paper can make is unavailable.
* **Interleaved class order.** The original recorded all 30 repetitions of
  ``hello``, then all 30 of ``my``, and so on. Consecutive repetitions of one
  sign share lighting, fatigue and body position, so a random split places
  near-duplicates on both sides of the test boundary. Interleaving by round
  breaks that correlation at the source.
* **Quality gating.** A sequence in which MediaPipe lost the hands is
  rejected and re-recorded rather than silently entering the dataset as
  zeros. In the original data the left hand was absent in 75% of frames.
* **Resumable.** Recording 20 signs x 30 repetitions is not a single sitting.

Usage
-----
    python -m slr.data.collect_dynamic --signer S02 --session S02_SES01 \\
        --vocab configs/vocabulary.txt --repetitions 30 --out data/raw/S02_SES01
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .. import schema
from ..manifest import DatasetManifest, SequenceRecord

# A live camera spends most of its time showing something that is not a sign.
# Without a negative class the model must assign one of its known labels to
# every window, so it fires words at someone scratching their nose. Recording
# explicit non-signing footage is the single cheapest fix for that.
IDLE_LABEL = "__idle__"

IDLE_PROMPTS = [
    "sit still, arms down",
    "look around, shift in your seat",
    "scratch your face / adjust your hair",
    "reach for something off-camera",
    "talk without signing",
    "rest hands in your lap",
    "stretch, then relax",
    "drink from a cup",
]


def extract_holistic_keypoints(results) -> np.ndarray:
    """MediaPipe Holistic results -> the 1662-d vector defined in slr.schema."""
    pose = (
        np.array([[r.x, r.y, r.z, r.visibility]
                  for r in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(schema.POSE_SIZE)
    )
    face = (
        np.array([[r.x, r.y, r.z]
                  for r in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks else np.zeros(schema.FACE_SIZE)
    )
    lh = (
        np.array([[r.x, r.y, r.z]
                  for r in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(schema.HAND_SIZE)
    )
    rh = (
        np.array([[r.x, r.y, r.z]
                  for r in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(schema.HAND_SIZE)
    )
    return np.concatenate([pose, face, lh, rh]).astype(np.float32)


def _hand_presence(seq: np.ndarray) -> float:
    lh = seq[:, schema.LEFT_HAND_SLICE]
    rh = seq[:, schema.RIGHT_HAND_SLICE]
    return float(((np.abs(lh).sum(1) > 0) | (np.abs(rh).sum(1) > 0)).mean())


def _build_schedule(vocab: list[str], repetitions: int, seed: int) -> list[tuple[int, str]]:
    """Interleaved recording schedule: round-robin with per-round shuffling.

    Returns [(round_index, label), ...]. Every label appears once per round,
    so repetitions of a sign are spread across the whole session instead of
    being recorded back to back.
    """
    rng = np.random.default_rng(seed)
    schedule = []
    for rep in range(repetitions):
        order = list(vocab)
        rng.shuffle(order)
        schedule.extend((rep, label) for label in order)
    return schedule


def collect(
    signer: str,
    session: str,
    vocab: list[str],
    out_dir: Path,
    repetitions: int = 30,
    sequence_length: int = 30,
    min_hand_presence: float = 0.7,
    countdown: float = 1.5,
    camera: int = 0,
    seed: int = 0,
) -> DatasetManifest:
    import cv2
    import mediapipe as mp

    out_dir = Path(out_dir)
    (out_dir / "sequences").mkdir(parents=True, exist_ok=True)

    # ---- resume ---------------------------------------------------------
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = DatasetManifest.load(manifest_path)
        done = {(r.label, r.order) for r in manifest.records}
        print(f"Resuming: {len(manifest.records)} sequences already recorded.")
    else:
        manifest = DatasetManifest(
            name=f"{signer}-{session}",
            modality="dynamic",
            feature_layout="holistic_1662",
            description=f"Word-level signs. Signer {signer}, session {session}.",
        )
        done = set()

    schedule = [(rep, lab) for rep, lab in _build_schedule(vocab, repetitions, seed)
                if (lab, rep) not in done]
    if not schedule:
        print("Nothing left to record for this session.")
        return manifest

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {camera}")

    rejected = 0
    try:
        with mp_holistic.Holistic(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as holistic:
            i = 0
            while i < len(schedule):
                rep, label = schedule[i]

                is_idle = label == IDLE_LABEL

                # ---- countdown -----------------------------------------
                t_end = time.time() + countdown
                while time.time() < t_end:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    remaining = t_end - time.time()
                    prompt = "NOT SIGNING" if is_idle else label.upper()
                    cv2.putText(frame, prompt, (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                                (60, 170, 250) if is_idle else (0, 220, 0), 3)
                    if is_idle:
                        cv2.putText(frame, IDLE_PROMPTS[rep % len(IDLE_PROMPTS)],
                                    (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                    (60, 170, 250), 2)
                    cv2.putText(frame, f"starts in {remaining:.1f}s", (20, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(frame,
                                f"round {rep + 1}/{repetitions}  "
                                f"{i + 1}/{len(schedule)} remaining",
                                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (200, 200, 200), 2)
                    cv2.imshow("collect", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise KeyboardInterrupt

                # ---- record --------------------------------------------
                frames = []
                while len(frames) < sequence_length:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    results = holistic.process(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
                    frames.append(extract_holistic_keypoints(results))

                    mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                              mp_holistic.POSE_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                                              mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                                              mp_holistic.HAND_CONNECTIONS)
                    cv2.putText(frame, f"REC {label}  {len(frames)}/{sequence_length}",
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 0, 255), 3)
                    cv2.imshow("collect", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise KeyboardInterrupt

                seq = np.stack(frames)
                presence = _hand_presence(seq)

                # ---- quality gate --------------------------------------
                # Idle sequences are *supposed* to lack hands, so the hand
                # presence gate does not apply to them.
                if not is_idle and presence < min_hand_presence:
                    rejected += 1
                    print(f"  REJECT {label} round {rep}: hands visible in "
                          f"{presence:.0%} of frames (need {min_hand_presence:.0%}) "
                          "- re-recording")
                    continue

                seq_id = f"{signer}_{session}_{label}_{rep:03d}"
                rel = f"sequences/{seq_id}.npy"
                np.save(out_dir / rel, seq)
                manifest.records.append(SequenceRecord(
                    sequence_id=seq_id,
                    label=label,
                    signer_id=signer,
                    session_id=session,
                    order=len(manifest.records),   # true chronological order
                    n_frames=int(seq.shape[0]),
                    source=rel,
                    hand_presence=presence,
                ))
                manifest.save(manifest_path)       # checkpoint every sequence
                print(f"  [{len(manifest.records):4d}] {label:12s} round {rep:2d}  "
                      f"hands {presence:.0%}")
                i += 1
    except KeyboardInterrupt:
        print("\nInterrupted - progress saved, rerun the same command to resume.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    manifest.save(manifest_path)
    print(f"\n{manifest.describe()}")
    if rejected:
        print(f"Rejected and re-recorded {rejected} sequences on quality grounds.")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signer", required=True,
                    help="stable pseudonymous id, e.g. S02. Never a real name.")
    ap.add_argument("--session", default=None,
                    help="defaults to <signer>_SES01")
    ap.add_argument("--vocab", default="configs/vocabulary.txt")
    ap.add_argument("--out", default=None, help="defaults to data/raw/<session>")
    ap.add_argument("--repetitions", type=int, default=30)
    ap.add_argument("--sequence-length", type=int, default=30)
    ap.add_argument("--min-hand-presence", type=float, default=0.7)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="record only the first N signs of the vocabulary")
    ap.add_argument("--no-idle", action="store_true",
                    help="skip the non-signing negative class (not recommended)")
    args = ap.parse_args()

    session = args.session or f"{args.signer}_SES01"
    out = Path(args.out or f"data/raw/{session}")

    vocab_path = Path(args.vocab)
    if not vocab_path.exists():
        raise SystemExit(f"vocabulary file not found: {vocab_path}")
    vocab = [ln.strip() for ln in vocab_path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    if args.limit:
        vocab = vocab[: args.limit]
    if not args.no_idle and IDLE_LABEL not in vocab:
        vocab.append(IDLE_LABEL)

    print(f"Signer {args.signer} | session {session} | {len(vocab)} signs "
          f"x {args.repetitions} repetitions = {len(vocab) * args.repetitions} sequences")
    print("Press q at any time to stop; progress is saved after every sequence.\n")

    collect(args.signer, session, vocab, out,
            repetitions=args.repetitions,
            sequence_length=args.sequence_length,
            min_hand_presence=args.min_hand_presence,
            camera=args.camera, seed=args.seed)


if __name__ == "__main__":
    main()

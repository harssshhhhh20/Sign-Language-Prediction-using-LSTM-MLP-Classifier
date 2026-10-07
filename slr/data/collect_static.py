"""Record static fingerspelling handshapes with provenance.

Replaces the collection half of the original ``sign_live.py``. Same
provenance requirements as the dynamic collector, plus two corrections:

* **J and Z are excluded by default.** Both are traced with motion in ASL
  and are not recoverable from a single frame. The original alphabet list
  included them, so ~7% of the training signal was unlearnable noise. Pass
  ``--include-dynamic-letters`` to record them anyway for the dynamic
  pathway.
* **Samples are captured across poses, not from one held position.** Holding
  a handshape still and capturing 150 frames yields 150 near-identical rows,
  which inflates the dataset size without adding information. The collector
  enforces a minimum inter-sample movement so captures are actually distinct.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import schema
from ..features import static_features_from_landmarks
from ..manifest import DatasetManifest, SequenceRecord


def collect_static(
    signer: str,
    session: str,
    out_dir: Path,
    letters: list[str],
    samples_per_letter: int = 60,
    min_movement: float = 0.012,
    camera: int = 0,
) -> DatasetManifest:
    import cv2
    import mediapipe as mp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    X: list[np.ndarray] = []
    labels: list[str] = []

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {camera}")

    try:
        with mp_hands.Hands(max_num_hands=1,
                            min_detection_confidence=0.6,
                            min_tracking_confidence=0.6) as hands:
            for letter in letters:
                captured = 0
                last: np.ndarray | None = None
                print(f"Recording {letter} - hold the shape, move your hand "
                      "slightly between captures. SPACE to capture, n to skip.")
                while captured < samples_per_letter:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    feats = None
                    if res.multi_hand_landmarks:
                        hand = res.multi_hand_landmarks[0]
                        mp_draw.draw_landmarks(frame, hand,
                                               mp_hands.HAND_CONNECTIONS)
                        coords = np.array([[p.x, p.y, p.z]
                                           for p in hand.landmark], dtype=np.float32)
                        feats = static_features_from_landmarks(coords)

                    moved = (last is None or feats is None
                             or float(np.linalg.norm(feats - last)) > min_movement)
                    colour = (0, 220, 0) if (feats is not None and moved) else (0, 0, 255)

                    cv2.putText(frame, f"{letter}  {captured}/{samples_per_letter}",
                                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, 3)
                    cv2.putText(frame,
                                "SPACE capture | n skip letter | q quit",
                                (20, frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    if feats is not None and not moved:
                        cv2.putText(frame, "move your hand - too similar to last",
                                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                    cv2.imshow("collect-static", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        raise KeyboardInterrupt
                    if key == ord("n"):
                        break
                    if key == ord(" ") and feats is not None and moved:
                        X.append(feats)
                        labels.append(letter)
                        last = feats
                        captured += 1
    except KeyboardInterrupt:
        print("\nInterrupted - saving what was captured.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not X:
        raise SystemExit("no samples captured")

    Xa = np.stack(X).astype(np.float32)
    np.save(out_dir / "X.npy", Xa)

    records, seen = [], {}
    for i, lab in enumerate(labels):
        order = seen.get(lab, 0)
        seen[lab] = order + 1
        records.append(SequenceRecord(
            sequence_id=f"{signer}_{session}_{lab}_{order:04d}",
            label=lab, signer_id=signer, session_id=session, order=i,
            n_frames=1, source=f"X.npy#{i}",
        ))

    manifest = DatasetManifest(
        name=f"{signer}-{session}-static",
        modality="static",
        feature_layout=f"hand_{Xa.shape[1]}",
        records=records,
        description=(
            f"Static fingerspelling handshapes, signer {signer}, session "
            f"{session}. Dynamic letters {schema.DYNAMIC_LETTERS} "
            f"{'included' if any(l in labels for l in schema.DYNAMIC_LETTERS) else 'excluded'}."
        ),
    )
    manifest.save(out_dir / "manifest.json")
    np.save(out_dir / "y.npy", manifest.label_array())
    print("\n" + manifest.describe())
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signer", required=True)
    ap.add_argument("--session", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--samples-per-letter", type=int, default=60)
    ap.add_argument("--include-dynamic-letters", action="store_true",
                    help="also record J and Z (not recoverable from one frame)")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    session = args.session or f"{args.signer}_SES01"
    out = Path(args.out or f"data/raw_static/{session}")
    letters = (schema.STATIC_ALPHABET_FULL if args.include_dynamic_letters
               else schema.STATIC_ALPHABET)

    collect_static(args.signer, session, out, letters,
                   samples_per_letter=args.samples_per_letter,
                   camera=args.camera)


if __name__ == "__main__":
    main()

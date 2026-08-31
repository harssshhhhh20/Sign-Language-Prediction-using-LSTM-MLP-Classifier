"""Live sign recognition application.

Webcam -> MediaPipe landmarks -> dual-pathway recognition -> text.

    python -m slr.runtime.app --model models/word_model

Keys
----
    q / Esc   quit
    space     clear the sentence
    u         undo the last word
    s         toggle speech output
    d         toggle the debug overlay
    f         toggle fingerspelling pathway
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np

from ..bundle import ModelBundle
from ..data.collect_dynamic import extract_holistic_keypoints
from ..pipeline.gate import MotionEnergyGate
from .fingerspell import FingerspellRecogniser
from .predictor import LivePredictor
from .sentence import SentenceBuilder

# Palette (BGR).
BG = (24, 24, 28)
FG = (235, 235, 235)
DIM = (140, 140, 148)
ACCENT = (120, 220, 120)
WARN = (80, 170, 250)
BAD = (90, 90, 235)

PANEL_W = 460


class _Speech:
    """Optional text-to-speech. Silently inert when pyttsx3 is absent."""

    def __init__(self, enabled: bool = False):
        self.engine = None
        self.enabled = enabled
        if enabled:
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 165)
            except Exception:
                self.engine = None
                self.enabled = False

    def say(self, text: str) -> None:
        if self.enabled and self.engine and text:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception:
                pass


def _draw_panel(cv2, height: int, state: dict) -> np.ndarray:
    panel = np.full((height, PANEL_W, 3), BG, dtype=np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX

    def text(s, xy, scale=0.6, color=FG, weight=1):
        cv2.putText(panel, s, xy, f, scale, color, weight, cv2.LINE_AA)

    y = 42
    text("Sign Language Translator", (24, y), 0.78, FG, 2)
    y += 26
    text(f"{state['n_classes']} signs  |  {state['model']}", (24, y), 0.5, DIM)

    # ---- routing state -------------------------------------------------
    y += 44
    route = state["route"]
    route_color = {"dynamic": ACCENT, "static": WARN, "idle": DIM}[route]
    cv2.circle(panel, (32, y - 5), 7, route_color, -1)
    text({"dynamic": "signing", "static": "fingerspelling",
          "idle": "waiting"}[route], (50, y), 0.62, route_color)

    # ---- current prediction --------------------------------------------
    y += 46
    text("CURRENT", (24, y), 0.44, DIM)
    y += 34
    if state["label"]:
        text(state["label"].upper(), (24, y), 1.0, ACCENT, 2)
    else:
        text(state["reason"] or "-", (24, y), 0.55, DIM)

    # confidence bar
    y += 26
    bar_w, bar_h = PANEL_W - 48, 8
    cv2.rectangle(panel, (24, y), (24 + bar_w, y + bar_h), (54, 54, 60), -1)
    fill = int(bar_w * max(0.0, min(1.0, state["confidence"])))
    bar_color = ACCENT if state["confidence"] >= state["threshold"] else DIM
    if fill:
        cv2.rectangle(panel, (24, y), (24 + fill, y + bar_h), bar_color, -1)
    thresh_x = 24 + int(bar_w * state["threshold"])
    cv2.line(panel, (thresh_x, y - 3), (thresh_x, y + bar_h + 3), WARN, 1)
    y += 26
    text(f"confidence {state['confidence']:.2f}   margin {state['margin']:.2f}",
         (24, y), 0.44, DIM)

    # ---- sentence -------------------------------------------------------
    y += 48
    text("TEXT", (24, y), 0.44, DIM)
    y += 30
    for line in _wrap(state["text"] or "-", 30)[:4]:
        text(line, (24, y), 0.66, FG)
        y += 30

    y += 18
    text("GLOSS", (24, y), 0.44, DIM)
    y += 26
    for line in _wrap(state["gloss"] or "-", 38)[:3]:
        text(line, (24, y), 0.5, DIM)
        y += 24

    if state["pending"]:
        y += 10
        text(f"spelling: {state['pending']}_", (24, y), 0.6, WARN)

    # ---- debug ----------------------------------------------------------
    if state["debug"]:
        y += 46
        text("DEBUG", (24, y), 0.44, DIM)
        for k in ("energy", "fps", "buffer", "committed_total"):
            y += 24
            text(f"{k:16s} {state[k]}", (24, y), 0.46, DIM)

    # ---- footer ---------------------------------------------------------
    fy = height - 58
    text("q quit   space clear   u undo", (24, fy), 0.44, DIM)
    text(f"s speech {'on' if state['speech'] else 'off'}   "
         f"d debug   f spell {'on' if state['spelling'] else 'off'}",
         (24, fy + 22), 0.44, DIM)
    return panel


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = f"{cur} {w}".strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def run(
    model_path: str = "models/word_model",
    static_model_path: str | None = None,
    camera: int = 0,
    stride: int = 3,
    speech: bool = False,
    mirror: bool = True,
    show_landmarks: bool = True,
) -> None:
    import cv2
    import mediapipe as mp

    bundle = ModelBundle.load(model_path)
    print(bundle.describe())
    print()

    speller = None
    if static_model_path and Path(static_model_path).exists():
        try:
            static_bundle = ModelBundle.load(static_model_path)
            speller = FingerspellRecogniser(static_bundle)
            print(f"Fingerspelling pathway: {static_bundle.config.n_classes} letters")
        except Exception as exc:
            print(f"Fingerspelling pathway unavailable: {exc}")

    predictor = LivePredictor(bundle, stride=stride, gate=MotionEnergyGate())
    sentence = SentenceBuilder()
    tts = _Speech(speech)

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_times: deque[float] = deque(maxlen=30)
    debug = False
    spelling_on = speller is not None
    committed_total = 0
    last = time.perf_counter()

    print("Running. Sign in front of the camera; press q to quit.\n")
    try:
        with mp_holistic.Holistic(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5,
                                  model_complexity=1) as holistic:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                if mirror:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False          # lets MediaPipe skip a copy
                results = holistic.process(rgb)

                keypoints = extract_holistic_keypoints(results)
                pred = predictor.update(keypoints)
                sentence.tick()

                if pred.committed:
                    sentence.add_gloss(pred.committed)
                    committed_total += 1
                    print(f"  + {pred.committed}")
                    if tts.enabled:
                        tts.say(pred.committed.replace("-", " "))

                # Static pathway: a hand held still is a fingerspelled letter,
                # which is how anything outside the word vocabulary gets in.
                if speller is not None and spelling_on:
                    if pred.route == "static":
                        letter = speller.update(keypoints)
                        if letter:
                            sentence.add_letter(letter)
                            print(f"  . {letter}")
                    elif pred.route == "dynamic":
                        speller.reset()

                if show_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles
                        .get_default_pose_landmarks_style())
                    for hand in (results.left_hand_landmarks,
                                 results.right_hand_landmarks):
                        mp_drawing.draw_landmarks(
                            frame, hand, mp_holistic.HAND_CONNECTIONS)

                now = time.perf_counter()
                fps_times.append(now - last)
                last = now
                fps = 1.0 / max(float(np.mean(fps_times)), 1e-6)

                panel = _draw_panel(cv2, frame.shape[0], {
                    "n_classes": bundle.config.n_classes,
                    "model": bundle.config.model_name,
                    "route": pred.route,
                    "label": pred.label,
                    "reason": pred.reason,
                    "confidence": pred.confidence,
                    "margin": pred.margin,
                    "threshold": bundle.config.confidence_threshold,
                    "text": sentence.text_line(),
                    "gloss": sentence.gloss_line(),
                    "pending": sentence.pending_word,
                    "debug": debug,
                    "energy": f"{pred.energy:.5f}",
                    "fps": f"{fps:.1f}",
                    "buffer": f"{pred.buffer_fill:.0%}",
                    "committed_total": str(committed_total),
                    "speech": tts.enabled,
                    "spelling": spelling_on,
                })

                cv2.imshow("Sign Language Translator",
                           np.hstack([frame, panel]))

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    sentence.clear()
                    predictor.reset()
                elif key == ord("u"):
                    sentence.undo()
                elif key == ord("d"):
                    debug = not debug
                elif key == ord("s"):
                    tts.enabled = not tts.enabled and tts.engine is not None
                    if tts.engine is None:
                        print("  speech unavailable (pip install pyttsx3)")
                elif key == ord("f"):
                    spelling_on = not spelling_on and speller is not None
    finally:
        cap.release()
        cv2.destroyAllWindows()

    final = sentence.text_line()
    if final:
        print(f"\nFinal: {final}")
        print(f"Gloss: {sentence.gloss_line()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/word_model")
    ap.add_argument("--static-model", default="models/letter_model")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--speech", action="store_true")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--no-landmarks", action="store_true")
    args = ap.parse_args()

    run(args.model, args.static_model, args.camera, args.stride,
        args.speech, not args.no_mirror, not args.no_landmarks)


if __name__ == "__main__":
    main()

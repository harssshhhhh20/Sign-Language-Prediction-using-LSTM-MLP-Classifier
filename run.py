#!/usr/bin/env python
"""Sign Language Translator - single entry point.

    python run.py doctor                        check the install
    python run.py collect --signer S01          record signs from a webcam
    python run.py merge                         combine per-signer recordings
    python run.py train                         train and package a model
    python run.py live                          run the translator
    python run.py replay                        test the live path, no webcam

Typical first run
-----------------
    python run.py collect --signer S01 --limit 15 --repetitions 25
    python run.py merge
    python run.py train --compare
    python run.py live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_doctor(args) -> int:
    """Check that everything needed is importable and report what is missing."""
    import importlib

    print("Environment check\n" + "-" * 50)
    required = {
        "numpy": "core",
        "sklearn": "classical models, metrics",
        "cv2": "webcam capture and UI",
        "mediapipe": "landmark extraction",
    }
    optional = {
        "tensorflow": "neural models (GRU/LSTM/CNN/Transformer)",
        "matplotlib": "figures",
        "pyttsx3": "speech output",
    }

    missing_required = []
    for mod, why in required.items():
        try:
            m = importlib.import_module(mod)
            print(f"  [ok]      {mod:14s} {getattr(m, '__version__', '?'):12s} {why}")
        except Exception:
            print(f"  [MISSING] {mod:14s} {'':12s} {why}")
            missing_required.append(mod)

    for mod, why in optional.items():
        try:
            m = importlib.import_module(mod)
            print(f"  [ok]      {mod:14s} {getattr(m, '__version__', '?'):12s} {why}")
        except Exception:
            print(f"  [absent]  {mod:14s} {'':12s} {why} (optional)")

    print("\nData and models\n" + "-" * 50)
    for path, what in [("data/dynamic/manifest.json", "training dataset"),
                       ("models/word_model/config.json", "word model"),
                       ("models/letter_model/config.json", "fingerspelling model"),
                       ("configs/vocabulary.txt", "vocabulary")]:
        print(f"  [{'ok' if Path(path).exists() else '--'}]  {path:38s} {what}")

    if Path("data/dynamic/manifest.json").exists():
        from slr.manifest import DatasetManifest
        m = DatasetManifest.load("data/dynamic/manifest.json")
        print("\n" + m.describe())

    if missing_required:
        print(f"\nInstall what is missing:\n    pip install -r requirements.txt")
        return 1
    print("\nReady.")
    return 0


def cmd_collect(args) -> int:
    from slr.data import collect_dynamic
    sys.argv = ["collect_dynamic"] + _passthrough(args, [
        "signer", "session", "vocab", "out", "repetitions", "sequence_length",
        "min_hand_presence", "camera", "seed", "limit", "no_idle",
    ])
    collect_dynamic.main()
    return 0


def cmd_collect_letters(args) -> int:
    from slr.data import collect_static
    sys.argv = ["collect_static"] + _passthrough(args, [
        "signer", "session", "out", "samples_per_letter",
        "include_dynamic_letters", "camera",
    ])
    collect_static.main()
    return 0


def cmd_migrate(args) -> int:
    """Rebuild the bundled dataset from the original MP_Data recordings.

    The manifest-backed dataset is derived, so it is not committed - this
    regenerates it in one step after a fresh clone.
    """
    from slr.data.migrate import migrate_dynamic, migrate_static

    did = False
    if Path(args.dynamic_source).exists():
        m = migrate_dynamic(Path(args.dynamic_source), Path("data/dynamic"))
        print(m.describe())
        print(f"\nWrote data/dynamic/manifest.json\n")
        did = True
    else:
        print(f"No dynamic source at {args.dynamic_source} - skipping.")

    if Path(args.static_source).exists():
        m = migrate_static(Path(args.static_source), Path("data/static"))
        print(m.describe())
        print(f"\nWrote data/static/manifest.json")
        did = True
    else:
        print(f"No static source at {args.static_source} - skipping.")

    if not did:
        print("\nNothing to migrate. If you have recorded your own data, use:")
        print("    python run.py merge")
        return 1
    return 0


def cmd_merge(args) -> int:
    """Combine every per-signer recording directory into one dataset."""
    from slr.data.loaders import merge_manifests

    raw = Path(args.raw)
    roots = sorted(p for p in raw.iterdir()
                   if p.is_dir() and (p / "manifest.json").exists())
    if not roots:
        print(f"No recording sessions found under {raw}/")
        print("Record one first:  python run.py collect --signer S01")
        return 1

    print(f"Merging {len(roots)} session(s):")
    for r in roots:
        print(f"  {r}")
    manifest = merge_manifests(roots, args.out)
    print("\n" + manifest.describe())
    print(f"\nWrote {args.out}/manifest.json")
    return 0


def cmd_train(args) -> int:
    from slr.train import train
    train(args.data, args.out, args.model, args.feature_group,
          args.normalisation, args.augment, args.seed, args.compare)
    return 0


def cmd_live(args) -> int:
    from slr.runtime.app import run
    run(args.model, args.static_model, args.camera, args.stride,
        args.speech, not args.no_mirror, not args.no_landmarks)
    return 0


def cmd_replay(args) -> int:
    from slr.runtime.replay import main as replay_main
    sys.argv = ["replay"] + _passthrough(args, ["model", "data", "limit", "verbose"])
    replay_main()
    return 0


def _passthrough(args, keys: list[str]) -> list[str]:
    """Rebuild argv for the delegated module's own parser."""
    out = []
    for k in keys:
        v = getattr(args, k, None)
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            out.append(flag)
        else:
            out.extend([flag, str(v)])
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="check the install")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("collect", help="record word signs from a webcam")
    p.add_argument("--signer", required=True)
    p.add_argument("--session", default=None)
    p.add_argument("--vocab", default="configs/vocabulary.txt")
    p.add_argument("--out", default=None)
    p.add_argument("--repetitions", type=int, default=30)
    p.add_argument("--sequence-length", type=int, default=30)
    p.add_argument("--min-hand-presence", type=float, default=0.7)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-idle", action="store_true")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("collect-letters", help="record fingerspelling")
    p.add_argument("--signer", required=True)
    p.add_argument("--session", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--samples-per-letter", type=int, default=60)
    p.add_argument("--include-dynamic-letters", action="store_true")
    p.add_argument("--camera", type=int, default=0)
    p.set_defaults(func=cmd_collect_letters)

    p = sub.add_parser("migrate",
                       help="rebuild the bundled dataset from MP_Data")
    p.add_argument("--dynamic-source", default="gesture/MP_Data")
    p.add_argument("--static-source", default="self_data_model/sign_data.pkl")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("merge", help="combine per-signer recordings")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/dynamic")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("train", help="train and package a model bundle")
    p.add_argument("--data", default="data/dynamic")
    p.add_argument("--out", default="models/word_model")
    p.add_argument("--model", default="random_forest")
    p.add_argument("--feature-group", default="pose_hands")
    p.add_argument("--normalisation", default="body_centred")
    p.add_argument("--augment", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare", action="store_true",
                   help="try several architectures and keep the best")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("live", help="run the translator on a webcam")
    p.add_argument("--model", default="models/word_model")
    p.add_argument("--static-model", default="models/letter_model")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--speech", action="store_true")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--no-landmarks", action="store_true")
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("replay", help="test the live path without a webcam")
    p.add_argument("--model", default="models/word_model")
    p.add_argument("--data", default="data/dynamic")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_replay)

    return ap


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())

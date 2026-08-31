"""Latency, throughput and footprint benchmark.

The efficiency claim ("runs on any machine") needs numbers attached, and the
numbers a reviewer expects are per-stage: landmark extraction dominates the
budget in landmark-based pipelines, and a paper that reports only classifier
latency is reporting the small half.

Reported per model: median / p95 / p99 single-window inference latency, the
implied classifier-only throughput, trainable parameters and float32
footprint. When MediaPipe is installed, the landmark-extraction stage is
timed too and an end-to-end budget is produced.

Usage
-----
    python -m slr.experiments.benchmark_latency --models lstm cnn1d random_forest
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from ..data.loaders import load_dynamic
from ..features import normalise_sequence, select_features
from ..models import build_model, available_models


def _percentiles(times_ms: list[float]) -> dict:
    a = np.asarray(times_ms)
    return {
        "mean_ms": float(a.mean()),
        "median_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def benchmark_models(
    data_root: str = "data/dynamic",
    models: list[str] | None = None,
    feature_group: str = "pose_hands",
    normalisation: str = "body_centred",
    n_warmup: int = 10,
    n_trials: int = 200,
    seed: int = 0,
) -> list[dict]:
    X_raw, y, manifest = load_dynamic(data_root)
    n_classes = len(manifest.labels)
    Xn = normalise_sequence(X_raw, normalisation)
    Xg = select_features(Xn, feature_group,
                         manifest.records[0].dominant_hand if manifest.records else "right")

    models = models or ["lstm_original", "lstm", "cnn1d", "gru",
                        "random_forest", "svm_rbf", "logreg"]
    have = set(available_models())
    rows = []

    for name in models:
        if name not in have:
            print(f"  skip {name} (unavailable)")
            continue
        print(f"  benchmarking {name} ...")
        model = build_model(name, n_classes, seed=seed)
        model.fit(Xg, y)

        one = Xg[:1]
        for _ in range(n_warmup):
            model.predict_proba(one)

        times = []
        for i in range(n_trials):
            sample = Xg[i % len(Xg): i % len(Xg) + 1]
            t0 = time.perf_counter()
            model.predict_proba(sample)
            times.append((time.perf_counter() - t0) * 1000.0)

        stats = _percentiles(times)
        rows.append({
            "model": name,
            "kind": model.kind,
            "feature_group": feature_group,
            "n_features": int(Xg.shape[-1]),
            "sequence_length": int(Xg.shape[1]),
            "n_params": model.n_parameters(),
            "size_kb": round(model.model_size_bytes() / 1024, 1)
            if model.model_size_bytes() > 0 else -1,
            "fit_seconds": round(model.fit_seconds, 2),
            **{k: round(v, 4) for k, v in stats.items()},
            "classifier_fps": round(1000.0 / stats["median_ms"], 1),
        })
    return rows


def benchmark_landmark_extraction(n_frames: int = 120,
                                  resolution: tuple[int, int] = (640, 480)) -> dict | None:
    """Time MediaPipe Holistic on synthetic frames.

    Synthetic frames give a stable lower bound on extraction cost; the model
    still runs its full graph. Real-camera timings vary with capture driver
    and are reported separately by the live demo.
    """
    try:
        import cv2  # noqa: F401
        import mediapipe as mp
    except Exception as exc:
        print(f"  landmark stage skipped ({exc.__class__.__name__}: {exc})")
        return None

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (resolution[1], resolution[0], 3),
                           dtype=np.uint8) for _ in range(min(n_frames, 30))]

    times = []
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5) as holistic:
        for _ in range(5):
            holistic.process(frames[0])
        for i in range(n_frames):
            f = frames[i % len(frames)]
            t0 = time.perf_counter()
            holistic.process(f)
            times.append((time.perf_counter() - t0) * 1000.0)

    stats = _percentiles(times)
    stats["extraction_fps"] = round(1000.0 / stats["median_ms"], 1)
    stats["resolution"] = f"{resolution[0]}x{resolution[1]}"
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/dynamic")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--feature-group", default="pose_hands")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--out", default="results/latency.json")
    ap.add_argument("--skip-landmarks", action="store_true")
    args = ap.parse_args()

    host = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count": __import__("os").cpu_count(),
    }
    print("Host:")
    for k, v in host.items():
        print(f"  {k:12s} {v}")

    print("\nClassifier inference:")
    rows = benchmark_models(args.data, args.models,
                            feature_group=args.feature_group,
                            n_trials=args.trials)

    landmark = None
    if not args.skip_landmarks:
        print("\nLandmark extraction:")
        landmark = benchmark_landmark_extraction()

    print("\n" + "=" * 96)
    print(f"{'model':16s} {'params':>10s} {'size KB':>9s} "
          f"{'median ms':>10s} {'p95 ms':>9s} {'p99 ms':>9s} {'clf FPS':>9s}")
    print("=" * 96)
    for r in sorted(rows, key=lambda x: x["median_ms"]):
        params = f"{r['n_params']:,}" if r["n_params"] > 0 else "-"
        size = f"{r['size_kb']}" if r["size_kb"] > 0 else "-"
        print(f"{r['model']:16s} {params:>10s} {size:>9s} "
              f"{r['median_ms']:>10.3f} {r['p95_ms']:>9.3f} "
              f"{r['p99_ms']:>9.3f} {r['classifier_fps']:>9.1f}")

    if landmark:
        print("\nLandmark extraction (MediaPipe Holistic, "
              f"{landmark['resolution']}): median {landmark['median_ms']:.1f} ms "
              f"-> {landmark['extraction_fps']:.1f} FPS")
        best = min(rows, key=lambda x: x["median_ms"]) if rows else None
        if best:
            total = landmark["median_ms"] + best["median_ms"]
            share = 100 * landmark["median_ms"] / total
            print(f"End-to-end with {best['model']}: {total:.1f} ms "
                  f"({1000 / total:.1f} FPS); extraction is {share:.0f}% of the budget.")
    else:
        print("\nLandmark stage not measured (MediaPipe unavailable). "
              "Classifier latency alone understates the real budget - "
              "install mediapipe and rerun before quoting an end-to-end number.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"host": host, "classifiers": rows, "landmark_extraction": landmark},
        indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

"""Train a deployable model bundle from collected data.

Does the three things that decide whether the live system works:

1. **Trains on held-out-honest splits**, so the accuracy printed here is
   roughly what you will see in front of the camera rather than a number
   inflated by near-duplicate sequences.
2. **Calibrates live thresholds** on held-out data. The confidence and
   margin thresholds control how willing the system is to commit to a word;
   guessing them means either a jittery output that fires on nothing, or a
   silent one that never fires.
3. **Reports confusable pairs**, so you know which signs to re-record rather
   than staring at a wrong prediction wondering why.

Usage
-----
    python -m slr.train --data data/dynamic --out models/word_model
    python -m slr.train --data data/dynamic --model gru --augment 6
    python -m slr.train --data data/dynamic --compare      # try several, keep best
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import numpy as np

from . import __version__, splits
from .bundle import BundleConfig, Calibration, ModelBundle
from .data.loaders import load_dynamic
from .evaluate import evaluate_fold
from .features import augment_sequences, normalise_sequence, select_features
from .models import build_model, available_models

# Candidates tried by --compare. Ordered by how well they tend to do on the
# small landmark datasets this project produces.
COMPARE_MODELS = ["random_forest", "extra_trees", "svm_rbf", "mlp",
                  "cnn1d", "bilstm", "gru"]


def _pick_protocol(manifest) -> str:
    """Strongest evaluation protocol the data supports."""
    for name in reversed(splits.PROTOCOL_LADDER):
        try:
            splits.make_folds(manifest, name)
            return name
        except Exception:
            continue
    return "random"


def calibrate_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    labels: list[str],
    idle_label: str = "__idle__",
    min_confidence: float = 0.55,
) -> tuple[float, float, list[dict]]:
    """Choose confidence and margin thresholds from held-out predictions.

    Sweeps candidate thresholds and picks the pair that maximises accuracy on
    the predictions it accepts while still accepting most of them. A system
    that answers 30% of the time at 99% accuracy is not usable, so coverage
    is part of the objective rather than an afterthought.
    """
    top2 = np.sort(proba, axis=1)[:, -2:]
    conf = top2[:, -1]
    margin = top2[:, -1] - top2[:, -2]
    pred = np.argmax(proba, axis=1)
    correct = pred == y_true

    idle_idx = labels.index(idle_label) if idle_label in labels else -1
    real = pred != idle_idx if idle_idx >= 0 else np.ones(len(pred), bool)

    # A floor on confidence regardless of what the sweep prefers. Held-out
    # sequences are pre-segmented and clean, so the sweep happily picks a very
    # low threshold; a live camera feeds the model half-formed gestures and
    # reaching-for-the-mouse, where a permissive threshold produces a stream
    # of confident nonsense.
    sweep, best, best_score = [], (min_confidence, 0.10), -1.0
    for c in [c for c in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                          0.85, 0.90] if c >= min_confidence]:
        for m in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
            accept = (conf >= c) & (margin >= m) & real
            n_acc = int(accept.sum())
            if n_acc == 0:
                continue
            precision = float(correct[accept].mean())
            coverage = n_acc / max(int(real.sum()), 1)
            # Harmonic mean: both matter, and neither can be traded to zero.
            score = (2 * precision * coverage / (precision + coverage)
                     if (precision + coverage) > 0 else 0.0)
            sweep.append({"confidence": c, "margin": m,
                          "precision": round(precision, 4),
                          "coverage": round(coverage, 4),
                          "score": round(score, 4), "n_accepted": n_acc})
            if score > best_score:
                best_score, best = score, (c, m)
    return best[0], best[1], sweep


def live_score(model, labels: list[str], X_test_raw: np.ndarray,
               y_test: np.ndarray, config, seed: int = 0) -> dict:
    """Score a candidate on the live path, not on pre-segmented sequences.

    Offline accuracy is measured on clean windows that start and end exactly
    when the sign does. Live, the model sees a sliding window over a
    continuous stream and something has to decide *when* a sign happened.
    Models with identical offline scores can differ enormously here - an
    estimator that is confidently wrong on half-formed windows produces a
    stream of spurious commits, and one that is merely uncertain produces
    none at all.

    So candidates are ranked by what they do on a simulated stream.
    """
    from .bundle import ModelBundle
    from .runtime.replay import make_rest_frames
    from .runtime.predictor import LivePredictor
    from .pipeline.gate import MotionEnergyGate

    bundle = ModelBundle(config, model)
    predictor = LivePredictor(bundle, stride=1, gate=MotionEnergyGate())
    rng = np.random.default_rng(seed)

    correct = missed = extra = 0
    for i in range(len(X_test_raw)):
        truth = labels[y_test[i]]
        fired: list[str] = []
        for frame in X_test_raw[i]:
            p = predictor.update(frame)
            if p.committed:
                fired.append(p.committed)
        for frame in make_rest_frames(X_test_raw[i], 18, rng):
            p = predictor.update(frame)
            if p.committed:
                fired.append(p.committed)
        if not fired:
            missed += 1
        else:
            correct += int(fired[0] == truth)
            extra += len(fired) - 1

    n = max(len(X_test_raw), 1)
    return {
        "live_accuracy": correct / n,
        "miss_rate": missed / n,
        "extra_rate": extra / n,
        # Spurious commits are worse than silence: a wrong word on screen is
        # actively misleading, a missing one is visibly missing.
        "live_score": correct / n - 0.5 * (extra / n),
    }


def find_confusable_pairs(confusion: np.ndarray, labels: list[str],
                          top_k: int = 10) -> list[dict]:
    """Most-confused label pairs, so you know what to re-record."""
    cm = np.asarray(confusion, dtype=float)
    totals = np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)
    rate = cm / totals
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": labels[i], "predicted": labels[j],
                              "count": int(cm[i, j]),
                              "rate": round(float(rate[i, j]), 3)})
    pairs.sort(key=lambda p: -p["count"])
    return pairs[:top_k]


def train(
    data_root: str = "data/dynamic",
    out: str = "models/word_model",
    model_name: str = "random_forest",
    feature_group: str = "pose_hands",
    normalisation: str = "body_centred",
    augment: int = 6,
    seed: int = 0,
    compare: bool = False,
) -> ModelBundle:
    X_raw, y, manifest = load_dynamic(data_root)
    labels = manifest.labels
    n_classes = len(labels)
    dominant = manifest.records[0].dominant_hand if manifest.records else "right"

    print(manifest.describe())

    protocol = _pick_protocol(manifest)
    folds = splits.make_folds(manifest, protocol)
    print(f"\nEvaluation protocol: {protocol} ({len(folds)} fold(s))")

    def prepare(X: np.ndarray) -> np.ndarray:
        """raw holistic -> normalised -> feature group. Order matters."""
        return select_features(normalise_sequence(X, normalisation),
                               feature_group, dominant)

    Xg = prepare(X_raw)
    print(f"Input: {Xg.shape[1]} frames x {Xg.shape[2]} features "
          f"({feature_group}, {normalisation})")

    candidates = COMPARE_MODELS if compare else [model_name]
    have = set(available_models())
    candidates = [m for m in candidates if m in have]
    if not candidates:
        raise SystemExit(f"none of the requested models are available: {model_name}")

    probe_config = BundleConfig(
        labels=labels, feature_group=feature_group,
        normalisation=normalisation, sequence_length=int(Xg.shape[1]),
        dominant_hand=dominant, model_name="probe", model_kind="classical",
    )

    print(f"\n{'model':16s} {'offline acc':>12s} {'macro-F1':>10s} "
          f"{'live acc':>10s} {'missed':>8s} {'extra':>8s}")
    print("-" * 68)

    results = {}
    for cand in candidates:
        fold_scores, all_true, all_proba, live = [], [], [], []
        for fold in folds:
            # Augment on raw coordinates, then normalise and select.
            X_tr_raw, y_tr = X_raw[fold.train_idx], y[fold.train_idx]
            if augment:
                X_tr_raw, y_tr = augment_sequences(
                    X_tr_raw, y_tr, n_copies=augment, seed=seed)
            X_tr = prepare(X_tr_raw)
            X_te, y_te = Xg[fold.test_idx], y[fold.test_idx]

            model = build_model(cand, n_classes, seed=seed)
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)

            res = evaluate_fold(y_te, proba, labels, protocol=protocol,
                                fold_id=fold.fold_id, model=cand,
                                feature_group=feature_group,
                                normalisation=normalisation, seed=seed,
                                n_train=len(X_tr), n_features=Xg.shape[-1])
            fold_scores.append(res)
            all_true.append(y_te)
            all_proba.append(proba)

            probe_config.model_name = cand
            probe_config.model_kind = model.kind
            live.append(live_score(model, labels, X_raw[fold.test_idx],
                                   y_te, probe_config, seed))

        acc = float(np.mean([r.accuracy for r in fold_scores]))
        f1 = float(np.mean([r.macro_f1 for r in fold_scores]))
        lv = {k: float(np.mean([d[k] for d in live])) for k in live[0]}
        results[cand] = {
            "accuracy": acc, "macro_f1": f1,
            "y_true": np.concatenate(all_true),
            "proba": np.concatenate(all_proba),
            "folds": fold_scores, **lv,
        }
        print(f"{cand:16s} {acc:11.1%} {f1:9.1%} {lv['live_accuracy']:9.1%} "
              f"{lv['miss_rate']:7.1%} {lv['extra_rate']:7.1%}")

    # Rank by live behaviour: that is what the user experiences.
    best_name = max(results, key=lambda k: (results[k]["live_score"],
                                            results[k]["macro_f1"]))
    best = results[best_name]
    print(f"\nBest on the live path: {best_name} "
          f"(live {best['live_accuracy']:.1%}, offline {best['macro_f1']:.1%})")

    # ---- calibrate thresholds on the pooled held-out predictions ----------
    conf_t, margin_t, sweep = calibrate_thresholds(
        best["y_true"], best["proba"], labels)
    print(f"Calibrated thresholds: confidence >= {conf_t:.2f}, "
          f"margin >= {margin_t:.2f}")

    cm = np.zeros((n_classes, n_classes), dtype=float)
    for r in best["folds"]:
        cm += np.asarray(r.confusion, dtype=float)
    confusable = find_confusable_pairs(cm, labels)
    if confusable:
        print("\nMost confused pairs (re-record these first):")
        for p in confusable[:6]:
            print(f"  {p['true']:>16s} -> {p['predicted']:<16s} "
                  f"{p['count']:3d}x ({p['rate']:.0%} of that class)")

    # ---- retrain on everything, then package -----------------------------
    print(f"\nRetraining {best_name} on all {len(Xg)} sequences ...")
    if augment:
        X_all_raw, y_all = augment_sequences(X_raw, y, n_copies=augment, seed=seed)
        X_all = prepare(X_all_raw)
    else:
        X_all, y_all = Xg, y
    final = build_model(best_name, n_classes, seed=seed)
    final.fit(X_all, y_all)

    per_class = {}
    for r in best["folds"]:
        for lab, v in r.per_class_f1.items():
            per_class.setdefault(lab, []).append(v)
    per_class = {k: round(float(np.mean(v)), 4) for k, v in per_class.items()}

    config = BundleConfig(
        labels=labels,
        feature_group=feature_group,
        normalisation=normalisation,
        sequence_length=int(Xg.shape[1]),
        dominant_hand=dominant,
        model_name=best_name,
        model_kind=final.kind,
        confidence_threshold=conf_t,
        margin_threshold=margin_t,
        n_signers=len(manifest.signers),
        n_sequences=len(manifest),
        trained_at=_dt.datetime.now().isoformat(timespec="seconds"),
        slr_version=__version__,
        notes=f"augment x{augment}" if augment else "no augmentation",
    )
    calibration = Calibration(
        accuracy=best["accuracy"], macro_f1=best["macro_f1"],
        per_class_f1=per_class, n_eval=int(len(best["y_true"])),
        protocol=protocol, threshold_sweep=sweep[:40],
        confusable_pairs=confusable,
    )

    bundle = ModelBundle(config, final, calibration)
    path = bundle.save(out)
    print(f"\n{bundle.describe()}")
    print(f"\nSaved bundle to {path}/")

    weak = [k for k, v in per_class.items() if v < 0.6]
    if weak:
        print(f"\nWeak classes (F1 < 0.60): {', '.join(sorted(weak))}")
        print("  More repetitions of these, ideally from a different signer, "
              "is the highest-value thing you can record next.")
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/dynamic")
    ap.add_argument("--out", default="models/word_model")
    ap.add_argument("--model", default="random_forest")
    ap.add_argument("--feature-group", default="pose_hands")
    ap.add_argument("--normalisation", default="body_centred")
    ap.add_argument("--augment", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", action="store_true",
                    help="train several architectures and keep the best")
    args = ap.parse_args()

    train(args.data, args.out, args.model, args.feature_group,
          args.normalisation, args.augment, args.seed, args.compare)


if __name__ == "__main__":
    main()

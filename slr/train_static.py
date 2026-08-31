"""Train the fingerspelling (static handshape) model.

Separate from :mod:`slr.train` because the task is genuinely different: one
frame instead of a sequence, 69 hand-shape features instead of 258 body
landmarks, and no temporal segmentation to get right.

Usage
-----
    python -m slr.train_static --data data/static --out models/letter_model
"""

from __future__ import annotations

import argparse
import datetime as _dt

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from . import __version__
from .bundle import BundleConfig, Calibration, ModelBundle
from .data.loaders import load_static
from .features import normalise_static
from .models.classical import CLASSICAL
from .train import calibrate_thresholds, find_confusable_pairs

CANDIDATES = ["random_forest", "svm_rbf", "mlp", "extra_trees", "knn"]


def train_static(
    data_root: str = "data/static",
    out: str = "models/letter_model",
    model_name: str | None = None,
    scale_invariant: bool = True,
    n_folds: int = 5,
    seed: int = 0,
) -> ModelBundle:
    X, y, manifest = load_static(data_root)
    labels = manifest.labels
    n_classes = len(labels)

    print(manifest.describe())
    X = normalise_static(X, scale_invariant)
    print(f"\nInput: {X.shape[0]} samples x {X.shape[1]} features "
          f"(scale-invariant: {scale_invariant})")

    # A signer-independent split when the data supports it, otherwise
    # stratified k-fold. The warning above already says which one this is.
    signers = manifest.signer_array()
    groups = sorted(set(signers))
    if len(groups) >= 2:
        protocol = "signer_independent"
        fold_iter = [(np.flatnonzero(signers != g), np.flatnonzero(signers == g))
                     for g in groups]
    else:
        protocol = f"stratified_{n_folds}fold"
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_iter = list(skf.split(X, y))

    print(f"Evaluation protocol: {protocol}\n")
    print(f"{'model':16s} {'accuracy':>10s} {'macro-F1':>10s}")
    print("-" * 38)

    candidates = [model_name] if model_name else CANDIDATES
    results = {}
    for cand in candidates:
        if cand not in CLASSICAL:
            continue
        accs, f1s, trues, probas = [], [], [], []
        for tr, te in fold_iter:
            model = CLASSICAL[cand](n_classes=n_classes, seed=seed)
            model.fit(X[tr], y[tr])
            proba = model.predict_proba(X[te])
            pred = np.argmax(proba, axis=1)
            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
            trues.append(y[te])
            probas.append(proba)
        results[cand] = {
            "accuracy": float(np.mean(accs)),
            "macro_f1": float(np.mean(f1s)),
            "y_true": np.concatenate(trues),
            "proba": np.concatenate(probas),
        }
        print(f"{cand:16s} {results[cand]['accuracy']:9.1%} "
              f"{results[cand]['macro_f1']:9.1%}")

    if not results:
        raise SystemExit(f"no usable model among {candidates}")

    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best = results[best_name]
    print(f"\nBest: {best_name} (macro-F1 {best['macro_f1']:.1%})")

    # Fingerspelling runs continuously rather than in discrete signs, so a
    # wrong letter is cheap to correct but a stream of them is unreadable.
    # Hold a higher confidence floor here than for word signs.
    conf_t, margin_t, sweep = calibrate_thresholds(
        best["y_true"], best["proba"], labels, min_confidence=0.70)
    print(f"Calibrated thresholds: confidence >= {conf_t:.2f}, "
          f"margin >= {margin_t:.2f}")

    cm = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(best["y_true"], np.argmax(best["proba"], axis=1)):
        cm[t, p] += 1
    confusable = find_confusable_pairs(cm, labels)
    if confusable:
        print("\nMost confused letters:")
        for p in confusable[:6]:
            print(f"  {p['true']:>6s} -> {p['predicted']:<6s} "
                  f"{p['count']:3d}x ({p['rate']:.0%})")

    print(f"\nRetraining {best_name} on all {len(X)} samples ...")
    final = CLASSICAL[best_name](n_classes=n_classes, seed=seed)
    final.fit(X, y)

    config = BundleConfig(
        labels=labels,
        feature_group="static_hand",
        normalisation="scale_invariant" if scale_invariant else "raw",
        sequence_length=1,
        static_feature_dim=int(X.shape[1]),
        model_name=best_name,
        model_kind="classical",
        confidence_threshold=conf_t,
        margin_threshold=margin_t,
        smoothing_window=9,
        min_agreement=0.7,
        n_signers=len(manifest.signers),
        n_sequences=len(manifest),
        trained_at=_dt.datetime.now().isoformat(timespec="seconds"),
        slr_version=__version__,
        notes="static fingerspelling pathway",
    )
    calibration = Calibration(
        accuracy=best["accuracy"], macro_f1=best["macro_f1"],
        n_eval=int(len(best["y_true"])), protocol=protocol,
        threshold_sweep=sweep[:40], confusable_pairs=confusable,
    )

    bundle = ModelBundle(config, final, calibration)
    path = bundle.save(out)
    print(f"\n{bundle.describe()}")
    print(f"\nSaved bundle to {path}/")
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/static")
    ap.add_argument("--out", default="models/letter_model")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-scale-invariant", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_static(args.data, args.out, args.model,
                 not args.no_scale_invariant, args.folds, args.seed)


if __name__ == "__main__":
    main()

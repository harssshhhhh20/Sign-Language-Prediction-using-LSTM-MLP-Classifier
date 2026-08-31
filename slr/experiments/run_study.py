"""Main experiment driver: protocol x feature-group x model x seed.

This produces the tables the paper is built from:

    Table I    accuracy under each evaluation protocol (the headline result)
    Table II   feature-group ablation
    Table III  model comparison at fixed protocol and feature group
    Table IV   cost: parameters, footprint, fit time

Usage
-----
    # fast sanity sweep
    python -m slr.experiments.run_study --preset quick

    # protocol ladder, the headline experiment
    python -m slr.experiments.run_study --preset protocol

    # full grid (slow)
    python -m slr.experiments.run_study --preset full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .. import splits
from ..data.loaders import load_dynamic
from ..evaluate import FoldResult, aggregate, evaluate_fold, format_mean_std
from ..features import augment_sequences, normalise_sequence, select_features
from ..manifest import DatasetManifest, build_capability_report
from ..models import build_model, available_models
from ..schema import group_size


# ---------------------------------------------------------------------------
# Inner validation split
# ---------------------------------------------------------------------------


def carve_validation(
    manifest: DatasetManifest,
    train_idx: np.ndarray,
    protocol: str,
    seed: int = 0,
    frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a training fold into fit/validation under the same protocol.

    Early stopping needs a validation set. Taking it by random shuffle - as
    ``validation_split=0.2`` in the original code does - reintroduces exactly
    the leakage the outer protocol removed, and the early-stopping decision
    is then made on data statistically identical to the training set. So the
    inner split mirrors the outer one.
    """
    rng = np.random.default_rng(seed)
    y = manifest.label_array()[train_idx]

    if protocol == "signer_independent":
        signers = manifest.signer_array()[train_idx]
        uniq = sorted(set(signers))
        if len(uniq) >= 2:
            held = uniq[rng.integers(len(uniq))]
            mask = signers == held
            if mask.sum() and (~mask).sum():
                return train_idx[~mask], train_idx[mask]

    if protocol == "session_independent":
        sessions = manifest.session_array()[train_idx]
        uniq = sorted(set(sessions))
        if len(uniq) >= 2:
            held = uniq[rng.integers(len(uniq))]
            mask = sessions == held
            if mask.sum() and (~mask).sum():
                return train_idx[~mask], train_idx[mask]

    if protocol == "random":
        # Mirror the leaky outer protocol so the baseline row stays faithful.
        perm = rng.permutation(len(train_idx))
        k = max(1, int(round(frac * len(train_idx))))
        return train_idx[perm[k:]], train_idx[perm[:k]]

    # Temporal tail of each class - the honest default.
    order = manifest.order_array()[train_idx]
    val_local = []
    for cls in np.unique(y):
        cls_pos = np.flatnonzero(y == cls)
        cls_pos = cls_pos[np.argsort(order[cls_pos])]
        k = max(1, int(round(frac * len(cls_pos))))
        val_local.append(cls_pos[-k:])
    val_local = np.concatenate(val_local)
    fit_local = np.setdiff1d(np.arange(len(train_idx)), val_local)
    return train_idx[fit_local], train_idx[val_local]


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
    "quick": dict(
        protocols=["random", "temporal_holdout"],
        feature_groups=["full", "hands"],
        normalisations=["body_centred"],
        models=["random_forest", "logreg"],
        seeds=[0],
        augment=0,
    ),
    "protocol": dict(
        protocols=["random", "temporal_holdout", "block_disjoint",
                   "session_independent", "signer_independent"],
        feature_groups=["pose_hands"],
        normalisations=["body_centred"],
        models=["lstm_original", "lstm", "cnn1d", "random_forest", "svm_rbf"],
        seeds=[0, 1, 2, 3, 4],
        augment=0,
    ),
    "ablation": dict(
        protocols=["temporal_holdout", "block_disjoint"],
        feature_groups=["full", "pose_hands", "upper_pose_hands",
                        "articulator_hands", "hands", "dominant_hand", "face"],
        normalisations=["body_centred"],
        models=["random_forest", "lstm"],
        seeds=[0, 1, 2],
        augment=0,
    ),
    "normalisation": dict(
        protocols=["block_disjoint"],
        feature_groups=["pose_hands", "hands"],
        normalisations=["raw", "wrist_relative", "body_centred"],
        models=["random_forest", "lstm"],
        seeds=[0, 1, 2],
        augment=0,
    ),
    "models": dict(
        protocols=["block_disjoint"],
        feature_groups=["pose_hands"],
        normalisations=["body_centred"],
        models=["lstm_original", "lstm", "bilstm", "gru", "cnn1d", "transformer",
                "random_forest", "extra_trees", "svm_rbf", "logreg", "knn", "mlp"],
        seeds=[0, 1, 2],
        augment=0,
    ),
    "augmentation": dict(
        protocols=["block_disjoint"],
        feature_groups=["pose_hands"],
        normalisations=["body_centred"],
        models=["lstm", "random_forest"],
        seeds=[0, 1, 2],
        augment=4,
    ),
    "full": dict(
        protocols=["random", "temporal_holdout", "block_disjoint",
                   "session_independent", "signer_independent"],
        feature_groups=["full", "pose_hands", "articulator_hands",
                        "hands", "dominant_hand", "face"],
        normalisations=["raw", "body_centred"],
        models=["lstm_original", "lstm", "bilstm", "gru", "cnn1d", "transformer",
                "random_forest", "svm_rbf", "logreg"],
        seeds=[0, 1, 2],
        augment=0,
    ),
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_study(
    data_root: str = "data/dynamic",
    out_dir: str = "results",
    preset: str = "quick",
    overrides: dict | None = None,
) -> list[FoldResult]:
    cfg = dict(PRESETS[preset])
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})

    X_raw, y, manifest = load_dynamic(data_root)
    labels = manifest.labels
    n_classes = len(labels)
    dominant = manifest.records[0].dominant_hand if manifest.records else "right"

    print(manifest.describe())
    caps = build_capability_report(manifest)
    print("\nProtocol availability:")
    for name in splits.PROTOCOL_LADDER:
        ok = caps.get(name, False)
        print(f"  {'[ok]  ' if ok else '[skip]'} {name:22s} "
              f"{splits.PROTOCOL_DESCRIPTIONS[name]}")

    requested = [p for p in cfg["protocols"] if p in splits.PROTOCOLS]
    runnable = [p for p in requested if caps.get(p, False)]
    skipped = [p for p in requested if p not in runnable]

    have_models = set(available_models())
    models = [m for m in cfg["models"] if m in have_models]
    missing_models = [m for m in cfg["models"] if m not in have_models]
    if missing_models:
        print(f"\nNOTE: unavailable models skipped: {missing_models}")

    # Normalise once per mode; feature selection afterwards is a cheap slice.
    normed = {mode: normalise_sequence(X_raw, mode) for mode in cfg["normalisations"]}

    results: list[FoldResult] = []
    t_start = time.perf_counter()
    total = 0

    for protocol in runnable:
        folds = splits.make_folds(manifest, protocol)
        for fold in folds:
            for norm in cfg["normalisations"]:
                Xn = normed[norm]
                for group in cfg["feature_groups"]:
                    Xg = select_features(Xn, group, dominant)
                    n_feat = Xg.shape[-1]

                    fit_idx, val_idx = carve_validation(
                        manifest, fold.train_idx, protocol, seed=0
                    )
                    X_val, y_val = Xg[val_idx], y[val_idx]
                    X_test, y_test = Xg[fold.test_idx], y[fold.test_idx]

                    for model_name in models:
                        for seed in cfg["seeds"]:
                            if cfg.get("augment"):
                                # Augment raw coordinates, then normalise and
                                # select - the transforms address the landmark
                                # blocks individually.
                                X_aug, y_fit = augment_sequences(
                                    X_raw[fit_idx], y[fit_idx],
                                    n_copies=int(cfg["augment"]), seed=seed
                                )
                                X_fit = select_features(
                                    normalise_sequence(X_aug, norm), group, dominant
                                )
                            else:
                                X_fit, y_fit = Xg[fit_idx], y[fit_idx]
                            try:
                                model = build_model(model_name, n_classes, seed=seed)
                                model.fit(X_fit, y_fit, X_val, y_val)
                                proba = model.predict_proba(X_test)
                            except Exception as exc:      # keep the sweep alive
                                print(f"    !! {model_name}/{group}/{protocol}"
                                      f"/{fold.fold_id}/seed{seed}: {exc}")
                                continue

                            res = evaluate_fold(
                                y_test, proba, labels,
                                protocol=protocol,
                                fold_id=fold.fold_id,
                                model=model_name,
                                feature_group=group,
                                normalisation=norm,
                                seed=seed,
                                n_train=len(X_fit),
                                n_features=n_feat,
                                n_params=model.n_parameters(),
                                size_bytes=model.model_size_bytes(),
                                fit_seconds=model.fit_seconds,
                                held_out=fold.held_out,
                            )
                            results.append(res)
                            total += 1
                            print(f"  {protocol:20s} {fold.fold_id:16s} "
                                  f"{model_name:14s} {group:18s} s{seed} "
                                  f"acc={res.accuracy:.3f} "
                                  f"f1={res.macro_f1:.3f} "
                                  f"n_test={res.n_test}")

    elapsed = time.perf_counter() - t_start
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_results(out, preset, results, cfg, manifest, caps, skipped, elapsed)
    _print_summary(results, skipped)
    print(f"\n{total} runs in {elapsed:.1f}s -> {out}/{preset}_results.csv")
    return results


def _write_results(out: Path, preset: str, results: list[FoldResult], cfg: dict,
                   manifest: DatasetManifest, caps: dict, skipped: list, elapsed: float):
    import csv

    rows = [r.to_row() for r in results]
    if rows:
        with open(out / f"{preset}_results.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    payload = {
        "preset": preset,
        "config": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                   for k, v in cfg.items()},
        "dataset": manifest.summary(),
        "protocol_availability": caps,
        "protocols_skipped": skipped,
        "elapsed_seconds": round(elapsed, 2),
        "results": [asdict(r) for r in results],
    }
    (out / f"{preset}_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _print_summary(results: list[FoldResult], skipped: list[str]) -> None:
    if not results:
        print("\nNo results produced.")
        return

    print("\n" + "=" * 78)
    print("ACCURACY BY PROTOCOL (mean +/- std over folds and seeds)")
    print("=" * 78)
    by_proto: dict[str, list[FoldResult]] = {}
    for r in results:
        by_proto.setdefault(r.protocol, []).append(r)

    for proto in splits.PROTOCOL_LADDER:
        if proto not in by_proto:
            continue
        group = by_proto[proto]
        acc, f1 = aggregate(group, "accuracy"), aggregate(group, "macro_f1")
        print(f"  {proto:22s} acc {format_mean_std(acc):>14s}   "
              f"macro-F1 {format_mean_std(f1):>14s}   (n={acc['n']})")

    print("\n" + "=" * 78)
    print("ACCURACY BY MODEL")
    print("=" * 78)
    by_model: dict[str, list[FoldResult]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)
    for model, group in sorted(by_model.items(),
                               key=lambda kv: -aggregate(kv[1], "accuracy")["mean"]):
        acc = aggregate(group, "accuracy")
        params = group[0].n_params
        print(f"  {model:22s} acc {format_mean_std(acc):>14s}   "
              f"params={params:>10,}   (n={acc['n']})")

    print("\n" + "=" * 78)
    print("ACCURACY BY FEATURE GROUP")
    print("=" * 78)
    by_group: dict[str, list[FoldResult]] = {}
    for r in results:
        by_group.setdefault(r.feature_group, []).append(r)
    for grp, rs in sorted(by_group.items(), key=lambda kv: -kv[1][0].n_features):
        acc = aggregate(rs, "accuracy")
        print(f"  {grp:22s} dims={rs[0].n_features:>5d}   "
              f"acc {format_mean_std(acc):>14s}   (n={acc['n']})")

    if skipped:
        print("\n" + "=" * 78)
        print("PROTOCOLS SKIPPED - dataset cannot express them")
        print("=" * 78)
        for p in skipped:
            print(f"  {p}: {splits.PROTOCOL_DESCRIPTIONS[p]}")
        print("\n  These are the protocols a reviewer will ask for. Recording")
        print("  additional signers and sessions is what unlocks them.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/dynamic")
    ap.add_argument("--out", default="results")
    ap.add_argument("--preset", default="quick", choices=sorted(PRESETS))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--protocols", nargs="*", default=None)
    ap.add_argument("--feature-groups", nargs="*", default=None, dest="feature_groups")
    ap.add_argument("--normalisations", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--augment", type=int, default=None)
    args = ap.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if k in {"models", "protocols", "feature_groups",
                          "normalisations", "seeds", "augment"}}
    run_study(args.data, args.out, args.preset, overrides)


if __name__ == "__main__":
    sys.exit(main())

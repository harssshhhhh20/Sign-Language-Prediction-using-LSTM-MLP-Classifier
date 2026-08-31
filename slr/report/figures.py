"""Generate the paper's figures.

Figures are written at 300 dpi in both PNG and PDF. IEEE templates want
vector art; PDF is the version to \\includegraphics.

Usage
-----
    python -m slr.report.figures --results results/ablation_results.json \\
        --out paper/figures
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from ..splits import PROTOCOL_LADDER     # noqa: E402

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Colour-blind safe (Okabe-Ito).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def _load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        payload = json.loads(Path(p).read_text(encoding="utf-8"))
        rows.extend(payload["results"] if isinstance(payload, dict) else payload)
    return rows


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {out / name}.png / .pdf")


def _stats(rows: list[dict], key: str = "accuracy") -> tuple[float, float]:
    v = np.asarray([r[key] for r in rows], dtype=float)
    return (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0)


# ---------------------------------------------------------------------------


def fig_protocol_ladder(rows: list[dict], out: Path) -> None:
    """Accuracy as the evaluation protocol tightens - the headline figure."""
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["protocol"], r["model"])].append(r)

    protocols = [p for p in PROTOCOL_LADDER if any(k[0] == p for k in by)]
    models = sorted({k[1] for k in by})
    if not protocols:
        return

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(protocols))
    width = min(0.8 / max(len(models), 1), 0.22)

    for i, m in enumerate(models):
        means, stds = [], []
        for p in protocols:
            rs = by.get((p, m), [])
            mu, sd = _stats(rs) if rs else (np.nan, 0.0)
            means.append(mu * 100)
            stds.append(sd * 100)
        ax.bar(x + (i - len(models) / 2 + 0.5) * width, means, width,
               yerr=stds, capsize=2.5, label=m,
               color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.5)

    n_classes = 4
    ax.axhline(100 / n_classes, ls="--", lw=1, color="#666",
               label=f"chance ({100 / n_classes:.0f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in protocols], fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Evaluation protocol  (leakier $\\leftarrow$ | $\\rightarrow$ stricter)")
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="lower left")
    ax.set_title("Reported accuracy depends on how the test set was built",
                 fontsize=10, loc="left")
    _save(fig, out, "protocol_ladder")


def fig_feature_ablation(rows: list[dict], out: Path) -> None:
    """Accuracy vs input dimensionality."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["feature_group"]].append(r)
    if not by:
        return

    groups = sorted(by, key=lambda g: by[g][0]["n_features"])
    dims = [by[g][0]["n_features"] for g in groups]
    means = [_stats(by[g])[0] * 100 for g in groups]
    stds = [_stats(by[g])[1] * 100 for g in groups]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = ["#D55E00" if g == "face" else "#0072B2" for g in groups]
    ax.errorbar(dims, means, yerr=stds, fmt="none", ecolor="#999",
                capsize=3, lw=1, zorder=1)
    ax.scatter(dims, means, s=48, c=colors, zorder=2, edgecolor="white",
               linewidth=0.8)

    for g, d, m in zip(groups, dims, means):
        ax.annotate(f"{g}\n({d}d)", (d, m), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7,
                    color="#D55E00" if g == "face" else "#333")

    ax.set_xscale("log")
    ax.set_xlabel("Input dimensionality per frame (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 112)
    ax.axhline(25, ls="--", lw=1, color="#666")
    ax.annotate("chance", (dims[0], 26), fontsize=7, color="#666")
    ax.set_title("Most of the holistic feature vector is not needed",
                 fontsize=10, loc="left")
    _save(fig, out, "feature_ablation")


def fig_seed_variance(rows: list[dict], out: Path) -> None:
    """Spread across seeds - the case for reporting mean ± std."""
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r["accuracy"] * 100)
    if not by:
        return

    models = sorted(by, key=lambda m: -np.mean(by[m]))
    data = [by[m] for m in models]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    # matplotlib renamed `labels` to `tick_labels` in 3.9; support both.
    try:
        bp = ax.boxplot(data, tick_labels=models, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", lw=1.2),
                        flierprops=dict(marker="o", ms=3, alpha=0.5))
    except TypeError:
        bp = ax.boxplot(data, labels=models, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", lw=1.2),
                        flierprops=dict(marker="o", ms=3, alpha=0.5))
    for patch, c in zip(bp["boxes"], PALETTE * 4):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#444")

    for i, vals in enumerate(data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.045, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=8,
                   color="#222", alpha=0.45, zorder=3)

    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_title("Run-to-run spread across folds and seeds",
                 fontsize=10, loc="left")
    _save(fig, out, "seed_variance")


def fig_confusion(rows: list[dict], out: Path, labels: list[str] | None = None) -> None:
    """Summed confusion matrix over the strictest available protocol."""
    for proto in reversed(PROTOCOL_LADDER):
        subset = [r for r in rows if r["protocol"] == proto and r.get("confusion")]
        if subset:
            break
    else:
        return

    cms = np.array([r["confusion"] for r in subset], dtype=float)
    cm = cms.sum(axis=0)
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)
    n = cm.shape[0]
    labels = labels or [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if cmn[i, j] > 0.5 else "#222")
    fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
    ax.set_title(f"Confusion, {proto.replace('_', ' ')}", fontsize=10, loc="left")
    _save(fig, out, "confusion")


FIGURES = {
    "protocol_ladder": fig_protocol_ladder,
    "feature_ablation": fig_feature_ablation,
    "seed_variance": fig_seed_variance,
    "confusion": fig_confusion,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--figures", nargs="*", default=sorted(FIGURES))
    ap.add_argument("--labels", nargs="*", default=None)
    args = ap.parse_args()

    rows = _load(args.results)
    if not rows:
        raise SystemExit("no results rows found")
    out = Path(args.out)

    print(f"Generating figures from {len(rows)} result rows:")
    for name in args.figures:
        fn = FIGURES[name]
        if name == "confusion":
            fn(rows, out, args.labels)
        else:
            fn(rows, out)


if __name__ == "__main__":
    main()

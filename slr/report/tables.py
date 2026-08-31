"""Generate the paper's tables from a results file.

Emits both Markdown (for the repository README and for reading) and LaTeX
(``booktabs``, IEEE two-column safe) so the numbers in the paper are
generated from the results file rather than retyped. Retyped numbers drift,
and a drifted number in a submitted paper is the kind of error that is found
by a reviewer rather than by you.

Usage
-----
    python -m slr.report.tables --results results/protocol_results.json \\
        --out paper/tables
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..splits import PROTOCOL_DESCRIPTIONS, PROTOCOL_LADDER


def _load(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["results"] if isinstance(payload, dict) else payload


def _agg(rows: list[dict], key: str) -> tuple[float, float, int]:
    v = np.asarray([r[key] for r in rows], dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"), 0)
    return (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size))


def _fmt(mean: float, std: float, pct: bool = True) -> str:
    if np.isnan(mean):
        return "--"
    s = 100 if pct else 1
    return f"{mean * s:.1f} $\\pm$ {std * s:.1f}"


def _fmt_md(mean: float, std: float, pct: bool = True) -> str:
    if np.isnan(mean):
        return "--"
    s = 100 if pct else 1
    return f"{mean * s:.1f} ± {std * s:.1f}"


# ---------------------------------------------------------------------------
# Table I - evaluation protocol
# ---------------------------------------------------------------------------


def table_protocol(rows: list[dict], latex: bool = False) -> str:
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["protocol"], r["model"])].append(r)

    protocols = [p for p in PROTOCOL_LADDER if any(k[0] == p for k in by)]
    models = sorted({k[1] for k in by})
    f = _fmt if latex else _fmt_md

    header = ["Protocol"] + models
    body = []
    for p in protocols:
        row = [p.replace("_", r"\_" if latex else "_")]
        for m in models:
            rs = by.get((p, m), [])
            row.append(f(*_agg(rs, "accuracy")[:2]) if rs else "--")
        body.append(row)

    caption = ("Accuracy (\\%) by evaluation protocol, mean $\\pm$ std over "
               "folds and seeds. Protocols are ordered from most to least "
               "leakage.")
    return _render(header, body, caption, "tab:protocol", latex)


# ---------------------------------------------------------------------------
# Table II - feature ablation
# ---------------------------------------------------------------------------


def table_ablation(rows: list[dict], latex: bool = False) -> str:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["feature_group"]].append(r)

    f = _fmt if latex else _fmt_md
    order = sorted(by, key=lambda g: -by[g][0]["n_features"])

    header = ["Feature group", "Dims", "% of full", "Accuracy", "Macro-F1"]
    full_dims = max((by[g][0]["n_features"] for g in by), default=1)
    body = []
    for g in order:
        rs = by[g]
        d = rs[0]["n_features"]
        body.append([
            g.replace("_", r"\_" if latex else "_"),
            str(d),
            f"{100 * d / full_dims:.1f}",
            f(*_agg(rs, "accuracy")[:2]),
            f(*_agg(rs, "macro_f1")[:2]),
        ])

    caption = ("Feature-group ablation. Dimensionality is per frame. The "
               "\\texttt{face} row is a control: manual signs cannot be "
               "distinguished by face landmarks, so accuracy above chance "
               "there measures leakage rather than recognition.")
    return _render(header, body, caption, "tab:ablation", latex)


# ---------------------------------------------------------------------------
# Table III - model comparison
# ---------------------------------------------------------------------------


def table_models(rows: list[dict], latex: bool = False) -> str:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)

    f = _fmt if latex else _fmt_md
    entries = []
    for m, rs in by.items():
        acc_mean, acc_std, n = _agg(rs, "accuracy")
        params = rs[0].get("n_params", -1)
        entries.append((acc_mean, m, rs, acc_mean, acc_std, params, n))
    entries.sort(reverse=True)

    header = ["Model", "Params", "Accuracy", "Macro-F1", "Fit (s)", "Runs"]
    body = []
    for _, m, rs, acc_mean, acc_std, params, n in entries:
        fit = np.mean([r.get("fit_seconds", 0) for r in rs])
        body.append([
            m.replace("_", r"\_" if latex else "_"),
            f"{params:,}" if params and params > 0 else "--",
            f(acc_mean, acc_std),
            f(*_agg(rs, "macro_f1")[:2]),
            f"{fit:.2f}",
            str(n),
        ])

    caption = ("Model comparison at fixed protocol and feature group. "
               "Parameter counts are trainable weights for neural models and "
               "total tree nodes for forests.")
    return _render(header, body, caption, "tab:models", latex)


# ---------------------------------------------------------------------------
# Table IV - normalisation
# ---------------------------------------------------------------------------


def table_normalisation(rows: list[dict], latex: bool = False) -> str:
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["normalisation"], r["feature_group"])].append(r)

    norms = sorted({k[0] for k in by})
    groups = sorted({k[1] for k in by})
    f = _fmt if latex else _fmt_md

    header = ["Normalisation"] + [g.replace("_", r"\_" if latex else "_")
                                  for g in groups]
    body = []
    for nm in norms:
        row = [nm.replace("_", r"\_" if latex else "_")]
        for g in groups:
            rs = by.get((nm, g), [])
            row.append(f(*_agg(rs, "accuracy")[:2]) if rs else "--")
        body.append(row)

    caption = ("Effect of coordinate normalisation. Raw MediaPipe coordinates "
               "are relative to the image frame and therefore encode signer "
               "position and camera distance.")
    return _render(header, body, caption, "tab:normalisation", latex)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render(header: list[str], body: list[list[str]], caption: str,
            label: str, latex: bool) -> str:
    if not latex:
        cap = caption.replace("\\%", "%").replace("\\texttt{", "`")
        cap = cap.replace("$\\pm$", "±").replace("}", "`") if "`" in cap else cap
        lines = ["| " + " | ".join(header) + " |",
                 "|" + "|".join(["---"] * len(header)) + "|"]
        lines += ["| " + " | ".join(r) + " |" for r in body]
        return "\n".join(lines) + f"\n\n_{cap}_\n"

    cols = "l" + "r" * (len(header) - 1)
    out = [r"\begin{table}[t]", r"\centering",
           rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
           " & ".join(header) + r" \\", r"\midrule"]
    out += [" & ".join(r) + r" \\" for r in body]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


TABLES = {
    "protocol": table_protocol,
    "ablation": table_ablation,
    "models": table_models,
    "normalisation": table_normalisation,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="paper/tables")
    ap.add_argument("--tables", nargs="*", default=sorted(TABLES))
    args = ap.parse_args()

    rows = []
    for p in args.results:
        rows.extend(_load(p))
    if not rows:
        raise SystemExit("no results rows found")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name in args.tables:
        fn = TABLES[name]
        md, tex = fn(rows, latex=False), fn(rows, latex=True)
        (out / f"{name}.md").write_text(md, encoding="utf-8")
        (out / f"{name}.tex").write_text(tex, encoding="utf-8")
        print(f"\n### {name}\n")
        print(md)

    print(f"\nWrote {len(args.tables)} tables (.md and .tex) to {out}/")


if __name__ == "__main__":
    main()

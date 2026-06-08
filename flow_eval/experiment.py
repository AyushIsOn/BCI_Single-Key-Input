"""
experiment.py
=============
The noise-robustness experiment for the Flow Dial paper.

Central hypothesis
------------------
H1: The continuous-flow mechanic degrades MORE GRACEFULLY than discrete
    point-and-dwell as cursor noise increases (its WPM and error curves stay
    flatter), because likely characters occupy wide, hard-to-miss arcs.
H2: Removing the language model hurts the Flow Dial substantially, showing the
    mechanic and the predictive arc-sizing are coupled (and implying the design
    will struggle on low-redundancy input such as code -- an honest limitation).

Design
------
- Sweep cursor noise sigma across a realistic range.
- For each sigma, run three conditions:
    * flow + prediction   (the proposed system)
    * flow - prediction   (ablation: arcs sized by unigram freq only)
    * dwell (baseline)    (conventional fixed-key hold-to-select)
- Repeat each cell over N_SEEDS independent seeds -> mean +/- 95% CI.

Outputs
-------
- results_sweep.csv                : raw table for the paper
- fig_wpm_vs_noise.png             : effective WPM vs sigma (with CIs)
- fig_error_vs_noise.png           : character error rate vs sigma (with CIs)
- prints a markdown summary table
"""

from __future__ import annotations

import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulate import CursorParams, run

# noise levels: from a clean cursor to a very degraded BCI signal
SIGMAS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20]
N_SEEDS = 12

CONDITIONS = [
    ("flow",  True,  "Flow Dial (+prediction)", "#2ea043", "o-"),
    ("flow",  False, "Flow Dial (-prediction)", "#56a3f5", "s--"),
    ("dwell", True,  "Point-dwell baseline",    "#f85149", "^-"),
]


def mean_ci(values):
    """Mean and 95% CI half-width (normal approx)."""
    a = np.array(values, dtype=float)
    m = a.mean()
    if len(a) > 1:
        se = a.std(ddof=1) / np.sqrt(len(a))
        ci = 1.96 * se
    else:
        ci = 0.0
    return m, ci


def run_sweep():
    rows = []
    # rows: dict per (condition, sigma) with wpm/cer mean+ci
    for mech, pred, label, _, _ in CONDITIONS:
        for sigma in SIGMAS:
            wpms, cers = [], []
            for seed in range(N_SEEDS):
                r = run(mech, CursorParams(sigma=sigma),
                        use_prediction=pred, seed=seed)
                wpms.append(r.wpm)
                cers.append(r.cer * 100.0)  # to percent
            wm, wci = mean_ci(wpms)
            cm, cci = mean_ci(cers)
            rows.append({
                "condition": label, "mechanic": mech, "prediction": pred,
                "sigma": sigma,
                "wpm_mean": round(wm, 2), "wpm_ci": round(wci, 2),
                "cer_mean": round(cm, 2), "cer_ci": round(cci, 2),
            })
    return rows


def write_csv(rows, path="results_sweep.csv"):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _series(rows, label, key):
    pts = [(r["sigma"], r[f"{key}_mean"], r[f"{key}_ci"])
           for r in rows if r["condition"] == label]
    pts.sort()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    es = [p[2] for p in pts]
    return xs, ys, es


def plot_wpm(rows, path="fig_wpm_vs_noise.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mech, pred, label, color, style in CONDITIONS:
        xs, ys, es = _series(rows, label, "wpm")
        ax.errorbar(xs, ys, yerr=es, fmt=style, color=color, capsize=3,
                    label=label, linewidth=2, markersize=6)
    ax.set_xlabel("Cursor noise  σ  (worse BCI signal →)")
    ax.set_ylabel("Effective words per minute")
    ax.set_title("Text-entry speed vs. cursor noise")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_error(rows, path="fig_error_vs_noise.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mech, pred, label, color, style in CONDITIONS:
        xs, ys, es = _series(rows, label, "cer")
        ax.errorbar(xs, ys, yerr=es, fmt=style, color=color, capsize=3,
                    label=label, linewidth=2, markersize=6)
    ax.set_xlabel("Cursor noise  σ  (worse BCI signal →)")
    ax.set_ylabel("Character error rate (%)")
    ax.set_title("Error rate vs. cursor noise")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def markdown_summary(rows):
    # compact table at a few representative sigmas
    show = [0.04, 0.10, 0.16, 0.20]
    lines = ["| Condition | " + " | ".join(f"σ={s}" for s in show) + " |",
             "|" + "---|" * (len(show) + 1)]
    for mech, pred, label, _, _ in CONDITIONS:
        cells = []
        for s in show:
            r = next(r for r in rows if r["condition"] == label and r["sigma"] == s)
            cells.append(f"{r['wpm_mean']:.1f} wpm / {r['cer_mean']:.0f}%")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Running sweep: {len(SIGMAS)} noise levels x {len(CONDITIONS)} "
          f"conditions x {N_SEEDS} seeds ...")
    rows = run_sweep()
    write_csv(rows)
    plot_wpm(rows)
    plot_error(rows)
    print("\nWrote: results_sweep.csv, fig_wpm_vs_noise.png, fig_error_vs_noise.png\n")
    print("Summary (effective WPM / character error rate):\n")
    print(markdown_summary(rows))

    # headline numbers for the paper narrative
    def get(label, s, key):
        return next(r[f"{key}_mean"] for r in rows
                    if r["condition"] == label and r["sigma"] == s)
    hi = 0.16
    flow = get("Flow Dial (+prediction)", hi, "wpm")
    dwell = get("Point-dwell baseline", hi, "wpm")
    print(f"\nAt high noise (σ={hi}): Flow={flow:.1f} wpm vs dwell={dwell:.1f} wpm "
          f"-> {flow/dwell:.1f}x faster")
    fe = get("Flow Dial (+prediction)", hi, "cer")
    de = get("Point-dwell baseline", hi, "cer")
    print(f"                        Flow error={fe:.0f}% vs dwell error={de:.0f}%")

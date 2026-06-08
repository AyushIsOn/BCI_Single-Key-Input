"""
results.py
==========
Top-level experiment runner. Produces the comparison table and figures for the
paper, combining:

  (A) Fitts' analytical model        -> idealized expert speed + reachability
  (B) Noisy-cursor Monte Carlo       -> realistic speed under BCI noise
  (C) Layer-switch fairness penalty  -> the crucial correction

The fairness penalty
--------------------
The prose / QWERTY layouts do not expose code symbols ( ) { } ; = etc. on their
base surface. On real soft keyboards those characters live behind a "123/#+="
layer toggle (visible in the Irisbond design). Every time the next code char is
NOT on the base layer, the user must:
    toggle layer  ->  select symbol  ->  toggle back
We charge a fixed `LAYER_SWITCH_COST` (in equivalent character-times) for the
fraction of code characters that are off-base. The code-optimized radial layout
exposes all of them at base level, so it pays no penalty. This converts the
"100% vs 78% vs 73% coverage" finding into an apples-to-apples effective WPM.

Outputs:
  results.csv            -- machine-readable table
  results_table.md       -- markdown table for the paper
  fig_layouts.png        -- the three keyboard geometries
  fig_effective_wpm.png  -- the headline bar chart
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from corpus import build_profiles, ALL_KEYS_CODE
from layouts import build_all_layouts, qwerty_grid, radial_prose, radial_code
from fitts_model import FittsParams, evaluate_layout_fitts
from montecarlo import CursorParams, simulate_layout


# how many extra character-equivalent actions a layer switch round-trip costs
LAYER_SWITCH_COST = 2.0   # toggle in + toggle out (each ~ one selection)


def base_layer_coverage(layout, profile) -> float:
    """Fraction of code-character mass reachable on the layout's base surface."""
    total = sum(profile.unigram.values())
    covered = sum(c for ch, c in profile.unigram.items() if ch in layout)
    return covered / total if total else 0.0


@dataclass
class CombinedRow:
    layout: str
    fitts_wpm: float
    fitts_throughput: float
    base_coverage: float
    mc_wpm_raw: float
    mc_misselect: float
    mc_kspc: float
    effective_wpm: float   # MC speed adjusted for layer-switch penalty
    effective_kspc: float


def compute_effective(mc, coverage, mean_tpc) -> tuple[float, float]:
    """
    Apply the layer-switch penalty. off_base = (1 - coverage) of characters need
    a round-trip costing LAYER_SWITCH_COST selection-times each.

    extra_time_per_char = off_base * LAYER_SWITCH_COST * mean_tpc
    effective_tpc       = mean_tpc + extra_time_per_char
    """
    off_base = 1.0 - coverage
    extra_time = off_base * LAYER_SWITCH_COST * mean_tpc
    eff_tpc = mean_tpc + extra_time
    eff_wpm = 60.0 / eff_tpc / 5.0
    eff_kspc = mc.kspc + off_base * LAYER_SWITCH_COST
    return eff_wpm, eff_kspc


def run(corpus_name: str = "code") -> list[CombinedRow]:
    profs = build_profiles()
    profile = profs[corpus_name]
    layouts = build_all_layouts(profs)

    fitts_params = FittsParams()
    cursor_params = CursorParams()

    rows: list[CombinedRow] = []
    for name, lay in layouts.items():
        fitts = evaluate_layout_fitts(lay, profile, fitts_params, name)
        mc = simulate_layout(lay, profile, cursor_params, name, n_chars=1500)
        coverage = base_layer_coverage(lay, profile)
        eff_wpm, eff_kspc = compute_effective(mc, coverage, mc.mean_time_per_char)
        rows.append(CombinedRow(
            layout=name,
            fitts_wpm=round(fitts.wpm, 1),
            fitts_throughput=round(fitts.throughput_bits_s, 2),
            base_coverage=round(coverage, 3),
            mc_wpm_raw=round(mc.wpm_effective, 1),
            mc_misselect=round(mc.misselect_rate, 3),
            mc_kspc=round(mc.kspc, 2),
            effective_wpm=round(eff_wpm, 1),
            effective_kspc=round(eff_kspc, 2),
        ))
    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_csv(rows, path="results.csv"):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def write_markdown(rows, path="results_table.md"):
    headers = ["Layout", "Fitts WPM", "TP (b/s)", "Base cov.",
               "MC WPM (raw)", "Mis-sel %", "MC KSPC",
               "Effective WPM", "Effective KSPC"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join([
            r.layout,
            f"{r.fitts_wpm}",
            f"{r.fitts_throughput}",
            f"{100*r.base_coverage:.0f}%",
            f"{r.mc_wpm_raw}",
            f"{100*r.mc_misselect:.1f}",
            f"{r.mc_kspc}",
            f"**{r.effective_wpm}**",
            f"{r.effective_kspc}",
        ]) + " |")
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_layouts(path="fig_layouts.png"):
    profs = build_profiles()
    layouts = {
        "QWERTY grid (Neuralink-style)": qwerty_grid(),
        "Radial PROSE (Irisbond-style)": radial_prose(profs["prose"]),
        "Radial CODE (proposed)": radial_code(profs["code"]),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, lay) in zip(axes, layouts.items()):
        for ch, (x, y) in lay.items():
            label = "␣" if ch == " " else ch
            is_symbol = ch in "_.,:;=()[]{}<>\"'/*+-!&|#$%@\\`?"
            color = "#d62728" if is_symbol else "#1f77b4"
            ax.scatter([x], [y], s=260, c=color, alpha=0.25, edgecolors="none")
            ax.text(x, y, label, ha="center", va="center", fontsize=8)
        circle = plt.Circle((0, 0), 1.0, fill=False, color="gray", ls="--", alpha=0.4)
        ax.add_patch(circle)
        ax.scatter([0], [0], marker="+", c="black", s=120)  # cursor rest
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.axis("off")
    fig.suptitle("Keyboard geometries (blue = letters/digits, red = code symbols)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_effective_wpm(rows, path="fig_effective_wpm.png"):
    names = [r.layout for r in rows]
    raw = [r.mc_wpm_raw for r in rows]
    eff = [r.effective_wpm for r in rows]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, raw, w, label="MC WPM (base chars only)", color="#9ecae1")
    ax.bar(x + w/2, eff, w, label="Effective WPM (incl. layer-switch penalty)",
           color="#d62728")
    for i, v in enumerate(eff):
        ax.text(x[i] + w/2, v + 0.3, f"{v}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylabel("Words per minute")
    ax.set_title("Code-entry throughput on a noisy single cursor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rows = run("code")
    write_csv(rows)
    table = write_markdown(rows)
    plot_layouts()
    plot_effective_wpm(rows)

    print("\n================  CODE-ENTRY EVALUATION  ================\n")
    print(table)
    print("Wrote: results.csv, results_table.md, fig_layouts.png, fig_effective_wpm.png")

    best = max(rows, key=lambda r: r.effective_wpm)
    print(f"\nBest effective code-entry layout: {best.layout} "
          f"({best.effective_wpm} WPM, KSPC {best.effective_kspc})")

# A Code-Optimized Circular Keyboard for Cursor-Driven / BCI Text Entry

A hardware-free simulation harness for evaluating a **code-optimized concentric
radial keyboard** against (a) a Neuralink-style full-width QWERTY-swipe keyboard
and (b) a prose-optimized radial keyboard (Irisbond / Scott Morgan Foundation
style). Built as a research artifact for an India HCI submission.

## Motivation

Cursor-driven BCIs (e.g. Neuralink) and gaze keyboards are fundamentally **noisy
2D pointing devices**. Existing radial/gaze keyboards (and predictive engines
like Dasher) are optimized for **natural-language** statistics. **Code has very
different statistics**: symbols such as `. ( ) ; = { }` are high-frequency, and
identifiers (`camelCase`, `snake_case`) lack the redundancy that word-prediction
relies on. Predictive text papers over this for prose, but the underlying
*layout* is wrong for code.

**Hypothesis:** a radial layout whose ring assignment is derived from *code*
token frequency (frequent symbols + letters in the fast inner rings, all symbols
on the base layer) beats both baselines for code entry on a noisy single cursor.

## Method (triangulated, no BCI required)

1. **Fitts' law analytical model** (`fitts_model.py`) - expected expert entry
   speed as the bigram-frequency-weighted sum of movement times. Standard
   technique for comparing soft-keyboard layouts without users (Zhai et al.;
   MacKenzie & Soukoreff).
2. **Noisy-cursor Monte Carlo** (`montecarlo.py`) - simulates a single BCI cursor
   (proportional control + Gaussian motor noise, dwell-to-commit). Captures
   mis-selections + correction overhead that Fitts cannot.
3. **Layer-switch fairness penalty** (`results.py`) - charges prose/QWERTY
   layouts for the `123`/symbol-layer round-trip needed to reach code characters
   they don't expose at base level. Converts a coverage gap into effective WPM.

## Files

| File | Purpose |
|------|---------|
| `corpus.py` | Builds CODE and PROSE character unigram/bigram frequency profiles |
| `layouts.py` | Generates 2D key coordinates for the 3 layouts (unit-circle space) |
| `fitts_model.py` | Fitts' law expert-speed model + reachability coverage |
| `montecarlo.py` | Noisy single-cursor simulation (effective WPM, KSPC, mis-select rate) |
| `results.py` | Runs everything, applies fairness penalty, writes table + figures |
| `inspect_rings.py` | Shows which keys land in the fast inner rings |
| `visualizer.html` | **Interactive layout viewer** &mdash; open in any browser (no server needed) |

## See the layout

Open **`visualizer.html`** in any web browser (just double-click it). It renders
all three layouts from the exact simulation coordinates, lets you switch between
them, sizes each key by its code frequency, and animates the single cursor's
travel path as you type code into the demo box. The proposed radial-code layout
is shown first.

## Reproduce

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python numpy scipy matplotlib
.venv/bin/python results.py
```

Outputs: `results.csv`, `results_table.md`, `fig_layouts.png`,
`fig_effective_wpm.png`.

## Headline result (code corpus, gaze-like / ~8 BPS cursor params)

| Layout | Base coverage | Effective WPM | Effective KSPC |
|---|---|---|---|
| QWERTY grid (Neuralink-style swipe) | 84% | 13.6 | 1.63 |
| Radial PROSE (Irisbond-style) | 87% | 15.9 | 1.31 |
| **Radial CODE (proposed)** | **100%** | **18.8** | **1.16** |

The code-optimized radial layout is ~38% faster than the QWERTY-swipe baseline
for code entry, with the lowest keystrokes-per-character, because frequent code
symbols (`. ( ) = ; , :`) sit in the fast inner rings instead of behind a layer
toggle.

## Important caveats (state these in the paper)

- Device constants (`FittsParams`, `CursorParams`) are **assumptions**
  parameterized to BCI-plausible values, not measured from a real implant. Run a
  sensitivity sweep for camera-ready.
- The embedded corpus is small; swap in a larger GitHub corpus for final numbers
  (the API does not change).
- This is a **single-cursor** model. A two-cursor / bimanual BCI is future work.
- LLM/VLM "simulated users" should only be used for qualitative cognitive
  walkthroughs, not motor-performance claims (see related work in the paper).

## Related work anchors

- **Dasher** (Ward & MacKay) - continuous-pointing + language model, for prose.
- **Irisbond / Scott Morgan Foundation gaze keyboard** - frequency-optimized
  single radial ring + word prediction, for prose.
- **C-QWERTY / Virtual Radial Keyboard** - radial text entry, learning-curve
  tradeoffs.
- Neuralink Webgrid BPS - the pointing-throughput ceiling this design targets.


---

# Part II: The Flow Dial — a no-click, no-stop circular keyboard

A second, distinct exploration in this repo. Starting from first principles
about what a Neuralink cursor *is* (a noisy 2-D point whose hardest operations
are **holding still** and **clicking**), we designed the **Flow Dial**: a
continuous, selection-free circular keyboard. Candidates occupy angular arcs
whose width grows with language-model probability; you commit a character by
**flowing outward** through its arc — no dwell, no click.

## Files

| File | Purpose |
|------|---------|
| `concepts.html` | Animated gallery of 12 circular-keyboard concepts across 4 design families |
| `flow_dial.html` | **Drivable prototype** + built-in pilot study mode (logs trials, exports CSV) |
| `flow_eval/corpus_en.py` | English corpus + bigram language model for simulation |
| `flow_eval/simulate.py` | Shared noisy-cursor model + Flow Dial and point-dwell mechanics |
| `flow_eval/experiment.py` | Noise-robustness sweep → CSV + two figures |
| `flow_eval/PAPER_evaluation.md` | Paper-ready methods + results write-up |

## Try it

- **Drive the keyboard:** open `flow_dial.html`. Rest in the center, glide
  outward to a letter, cross the ring to commit. Toggle **tremor** and
  **prediction** to feel the design's robustness and its dependence on the
  language model.
- **Run a pilot:** click *Start study* in the prototype, transcribe the prompted
  phrases, then *Export CSV* (per-trial WPM/CER/KSPC + a full keystroke log).

## Reproduce the simulation

```bash
cd flow_eval
../.venv/bin/python experiment.py   # reuse the Part I venv (numpy, matplotlib)
```

## Headline result — graceful degradation under cursor noise

| Technique | σ=0.04 | σ=0.10 | σ=0.16 |
|---|---|---|---|
| Flow Dial (+prediction) | **36.8** wpm / 2% err | **31.1** wpm / 19% err | **26.7** wpm / 37% err |
| Flow Dial (−prediction) | 35.0 wpm / 6% err | 27.8 wpm / 33% err | 24.2 wpm / 51% err |
| Point-dwell baseline | 31.4 wpm / 16% err | 3.0 wpm / 46% err | 0.9 wpm / 75% err |

The point-dwell baseline collapses as cursor noise rises; the Flow Dial declines
gently and keeps the lowest error rate. The prediction ablation shows the
language model is essential as noise grows — which also predicts the design's
honest weakness on low-redundancy input such as **code**. Full analysis in
`flow_eval/PAPER_evaluation.md`.

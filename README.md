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

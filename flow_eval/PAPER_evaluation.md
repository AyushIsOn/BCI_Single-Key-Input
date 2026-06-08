# The Flow Dial: Evaluation Methods and Results

*Draft evaluation section for the India HCI submission. Numbers are produced by
`experiment.py` and are fully reproducible (`results_sweep.csv`).*

---

## 1. The Flow Dial, in one paragraph

The Flow Dial is a circular text-entry method for a single noisy cursor (e.g. a
Neuralink-style intracortical cursor). Unlike conventional gaze/BCI keyboards
that require the user to *acquire a target and then select it* (dwell or click),
the Flow Dial is **continuous and selection-free**: candidate characters occupy
angular *arcs* around a central rest zone, and the user commits a character
simply by **flowing outward** through its arc until the cursor crosses an outer
ring. Crucially, each arc's **angular width grows with the language-model
probability** of that character given the previous one, so likely next-letters
become wide, hard-to-miss targets and unlikely ones stay narrow but reachable.
There is no dwell time and no discrete click — the two operations a brain-driven
cursor performs worst.

## 2. Why evaluate in simulation (and what we are NOT claiming)

We do not have access to an implanted BCI, and an empirical implant study is
out of scope for a work-in-progress contribution. We therefore evaluate with a
**noisy-cursor simulation** plus an **instrumented surrogate-cursor pilot**
(mouse / trackpad / eye-tracker). This is an established practice for
text-entry method design (Fitts-law and Monte-Carlo keyboard evaluation;
BCI-emulation pilots). We explicitly do **not** claim implant-validated WPM.
We claim a **relative, mechanistic** result: how the *mechanic* behaves as
cursor noise grows, holding the input device model fixed.

## 3. Apparatus: a shared noisy-cursor model

Both the Flow Dial and the baseline are driven by the **same** cursor. The
cursor is a 2-D point pulled toward an on-screen goal by a proportional
controller with additive Gaussian motor noise each timestep:

```
pos += gain * (goal - pos) * dt  +  N(0, sigma^2)
```

`sigma` is the single knob we sweep; it stands in for **degrading BCI signal
quality**. Holding the cursor model fixed across mechanics makes the comparison
fair: any difference is attributable to the *interaction technique*, not the
input device.

## 4. Conditions and design

A 3 (technique) x 8 (noise) design, each cell repeated over **12 independent
seeds** (mean +/- 95% CI):

| Technique | Description |
|---|---|
| **Flow Dial (+prediction)** | Proposed system; arc width ∝ √P(next \| prev) from a bigram LM |
| **Flow Dial (−prediction)** | Ablation; arcs sized by unigram frequency only |
| **Point-dwell baseline** | 27 fixed keys on a ring; acquire-and-hold (dwell) selection |

Noise levels: `sigma ∈ {0.02 … 0.20}` (clean cursor → severely degraded). The
ablation isolates how much of the Flow Dial's behaviour depends on the language
model versus the geometry alone.

**Task.** Transcribe a fixed set of common English phrases. **Measures:**
effective words-per-minute (5 char/word), character error rate (CER, wrong
commits / intended), and — in the human pilot — keystrokes-per-character (KSPC).

## 5. Results

### 5.1 Speed degrades gracefully (H1 supported)

| Technique | σ=0.04 | σ=0.10 | σ=0.16 | σ=0.20 |
|---|---|---|---|---|
| Flow Dial (+pred) | **36.8** wpm | **31.1** wpm | **26.7** wpm | **24.3** wpm |
| Flow Dial (−pred) | 35.0 wpm | 27.8 wpm | 24.2 wpm | 22.6 wpm |
| Point-dwell | 31.4 wpm | 3.0 wpm | 0.9 wpm | 0.9 wpm |

The baseline is competitive only when the cursor is nearly clean. As noise
rises it **collapses** (dwell selection repeatedly fails to hold a stable
target), falling below 1 wpm by σ=0.16. The Flow Dial stays in a usable band,
declining gently from ~37 to ~24 wpm across the whole range. At σ=0.16 the Flow
Dial is roughly **29× faster** than dwell. See `fig_wpm_vs_noise.png`.

### 5.2 Errors stay bounded (H1 supported)

| Technique | σ=0.04 | σ=0.10 | σ=0.16 | σ=0.20 |
|---|---|---|---|---|
| Flow Dial (+pred) | **2.1%** | **19.5%** | **37.5%** | 48.0% |
| Flow Dial (−pred) | 6.4% | 32.6% | 51.2% | 60.6% |
| Point-dwell | 15.8% | 46.2% | 74.8% | 78.5% |

At every noise level the Flow Dial with prediction has the lowest error rate.
The baseline's error climbs steeply because tremor pushes the cursor into
neighbouring fixed keys. See `fig_error_vs_noise.png`.

### 5.3 Prediction is doing real work (H2 supported)

The +prediction and −prediction curves separate increasingly with noise: at
σ=0.10 prediction cuts CER from **32.6% → 19.5%** and lifts WPM from 27.8 → 31.1.
This confirms the mechanic and the predictive arc-sizing are **coupled** — the
geometry alone helps, but the language model is what keeps error bounded as the
signal degrades.

## 6. The honest limitation this exposes

Because the Flow Dial leans on next-character predictability, it will degrade on
**low-redundancy input** — exactly the case for source code (identifiers,
symbols, `camelCase`). The −prediction ablation is a direct proxy for that
worst case, and it is meaningfully worse. We therefore position the Flow Dial as
a strong design for **natural-language** entry on a noisy cursor, and flag
code/structured entry as future work (where a different arc-sizing model, or a
hybrid with an explicit symbol layer, would be required).

## 7. Surrogate-cursor pilot protocol (instrument provided)

`flow_dial.html` includes a **study mode**: it prompts the standard phrases,
logs every commit (intended vs. actual character, inter-commit time, correctness)
and per-phrase WPM/CER/KSPC, and exports a CSV. Recommended pilot: N=3–5
participants drive the dial with a mouse, a trackpad, and (if available) an
eye-tracker, with prediction and a "tremor" toggle crossed as within-subject
conditions. Even a small pilot converts this from a pure-simulation contribution
into one with human data and subjective workload (e.g. NASA-TLX).

## 8. Reproduce

```bash
cd flow_eval
../<venv>/bin/python experiment.py     # writes CSV + both figures
```

Artifacts: `results_sweep.csv`, `fig_wpm_vs_noise.png`, `fig_error_vs_noise.png`.
Cursor and technique parameters live at the top of `simulate.py` and `experiment.py` and
should be reported as assumptions; a sensitivity sweep over `gain`, dwell time,
and arc-width exponent is recommended for the camera-ready.

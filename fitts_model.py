"""
fitts_model.py
==============
Analytical (predictive) evaluation of keyboard layouts using Fitts' law.

This is the standard, peer-reviewed method for comparing soft-keyboard layouts
WITHOUT running human trials (Zhai et al., "Physics-based / Metropolis keyboard";
MacKenzie & Soukoreff soft-keyboard models). It estimates *expert* entry speed
as the frequency-weighted sum of movement times between consecutive keys.

Fitts' law (Shannon formulation, MacKenzie 1992):

        MT = a + b * log2( D / W + 1 )                       [seconds]

  D : distance from previous key to target key
  W : effective target width (key size)
  a, b : empirically-derived device constants (intercept, slope)

We sum MT over every bigram (c_i -> c_j), weighted by that bigram's probability
in the corpus, to get the expected per-character movement time. From that we
derive characters-per-minute (CPM) and words-per-minute (WPM, 5 chars/word).

KSPC (keystrokes per character) here is 1.0 for direct selection (no prediction
/ no disambiguation). The prediction-enabled variants are handled separately in
the results layer so the layout effect is isolated from the language-model
effect (the whole point of the paper: does the *layout* help code, independent
of prediction?).

Device constants
----------------
Defaults model a moderately noisy indirect cursor (gaze / BCI-like), slower than
a hand on a mouse. They are parameters, not magic numbers; the harness reports
which constants were used so results are reproducible and can be re-derived for
a real device once data exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from corpus import FrequencyProfile

Layout = dict[str, tuple[float, float]]


@dataclass
class FittsParams:
    a: float = 0.20          # intercept (s) -- fixed reaction/commit overhead
    b: float = 0.20          # slope (s/bit) -- higher => slower device
    key_width: float = 0.22  # effective target width W in normalized units
    name: str = "gaze-like"


def _distance(p: tuple[float, float], q: tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def movement_time(d: float, params: FittsParams) -> float:
    """Fitts' law Shannon formulation. d is center-to-center distance."""
    return params.a + params.b * math.log2(d / params.key_width + 1.0)


@dataclass
class FittsResult:
    layout_name: str
    corpus_name: str
    params_name: str
    mean_mt_per_char: float   # seconds
    cpm: float                # characters per minute
    wpm: float                # words per minute (5 chars/word)
    throughput_bits_s: float  # mean index of difficulty / mean MT
    covered_bigram_mass: float  # fraction of bigram prob mass actually modeled


def evaluate_layout_fitts(
    layout: Layout,
    profile: FrequencyProfile,
    params: FittsParams,
    layout_name: str = "",
) -> FittsResult:
    """
    Expected movement time per character = sum over bigrams of
        P(c_i, c_j) * MT(distance(c_i, c_j))
    normalized by the bigram probability mass we can actually place on the layout
    (some corpus chars may be absent from a given layout's universe, e.g. code
    symbols are not on the prose keyboard -> those bigrams are excluded and we
    report the covered mass for honesty).
    """
    total_bigrams = sum(profile.bigram.values())
    if total_bigrams == 0:
        raise ValueError("empty bigram profile")

    weighted_mt = 0.0
    weighted_id = 0.0   # weighted index of difficulty (bits)
    covered = 0
    for (c_i, c_j), count in profile.bigram.items():
        if c_i not in layout or c_j not in layout:
            continue  # this transition can't be performed on this layout
        d = _distance(layout[c_i], layout[c_j])
        mt = movement_time(d, params)
        idf = math.log2(d / params.key_width + 1.0)
        p = count / total_bigrams
        weighted_mt += p * mt
        weighted_id += p * idf
        covered += count

    covered_mass = covered / total_bigrams
    # renormalize by covered mass so MT/char is a fair per-(modeled-)char value
    if covered_mass > 0:
        mean_mt = weighted_mt / covered_mass
        mean_id = weighted_id / covered_mass
    else:
        mean_mt = float("inf")
        mean_id = 0.0

    cpm = 60.0 / mean_mt if mean_mt > 0 else 0.0
    wpm = cpm / 5.0
    throughput = mean_id / mean_mt if mean_mt > 0 else 0.0

    return FittsResult(
        layout_name=layout_name,
        corpus_name=profile.name,
        params_name=params.name,
        mean_mt_per_char=mean_mt,
        cpm=cpm,
        wpm=wpm,
        throughput_bits_s=throughput,
        covered_bigram_mass=covered_mass,
    )


if __name__ == "__main__":
    from corpus import build_profiles
    from layouts import build_all_layouts

    profs = build_profiles()
    layouts = build_all_layouts(profs)
    params = FittsParams()

    print(f"Fitts device profile: {params.name} (a={params.a}, b={params.b}, W={params.key_width})\n")
    print(f"{'layout':<14}{'corpus':<8}{'WPM':>7}{'CPM':>8}{'MT/char':>9}{'TP(b/s)':>9}{'cover':>8}")
    print("-" * 62)
    # Evaluate each layout against the CODE corpus (the scenario of interest)
    for lname, lay in layouts.items():
        res = evaluate_layout_fitts(lay, profs["code"], params, lname)
        print(f"{lname:<14}{'code':<8}{res.wpm:7.1f}{res.cpm:8.1f}"
              f"{res.mean_mt_per_char:9.3f}{res.throughput_bits_s:9.2f}"
              f"{100*res.covered_bigram_mass:7.0f}%")

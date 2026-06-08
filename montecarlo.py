"""
montecarlo.py
=============
Noisy-cursor Monte Carlo simulation of text entry.

Why this complements the Fitts' model
--------------------------------------
Fitts' law gives an *idealized expert* movement time and assumes every
selection lands on the intended key. A real BCI cursor is NOISY: the decoded
velocity/position jitters, so the cursor can land on a neighbouring key and
cause a mis-selection that must be corrected (backspace + retype). This module
simulates that process directly, which is what makes a *compact* layout
(short travel, well-separated targets) win under noise.

Cursor model
------------
We model selection of a single target as a 2D point-to-point movement with a
proportional controller plus Gaussian motor noise (a standard, defensible
abstraction of intracortical-cursor behaviour):

    pos += k * (target - pos) * dt  +  noise

The "click"/commit happens when the cursor dwells inside some key's hit-region
for the commit duration. Whichever key region the cursor commits in is the
selected key -- if it isn't the intended one, that's a mis-selection.

Noise is parameterized by `noise_sigma`, which we tie to an approximate BCI
throughput level (e.g. Neuralink ~8 bits/s) so reviewers can see the assumption.

Outputs per (layout, corpus): effective WPM (including correction overhead),
mis-selection rate, and KSPC (keystrokes per character, >1 due to corrections).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

from corpus import FrequencyProfile

Layout = dict[str, tuple[float, float]]


@dataclass
class CursorParams:
    gain: float = 12.0         # proportional controller gain k
    dt: float = 0.02           # simulation timestep (s)
    noise_sigma: float = 0.06  # motor noise std-dev per step (normalized units)
    key_radius: float = 0.11   # hit-region radius (half of Fitts key_width)
    commit_dwell: float = 0.10 # dwell time inside a region to commit (s)
    max_time: float = 4.0      # safety timeout per character (s)
    name: str = "bci-8bps"


@dataclass
class MCResult:
    layout_name: str
    corpus_name: str
    params_name: str
    wpm_effective: float
    misselect_rate: float
    kspc: float
    mean_time_per_char: float
    n_trials: int


def _nearest_key(pos, items_xy, items_keys):
    """Return key whose center is nearest to pos (the region the cursor is in)."""
    d2 = np.sum((items_xy - pos) ** 2, axis=1)
    j = int(np.argmin(d2))
    return items_keys[j], math.sqrt(d2[j])


def _simulate_one_selection(start, target_xy, layout_xy, layout_keys, p: CursorParams, rng):
    """
    Drive a noisy cursor from `start` toward `target_xy`. Commit when the cursor
    stays within key_radius of SOME key for commit_dwell seconds. Return
    (committed_key, end_pos, elapsed_time).
    """
    pos = np.array(start, dtype=float)
    target = np.array(target_xy, dtype=float)
    t = 0.0
    dwell_key = None
    dwell_time = 0.0
    steps_for_commit = p.commit_dwell

    while t < p.max_time:
        # proportional control toward target + Gaussian motor noise
        pos = pos + p.gain * (target - pos) * p.dt + rng.normal(0, p.noise_sigma, size=2)
        t += p.dt

        key, dist = _nearest_key(pos, layout_xy, layout_keys)
        if dist <= p.key_radius:
            if key == dwell_key:
                dwell_time += p.dt
            else:
                dwell_key = key
                dwell_time = p.dt
            if dwell_time >= steps_for_commit:
                return key, pos, t
        else:
            dwell_key = None
            dwell_time = 0.0

    # timed out: commit nearest key
    key, _ = _nearest_key(pos, layout_xy, layout_keys)
    return key, pos, t


def simulate_layout(
    layout: Layout,
    profile: FrequencyProfile,
    params: CursorParams,
    layout_name: str = "",
    n_chars: int = 1500,
    seed: int = 7,
) -> MCResult:
    """
    Sample target characters according to the corpus unigram distribution and
    simulate typing each one. Mis-selections incur a correction: one backspace
    movement + a re-attempt (counted in time and KSPC).
    """
    rng = np.random.default_rng(seed)

    # restrict to characters that exist on this layout (others need a layer
    # switch which we account for separately in results.py)
    keys = [k for k in layout.keys()]
    layout_xy = np.array([layout[k] for k in keys], dtype=float)

    # build sampling distribution over the characters present on the layout
    present = [c for c in profile.unigram if c in layout]
    weights = np.array([profile.unigram[c] for c in present], dtype=float)
    if weights.sum() == 0:
        raise ValueError("no overlap between corpus and layout")
    weights = weights / weights.sum()

    backspace_key = None  # we model backspace as returning to center (origin)

    total_time = 0.0
    total_keystrokes = 0
    misselects = 0
    produced = 0

    pos = np.array([0.0, 0.0])  # cursor starts at center rest
    targets = rng.choice(len(present), size=n_chars, p=weights)

    for ti in targets:
        ch = present[ti]
        target_xy = layout[ch]

        committed, pos, elapsed = _simulate_one_selection(
            pos, target_xy, layout_xy, keys, params, rng
        )
        total_time += elapsed
        total_keystrokes += 1

        if committed != ch:
            # mis-selection: user notices, hits backspace (move toward center),
            # then re-attempts the intended key.
            misselects += 1
            # backspace movement cost: travel from current pos back toward center
            bs_target = np.array([0.0, 0.0])
            _, pos, bs_time = _simulate_one_selection(
                pos, bs_target, layout_xy, keys, params, rng
            )
            total_time += bs_time
            total_keystrokes += 1  # the backspace itself
            # re-attempt the intended character
            committed2, pos, elapsed2 = _simulate_one_selection(
                pos, target_xy, layout_xy, keys, params, rng
            )
            total_time += elapsed2
            total_keystrokes += 1
            # (we optimistically assume the retry succeeds; if not, the next
            #  loop iteration's stats still reflect reality on average)
        produced += 1

    mean_tpc = total_time / produced
    cpm = 60.0 / mean_tpc
    wpm = cpm / 5.0
    kspc = total_keystrokes / produced
    misrate = misselects / produced

    return MCResult(
        layout_name=layout_name,
        corpus_name=profile.name,
        params_name=params.name,
        wpm_effective=wpm,
        misselect_rate=misrate,
        kspc=kspc,
        mean_time_per_char=mean_tpc,
        n_trials=produced,
    )


if __name__ == "__main__":
    from corpus import build_profiles
    from layouts import build_all_layouts

    profs = build_profiles()
    layouts = build_all_layouts(profs)
    params = CursorParams()

    print(f"Cursor model: {params.name} "
          f"(noise_sigma={params.noise_sigma}, key_radius={params.key_radius})\n")
    print(f"{'layout':<14}{'corpus':<8}{'WPM_eff':>9}{'mis%':>8}{'KSPC':>7}{'s/char':>8}")
    print("-" * 52)
    for lname, lay in layouts.items():
        res = simulate_layout(lay, profs["code"], params, lname, n_chars=1200)
        print(f"{lname:<14}{'code':<8}{res.wpm_effective:9.1f}"
              f"{100*res.misselect_rate:8.1f}{res.kspc:7.2f}{res.mean_time_per_char:8.3f}")

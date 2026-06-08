"""
simulate.py
===========
Head-to-head simulation of two circular-keyboard text-entry mechanics driven by
the SAME noisy 2D cursor model (a stand-in for a Neuralink-style intracortical
cursor):

  1. FLOW DIAL (proposed) -- no stop, no click. Candidates occupy angular arcs
     whose WIDTH grows with the language-model probability of that character.
     The user steers OUTWARD from a central rest zone; a character commits when
     the cursor crosses the outer ring while its heading lies within that arc.
     There is no dwell and no discrete click.

  2. POINT-DWELL (baseline) -- the conventional approach. 27 fixed keys on a
     ring; the user moves to a key and DWELLS (holds still) for a fixed time to
     select it. This is what most gaze / BCI keyboards do today.

Both mechanics are exercised over the same phrases with the same cursor noise so
the comparison is fair. The central scientific question:

    Does the continuous-flow mechanic degrade more GRACEFULLY under cursor
    noise than discrete point-and-dwell, and how much of that depends on the
    language model?

Cursor model
------------
The cursor is driven toward an intended on-screen goal by a proportional
controller plus Gaussian motor noise each timestep:

    pos += gain * (goal - pos) * dt  +  N(0, sigma)

`sigma` is the knob we sweep to represent worsening BCI signal quality.

Outputs per (mechanic, condition): effective WPM, character error rate (wrong
commits / intended), and mean time per character.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

from corpus_en import ALPHABET, PHRASES, next_char_distribution

# ---- geometry (normalized units; matches the spirit of flow_dial.html) ------
INNER = 0.28      # rest-zone radius
RING = 1.00       # commit radius (outer)
LABEL_R = 0.80    # where the "goal" point for a candidate sits (for aiming)


@dataclass
class CursorParams:
    gain: float = 10.0
    dt: float = 0.02
    sigma: float = 0.06      # motor noise std-dev per step (THE swept knob)
    max_steps: int = 400     # safety timeout per character
    name: str = "bci"


# ---------------------------------------------------------------------------
# Arc construction for the Flow Dial
# ---------------------------------------------------------------------------
def build_arcs(prev_char: str, use_prediction: bool):
    """
    Return list of (char, a0, a1, mid_angle, prob). Arc width is proportional to
    sqrt(prob) so high-probability characters get wide, easy-to-hit arcs while
    rare ones stay reachable (sqrt compresses the dynamic range, with a floor
    from the smoothed LM).
    """
    dist = next_char_distribution(prev_char, use_prediction)
    chars = list(ALPHABET)
    weights = np.array([math.sqrt(dist[c]) for c in chars])
    weights = weights / weights.sum()
    arcs = []
    a = -math.pi / 2
    for c, w in zip(chars, weights):
        span = w * 2 * math.pi
        arcs.append((c, a, a + span, a + span / 2, dist[c]))
        a += span
    return arcs


def _norm_angle_into(a, a0):
    while a < a0:
        a += 2 * math.pi
    while a >= a0 + 2 * math.pi:
        a -= 2 * math.pi
    return a


def arc_of_heading(angle, arcs):
    for i, (c, a0, a1, mid, p) in enumerate(arcs):
        if _norm_angle_into(angle, a0) < a1:
            return i
    return len(arcs) - 1


# ---------------------------------------------------------------------------
# FLOW DIAL: simulate committing one intended character
# ---------------------------------------------------------------------------
def flow_commit_one(intended, prev_char, params: CursorParams, rng,
                    use_prediction, start_pos):
    """
    Drive the cursor from its current position back to center, then push outward
    toward the intended character's arc, and commit whichever arc the heading is
    in when it crosses RING. Returns (committed_char, steps, end_pos).
    """
    arcs = build_arcs(prev_char, use_prediction)
    # find intended arc + its aiming goal point (just outside the ring)
    idx = next(i for i, a in enumerate(arcs) if a[0] == intended)
    mid = arcs[idx][3]
    goal = np.array([RING * 1.15 * math.cos(mid), RING * 1.15 * math.sin(mid)])

    # Phase A: return to the central rest zone (hysteresis requires re-entering
    # INNER before the next commit is armed). The cursor starts at the previous
    # commit point on the ring, so this return trip is a real, charged cost.
    pos = np.array([start_pos[0], start_pos[1]], dtype=float)
    return_steps = 0
    for _ in range(params.max_steps):
        pos = pos + params.gain * (np.zeros(2) - pos) * params.dt \
            + rng.normal(0, params.sigma, size=2)
        return_steps += 1
        if math.hypot(*pos) < INNER:
            break

    # Phase B: push outward toward the intended arc and commit on ring crossing.
    for step in range(params.max_steps):
        pos = pos + params.gain * (goal - pos) * params.dt \
            + rng.normal(0, params.sigma, size=2)
        r = math.hypot(*pos)
        if r >= RING:
            ang = math.atan2(pos[1], pos[0])
            committed = arcs[arc_of_heading(ang, arcs)][0]
            return committed, return_steps + step + 1, pos
        ang = math.atan2(pos[1], pos[0])
    # timed out: commit nearest arc by heading
    ang = math.atan2(pos[1], pos[0])
    return arcs[arc_of_heading(ang, arcs)][0], return_steps + params.max_steps, pos


# ---------------------------------------------------------------------------
# POINT-DWELL baseline: fixed keys + hold-still selection
# ---------------------------------------------------------------------------
def _fixed_key_positions():
    chars = list(ALPHABET)
    pos = {}
    n = len(chars)
    for i, c in enumerate(chars):
        a = -math.pi / 2 + (i / n) * 2 * math.pi
        pos[c] = np.array([LABEL_R * math.cos(a), LABEL_R * math.sin(a)])
    return pos


_KEYS = _fixed_key_positions()
_KEY_RADIUS = 0.16      # hit region around a key
_DWELL_STEPS = 6        # must stay inside the hit region this many steps


def dwell_commit_one(intended, params: CursorParams, rng, start_pos):
    """
    Move from the current position toward the intended fixed key and dwell.
    Commit the key the cursor dwells inside for _DWELL_STEPS. If it dwells in
    the WRONG (neighbouring) key, that is a mis-selection -- which is exactly
    what noise causes here. Returns (committed_char, steps, end_pos).
    """
    goal = _KEYS[intended]
    pos = np.array([start_pos[0], start_pos[1]], dtype=float)
    dwell_key, dwell_count = None, 0
    keys = list(_KEYS.items())
    for step in range(params.max_steps):
        pos = pos + params.gain * (goal - pos) * params.dt \
            + rng.normal(0, params.sigma, size=2)
        # nearest key
        nearest, nd = None, 1e9
        for c, kp in keys:
            d = (kp[0] - pos[0]) ** 2 + (kp[1] - pos[1]) ** 2
            if d < nd:
                nd, nearest = d, c
        if math.sqrt(nd) <= _KEY_RADIUS:
            if nearest == dwell_key:
                dwell_count += 1
            else:
                dwell_key, dwell_count = nearest, 1
            if dwell_count >= _DWELL_STEPS:
                return dwell_key, step + 1, pos
        else:
            dwell_key, dwell_count = None, 0
    # timed out without a stable dwell: commit whatever key the cursor is
    # nearest to at the final (noisy) position -- NOT the intended key. This
    # avoids artificially flattering the baseline at high noise.
    nearest, nd = intended, 1e9
    for c, kp in keys:
        d = (kp[0] - pos[0]) ** 2 + (kp[1] - pos[1]) ** 2
        if d < nd:
            nd, nearest = d, c
    return nearest, params.max_steps, pos


# ---------------------------------------------------------------------------
# Run a full phrase set
# ---------------------------------------------------------------------------
@dataclass
class Result:
    mechanic: str
    sigma: float
    use_prediction: bool
    wpm: float
    cer: float          # character error rate
    sec_per_char: float
    n_chars: int


def run(mechanic: str, params: CursorParams, use_prediction=True,
        seed=0, phrases=None) -> Result:
    rng = np.random.default_rng(seed)
    phrases = phrases or PHRASES
    total_steps = 0
    errors = 0
    n = 0
    for phrase in phrases:
        prev = " "
        pos = np.array([0.0, 0.0])  # start at center
        for ch in phrase:
            if mechanic == "flow":
                got, steps, pos = flow_commit_one(
                    ch, prev, params, rng, use_prediction, pos)
            elif mechanic == "dwell":
                got, steps, pos = dwell_commit_one(ch, params, rng, pos)
            else:
                raise ValueError(mechanic)
            total_steps += steps
            n += 1
            if got != ch:
                errors += 1
                # correction model: a wrong commit must be erased and retried.
                # We charge one extra return+select of comparable effort.
                total_steps += steps
            prev = ch  # next-char prediction conditions on the INTENDED char
    sec = total_steps * params.dt
    sec_per_char = sec / n
    wpm = (n / 5) / (sec / 60)
    return Result(mechanic, params.sigma, use_prediction,
                  round(wpm, 2), round(errors / n, 4),
                  round(sec_per_char, 3), n)


if __name__ == "__main__":
    print("Quick smoke test (sigma=0.06):")
    for mech in ("flow", "dwell"):
        r = run(mech, CursorParams(sigma=0.06), use_prediction=True)
        print(f"  {mech:6s}  WPM={r.wpm:6.2f}  CER={100*r.cer:5.1f}%  "
              f"s/char={r.sec_per_char:.3f}")

"""
layouts.py
==========
Generates 2D key coordinates for each keyboard layout under evaluation.

All coordinates are expressed in a normalized device space: a unit circle of
radius 1.0 centered at the origin (0, 0). The cursor's neutral rest position is
the origin. This lets the Fitts' / Monte-Carlo models compute travel distance
and angular separation consistently across very different layout geometries.

Layouts
-------
1. QWERTY_GRID      -- baseline: a flat full-width QWERTY block (models the
                       Neuralink-style swipe / point-to-type keyboard). Wide
                       aspect ratio => long diagonal travel between distant keys.

2. RADIAL_PROSE     -- concentric radial layout whose ring assignment is driven
                       by NATURAL-LANGUAGE letter frequency (Irisbond / Scott
                       Morgan Foundation style): frequent letters inner, rare
                       letters outer.

3. RADIAL_CODE      -- THE PROPOSED DESIGN. Same concentric-radial structure,
                       but ring/sector assignment is driven by CODE token
                       frequency, so symbols ( ) . ; = { } and frequent code
                       letters land in the fast inner ring.

Each layout is a dict: { char : (x, y) }.
"""

from __future__ import annotations

import math
from corpus import (
    FrequencyProfile,
    LETTERS,
    DIGITS,
    CODE_SYMBOLS,
    ALL_KEYS_CODE,
    ALL_KEYS_PROSE,
)


Layout = dict[str, tuple[float, float]]


# ---------------------------------------------------------------------------
# 1. QWERTY grid baseline (Neuralink-style full-width keyboard)
# ---------------------------------------------------------------------------
_QWERTY_ROWS = [
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
]


def qwerty_grid(include_space: bool = True) -> Layout:
    """
    Flat QWERTY. Keys laid on a grid, then scaled so the whole block fits inside
    the unit circle. A wide keyboard => large Euclidean distances between far
    keys, which is exactly the cost a swipe/point cursor pays.
    """
    layout: Layout = {}
    key_w, key_h = 1.0, 1.0
    # row horizontal offsets approximate the real QWERTY stagger
    row_offsets = [0.0, 0.5, 1.5]
    positions = []
    for r, row in enumerate(_QWERTY_ROWS):
        for c, ch in enumerate(row):
            x = (c + row_offsets[r]) * key_w
            y = -r * key_h  # rows go downward
            positions.append((ch, x, y))
    # space bar sits below, centered
    if include_space:
        positions.append((" ", 4.5, -3.0))

    # center and scale to unit circle
    xs = [p[1] for p in positions]
    ys = [p[2] for p in positions]
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys)) / 2
    for ch, x, y in positions:
        layout[ch] = ((x - cx) / span, (y - cy) / span)
    return layout


# ---------------------------------------------------------------------------
# Generic concentric-radial builder
# ---------------------------------------------------------------------------
def _concentric_radial(
    ordered_keys: list[str],
    ring_sizes: list[int],
    ring_radii: list[float],
    center_keys: list[str] | None = None,
) -> Layout:
    """
    Place keys on concentric rings. `ordered_keys` is sorted most-frequent first.
    The most frequent keys fill the innermost ring (smallest radius => shortest
    travel from the central rest point), then spill outward.

    center_keys : optional keys placed AT the very center (radius ~0), reserved
                  for the single most frequent token(s) -- mirrors the 't'/'e'
                  center seen in the Irisbond design.
    """
    layout: Layout = {}
    idx = 0
    keys = list(ordered_keys)

    if center_keys:
        # place center keys in a tiny inner cluster around origin
        n = len(center_keys)
        for i, ch in enumerate(center_keys):
            if n == 1:
                layout[ch] = (0.0, 0.0)
            else:
                ang = 2 * math.pi * i / n
                layout[ch] = (0.12 * math.cos(ang), 0.12 * math.sin(ang))
        keys = [k for k in keys if k not in set(center_keys)]

    for ring_i, (count, radius) in enumerate(zip(ring_sizes, ring_radii)):
        ring_keys = keys[idx: idx + count]
        idx += count
        m = len(ring_keys)
        if m == 0:
            continue
        # rotate alternate rings slightly so keys don't radially align (less
        # ambiguity for a noisy cursor)
        phase = (math.pi / m) * (ring_i % 2)
        for j, ch in enumerate(ring_keys):
            ang = 2 * math.pi * j / m + phase
            layout[ch] = (radius * math.cos(ang), radius * math.sin(ang))
    # any leftover keys (corpus had more chars than ring capacity) -> outermost
    if idx < len(keys):
        leftover = keys[idx:]
        m = len(leftover)
        for j, ch in enumerate(leftover):
            ang = 2 * math.pi * j / m
            r = ring_radii[-1] * 1.15
            layout[ch] = (r * math.cos(ang), r * math.sin(ang))
    return layout


def _frequency_ordered_keys(profile: FrequencyProfile, key_universe: str) -> list[str]:
    """All keys in `key_universe` sorted by descending frequency in `profile`.
    Keys never seen still appear (freq 0) so every layout covers the same set."""
    seen = profile.unigram
    return sorted(key_universe, key=lambda c: (-seen.get(c, 0), c))


# ---------------------------------------------------------------------------
# 2. Prose-optimized radial (Irisbond-style)
# ---------------------------------------------------------------------------
def radial_prose(profile: FrequencyProfile) -> Layout:
    """
    Concentric radial layout optimized for natural-language frequency.
    Models the existing gaze keyboard: inner ring = frequent letters, outer =
    rare letters, with the two most frequent at center.
    Universe is letters + space + light punctuation (no code symbols), matching
    a prose keyboard's primary surface.
    """
    universe = LETTERS + " " + ".,'"
    ordered = _frequency_ordered_keys(profile, universe)
    center = ordered[:2]                 # e.g. e / t at center
    rest = ordered[2:]
    # ring capacities chosen to resemble the Irisbond 2-ring design
    ring_sizes = [10, 12, 12]
    ring_radii = [0.45, 0.78, 1.05]
    return _concentric_radial(rest, ring_sizes, ring_radii, center_keys=center)


# ---------------------------------------------------------------------------
# 3. Code-optimized radial  (THE PROPOSAL)
# ---------------------------------------------------------------------------
def radial_code(profile: FrequencyProfile) -> Layout:
    """
    Concentric radial layout optimized for CODE token frequency. The full code
    key universe (letters + digits + symbols + space) is ranked by code
    frequency, so high-frequency symbols ( ) . ; = and frequent letters occupy
    the fast inner ring; rare symbols (`@ % \\ etc.) go to the periphery.
    """
    universe = ALL_KEYS_CODE
    ordered = _frequency_ordered_keys(profile, universe)
    center = ordered[:2]
    rest = ordered[2:]
    # more keys than prose => more rings
    ring_sizes = [10, 14, 18, 22]
    ring_radii = [0.42, 0.66, 0.88, 1.08]
    return _concentric_radial(rest, ring_sizes, ring_radii, center_keys=center)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def build_all_layouts(profiles: dict[str, FrequencyProfile]) -> dict[str, Layout]:
    return {
        "qwerty_grid": qwerty_grid(),
        "radial_prose": radial_prose(profiles["prose"]),
        "radial_code": radial_code(profiles["code"]),
    }


if __name__ == "__main__":
    from corpus import build_profiles

    profs = build_profiles()
    layouts = build_all_layouts(profs)
    for name, lay in layouts.items():
        print(f"\n=== {name} : {len(lay)} keys ===")
        # show the 8 keys closest to center (the 'fast' keys)
        by_dist = sorted(lay.items(), key=lambda kv: math.hypot(*kv[1]))
        near = [("<sp>" if k == " " else k) for k, _ in by_dist[:8]]
        print("  fastest (closest to center):", " ".join(near))

"""Quick inspection: which keys land in the fast inner rings of radial_code,
and what is the README summary table."""
import math
from corpus import build_profiles
from layouts import radial_code

profs = build_profiles()
lay = radial_code(profs["code"])
by_dist = sorted(lay.items(), key=lambda kv: math.hypot(*kv[1]))

print("Radial-CODE layout, keys ordered by distance from center (fastest first):")
print("-" * 60)
for i, (ch, (x, y)) in enumerate(by_dist[:24]):
    label = "<space>" if ch == " " else ch
    r = math.hypot(x, y)
    is_sym = ch in "_.,:;=()[]{}<>\"'/*+-!&|#$%@\\`?"
    tag = "  <-- code symbol" if is_sym else ""
    print(f"  {i+1:2d}. {label:<8} r={r:.3f}{tag}")

"""
corpus_en.py
============
Self-contained English corpus + bigram language model for the Flow Dial
evaluation. Embedded so the experiment is fully reproducible offline.

Provides:
  - PHRASES        : a list of test phrases to "type" in simulation
  - unigram_prob   : P(char)
  - bigram_prob    : P(next | prev), with add-k smoothing and a floor so every
                     character is always reachable (no zero-width arcs)

Character universe: 26 lowercase letters + space. (Backspace is handled by the
simulator as a correction action, not a corpus symbol.)
"""

from __future__ import annotations
from collections import Counter, defaultdict

ALPHABET = "abcdefghijklmnopqrstuvwxyz "
SPACE = " "

# A compact, public-domain-style English sample. Large enough to give stable
# bigram statistics for the alphabet; swap in a bigger corpus for camera-ready.
_TEXT = """
the quick brown fox jumps over the lazy dog and then the dog runs back to the
warm house where the fire is burning bright through the long cold night it was
the best of times it was the worst of times we were all going direct to heaven
we were all going direct the other way she sells sea shells by the sea shore
and the shells she sells are surely sea shells a journey of a thousand miles
begins with a single step and every step after that brings us a little closer
to where we hope to arrive before the sun goes down to be or not to be that is
the question whether it is nobler in the mind to suffer the slings and arrows
of outrageous fortune or to take arms against a sea of troubles people often
say that motivation does not last well neither does bathing that is why we
recommend it daily the only way to do great work is to love what you do if you
have not found it yet keep looking do not settle as with all matters of the
heart you will know it when you find it and like any great relationship it just
gets better and better as the years roll on
"""

# Test phrases for simulation (standard short, common-word phrases).
PHRASES = [
    "the quick brown fox",
    "she sells sea shells",
    "to be or not to be",
    "a journey of a thousand miles",
    "the best of times",
    "love what you do",
    "every step brings us closer",
    "the only way to do great work",
]


def _normalize(t: str) -> str:
    out = []
    prev_space = False
    for ch in t.lower():
        if ch.isalpha():
            out.append(ch)
            prev_space = False
        elif ch.isspace():
            if not prev_space:
                out.append(" ")
                prev_space = True
    return "".join(out).strip()


_NORM = _normalize(_TEXT)

# --- counts ---
_uni = Counter(_NORM)
_bi = defaultdict(Counter)
for a, b in zip(_NORM, _NORM[1:]):
    _bi[a][b] += 1

_TOTAL_UNI = sum(_uni.values())
_K = 0.5  # add-k smoothing
_V = len(ALPHABET)


def unigram_prob(ch: str) -> float:
    return (_uni.get(ch, 0) + _K) / (_TOTAL_UNI + _K * _V)


def bigram_prob(prev: str, nxt: str) -> float:
    row = _bi.get(prev)
    denom = (sum(row.values()) if row else 0) + _K * _V
    num = (row.get(nxt, 0) if row else 0) + _K
    return num / denom


def next_char_distribution(prev: str, use_prediction: bool = True) -> dict[str, float]:
    """Return a normalized P(next | prev) over the full alphabet."""
    if use_prediction and prev in _bi:
        p = {c: bigram_prob(prev, c) for c in ALPHABET}
    else:
        p = {c: unigram_prob(c) for c in ALPHABET}
    tot = sum(p.values())
    return {c: v / tot for c, v in p.items()}


if __name__ == "__main__":
    print(f"corpus chars: {_TOTAL_UNI}, alphabet size: {_V}")
    print("top unigrams:", [(c if c != ' ' else '<sp>', round(unigram_prob(c), 3))
                            for c, _ in _uni.most_common(8)])
    print("P(next|'t'):", sorted(
        ((round(v, 3), c if c != ' ' else '<sp>')
         for c, v in next_char_distribution('t').items()), reverse=True)[:6])

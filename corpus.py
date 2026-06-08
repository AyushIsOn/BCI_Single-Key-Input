"""
corpus.py
=========
Builds two character-frequency profiles used to drive layout optimization and
the typing simulations:

  1. CODE  -- representative source code (Python + JavaScript/TypeScript-ish).
  2. PROSE -- representative natural-language English text.

We compute:
  - unigram frequencies  P(c)           : how often each character is typed
  - bigram  frequencies  P(c_i, c_j)    : how often char j follows char i
    (bigrams drive the Fitts' movement-time model, since travel time depends
     on consecutive key positions)

The corpora are intentionally embedded as strings so the harness is fully
self-contained and reproducible offline (important for a paper artifact).
For a camera-ready you would swap in a larger external corpus; the API here
(`build_profiles`) does not change.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Embedded sample corpora
# ---------------------------------------------------------------------------
# A compact but representative slice of real-world code. Mixing Python and a
# C-family language gives a realistic symbol distribution ({}, ;, =>, (), etc.)
CODE_SAMPLE = r'''
import os
import sys
from typing import List, Dict, Optional

def load_config(path: str) -> Dict[str, str]:
    config = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip()
    return config

class TokenBucket:
    def __init__(self, rate: float, capacity: int) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.timestamp = time.monotonic()

    def allow(self, cost: int = 1) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.timestamp) * self.rate)
        self.timestamp = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

results = [fib(i) for i in range(20) if i % 2 == 0]
print(f"results = {results}")

const cache = new Map();
function memoize(fn) {
  return (...args) => {
    const key = JSON.stringify(args);
    if (cache.has(key)) return cache.get(key);
    const value = fn(...args);
    cache.set(key, value);
    return value;
  };
}

export async function fetchUser(id) {
  const res = await fetch(`/api/users/${id}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

for (let i = 0; i < items.length; i++) {
  total += items[i].price * items[i].qty;
}
'''

# A compact natural-language sample (public-domain style prose).
PROSE_SAMPLE = r'''
The quick brown fox jumps over the lazy dog. It was the best of times, it was
the worst of times, it was the age of wisdom, it was the age of foolishness.
She sells seashells by the seashore, and the shells she sells are surely
seashells. We hold these truths to be self evident, that all people are created
equal. To be or not to be, that is the question we must each answer for
ourselves in the fullness of time. Whether the weather is warm or whether the
weather is cold, we will weather the weather together as friends do. A journey
of a thousand miles begins with a single step, and every step thereafter brings
us a little closer to where we hope to arrive before the sun goes down.
'''


# ---------------------------------------------------------------------------
# Frequency profile
# ---------------------------------------------------------------------------
@dataclass
class FrequencyProfile:
    """Character usage statistics for a body of text."""
    name: str
    unigram: Counter = field(default_factory=Counter)   # char -> count
    bigram: Counter = field(default_factory=Counter)     # (char, char) -> count
    total_chars: int = 0

    def unigram_prob(self, ch: str) -> float:
        if self.total_chars == 0:
            return 0.0
        return self.unigram.get(ch, 0) / self.total_chars

    def top(self, n: int = 15):
        return self.unigram.most_common(n)


# The set of "typeable" characters we model. We fold uppercase to lowercase for
# the base layout (shift is modeled as a separate cost where relevant) and keep
# the code-relevant symbols explicitly.
LETTERS = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
CODE_SYMBOLS = "_.,:;=()[]{}<>\"'/*+-!&|#$%@\\`?"
SPACE = " "

ALL_KEYS_CODE = LETTERS + DIGITS + CODE_SYMBOLS + SPACE
ALL_KEYS_PROSE = LETTERS + ".,!?;:'\"-" + SPACE


def _normalize(text: str) -> str:
    # collapse newlines/tabs into spaces; lowercase letters
    text = text.replace("\t", "    ")
    text = re.sub(r"\n+", " ", text)
    return text.lower()


def _count(text: str, allowed: str) -> FrequencyProfile:
    allowed_set = set(allowed)
    prof = FrequencyProfile(name="")
    prev = None
    for ch in text:
        if ch not in allowed_set:
            # map any unknown char to space so we still capture word breaks
            ch = " " if ch.isspace() else None
            if ch is None:
                prev = None
                continue
        prof.unigram[ch] += 1
        prof.total_chars += 1
        if prev is not None:
            prof.bigram[(prev, ch)] += 1
        prev = ch
    return prof


def build_profiles() -> dict[str, FrequencyProfile]:
    """Return {'code': FrequencyProfile, 'prose': FrequencyProfile}."""
    code = _count(_normalize(CODE_SAMPLE), ALL_KEYS_CODE)
    code.name = "code"
    prose = _count(_normalize(PROSE_SAMPLE), ALL_KEYS_PROSE)
    prose.name = "prose"
    return {"code": code, "prose": prose}


if __name__ == "__main__":
    profs = build_profiles()
    for name, prof in profs.items():
        print(f"\n=== {name.upper()} corpus ===")
        print(f"total typed chars: {prof.total_chars}")
        print("top 15 characters (excluding space shown as <sp>):")
        for ch, cnt in prof.top(15):
            label = "<sp>" if ch == " " else ch
            print(f"  {label:>4}  {cnt:5d}  {100*prof.unigram_prob(ch):5.2f}%")

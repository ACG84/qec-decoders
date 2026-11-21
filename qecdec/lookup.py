"""Lookup-table decoder built by enumerating likely faults.

Enumerate every combination of up to `max_weight` error mechanisms from the detector error
model, and for each resulting syndrome store the logical effect of the most probable
combination. Decoding is one dictionary read. Syndromes not in the table go to `fallback`
(another decoder) or return "no flip". Table size grows as (number of mechanisms)^max_weight,
so this is for small codes; its appeal is constant-time, trivially parallel decoding.
"""
from itertools import combinations
import numpy as np


class LookupDecoder:
    name = "Lookup table"

    def __init__(self, graph, max_weight=2, fallback=None):
        self.g, self.fallback = graph, fallback
        self.name = f"Lookup table (weight <= {max_weight})"
        E = len(graph.eu)
        logp = np.log(graph.ep / (1 - graph.ep))
        table = {0: (0.0, 0)}
        masks = []
        for e in range(E):
            m = 0
            for x in (int(graph.eu[e]), int(graph.ev[e])):
                if not graph.is_boundary(x):
                    m ^= 1 << x
            masks.append(m)
        for w in range(1, max_weight + 1):
            for combo in combinations(range(E), w):
                key = 0; lp = 0.0; obs = 0
                for e in combo:
                    key ^= masks[e]; lp += logp[e]; obs ^= int(graph.eobs[e])
                if key not in table or lp > table[key][0]:
                    table[key] = (lp, obs)
        self.table = {k: v[1] for k, v in table.items()}
        self.misses = 0

    def decode(self, syndrome):
        key = 0
        for i in np.flatnonzero(syndrome[: self.g.n]):
            key |= 1 << int(i)
        hit = self.table.get(key)
        if hit is not None:
            return hit
        self.misses += 1
        return self.fallback.decode(syndrome) if self.fallback else 0

    def decode_batch(self, syndromes):
        return np.array([self.decode(s) for s in syndromes], dtype=np.uint8)

"""Clifford-clause decoder: a Tsetlin machine whose literals are parities it found itself.

On syndrome bits a CNOT circuit is a GF(2)-linear map, so a clause that ANDs outputs of a
Clifford circuit is an AND of parity checks: the indicator of an affine subspace of syndromes.
A lookup table is the special case with one clause per syndrome; a Clifford clause covers a
whole subspace of syndromes with one rule, so the rule list is a compressed table.

Two sources of literals:
- "cuts" (default): cut parities derived from the decoding graph, see `cut_literals`.
- "learned": `grow_parities` proposes parities from the training syndromes alone. Starting from
  single detectors, each round it XORs pairs of literals that fire together far more often than
  chance (one CNOT deeper per round). No graph or layout is used.
"""
import numpy as np
from .tsetlin_core import TsetlinMachine
from .clifford_literals import grow_parities, apply_supports, support_names


def cut_literals(graph):
    """Parity literals read off the decoding graph. Returns (matrix over detectors, names).

    Level of a detector = hop distance to the boundary whose faults flip the logical observable.
    cut_k = XOR of all detectors with level <= k. Every fault on that boundary flips every cut_k
    (the logical flip we want), while a bulk fault flips cut_k only if it crosses between levels
    k and k+1. So each cut is a noisy vote on the logical flip, and different cuts are wrong on
    different faults. A cut XORed with one nearby stabilizer's time-parity is the same cut bulged
    around that stabilizer: another vote, with its errors in a different place.
    Computing cut_k is a prefix-XOR over the levels, i.e. a CNOT ladder of logarithmic depth.
    """
    from collections import deque
    g = graph
    lvl = np.full(g.n, -1); q = deque()
    for e in np.flatnonzero(g.eobs):
        u = int(g.eu[e])
        if lvl[u] < 0:
            lvl[u] = 0; q.append(u)
    while q:
        u = q.popleft()
        for v, _ in g.adj[u]:
            if v < g.n and lvl[v] < 0:
                lvl[v] = lvl[u] + 1; q.append(v)
    K = int(lvl.max()) + 1
    cuts = [((lvl >= 0) & (lvl <= k)).astype(np.int32) for k in range(K)]
    co = g.circuit.get_detector_coordinates(); stab = {}
    for i in range(g.n):
        if lvl[i] >= 0:
            stab.setdefault((int(co[i][0]), int(co[i][1])), []).append(i)
    rows, names = list(cuts), [f"cut{k}" for k in range(K)]
    for (x, y), idx in sorted(stab.items()):
        v = np.zeros(g.n, np.int32); v[idx] = 1; l = int(lvl[idx[0]])
        rows.append(v); names.append(f"S[x{x},y{y}]")
        for k in range(K):
            if l in (k, k + 1):
                rows.append((cuts[k] + v) % 2); names.append(f"cut{k}^S[x{x},y{y}]")
    return np.array(rows, dtype=np.int32), names


class CliffordClauseDecoder:
    """literals="cuts": cut and bulged-cut parities from the decoding graph (default).
    literals="learned": parities proposed from the training syndromes alone, no graph used."""

    def __init__(self, graph, literals="cuts", rounds=5, n_clauses=None, T=None, s=None, seed=1):
        self.g, self.mode, self.rounds = graph, literals, rounds
        cfg = dict(n_clauses=100, T=12, s=5.0) if literals == "cuts" else dict(n_clauses=100, T=10, s=8.0)
        if n_clauses: cfg["n_clauses"] = n_clauses
        if T: cfg["T"] = T
        if s: cfg["s"] = s
        self.tm = TsetlinMachine(seed=seed, **cfg)
        self.name = "Clifford-clause TM (cut literals)" if literals == "cuts" else "Clifford-clause TM (learned literals)"
        self.supports = None
        if literals == "cuts":
            self.M, self.names = cut_literals(graph)

    def features(self, syndromes):
        X = np.atleast_2d(syndromes)
        if self.mode == "cuts":
            return (X.astype(np.int32) @ self.M.T) % 2 == 1
        return apply_supports(X, self.supports)

    def fit(self, syndromes, observables, epochs=8, val_fraction=0.2):
        S = np.asarray(syndromes, dtype=bool)
        if self.mode != "cuts":
            self.supports = [s for s in grow_parities(S, rounds=self.rounds) if len(s) > 1]
            self.names = support_names(self.supports, [f"D{i}" for i in range(self.g.n)])
        X = self.features(S); nv = int(len(X) * val_fraction)
        self.tm.fit(X[:-nv], observables[:-nv], epochs=epochs, val=(X[-nv:], observables[-nv:]))
        return self

    def decode(self, syndrome):
        return int(self.tm.predict(self.features(syndrome))[0])

    def decode_batch(self, syndromes):
        return self.tm.predict(self.features(syndromes)).astype(np.uint8)

    def rules(self):
        return self.tm.rules(self.names)

    def n_literals(self):
        return len(self.names)

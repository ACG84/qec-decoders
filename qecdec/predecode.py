"""Pair cancellation: a local first layer that removes the syndrome of isolated single faults.

Most faults flip two neighbouring detectors. If two neighbouring detectors both fired and neither
has any other fired neighbour, the pair is explained by the one fault on the edge between them:
clear both and record that edge's effect on the logical observable. Every edge is tested at
once, so this is one layer of constant depth: an AND of two detector bits and the negations of
their other neighbours, then two XORs. In the vocabulary of `clifford_clause.py` it is a clause
layer whose action is a correction instead of a vote.

What it buys: a bulk fault between two cut levels corrupts one cut parity. After cancellation
that fault is gone, so the cut parities computed on the residual syndrome are wrong far less
often, and the rule machine built on them answers many more shots.
What it does not do: it never touches a lone detector or a cluster of three or more, so anything
ambiguous is left for the later layers.
"""
import numpy as np


class PairCancel:
    def __init__(self, graph):
        g = self.g = graph
        dd = np.flatnonzero((g.eu < g.n) & (g.ev < g.n))
        self.u, self.v = g.eu[dd].astype(np.int64), g.ev[dd].astype(np.int64)
        self.eobs = g.eobs[dd].astype(np.int32)
        self.A = np.zeros((g.n, g.n), np.float32)
        self.A[self.u, self.v] = 1; self.A[self.v, self.u] = 1

    def apply(self, syndromes, chunk=4096):
        """Returns (residual syndromes, observable flips already accounted for)."""
        S = np.atleast_2d(syndromes).astype(np.uint8)
        R = S.copy(); obs = np.zeros(len(S), np.uint8)
        for a in range(0, len(S), chunk):
            s = S[a:a + chunk]
            c = s.astype(np.float32) @ self.A                     # fired neighbours of each detector
            lone = (s == 1) & (c == 1)                            # fired, with exactly one fired neighbour
            iso = lone[:, self.u] & lone[:, self.v]               # both ends lone: they are each other's neighbour
            rows, cols = np.nonzero(iso)
            R[a + rows, self.u[cols]] = 0; R[a + rows, self.v[cols]] = 0
            obs[a:a + chunk] = (iso.astype(np.int32) @ self.eobs) % 2
        return R, obs

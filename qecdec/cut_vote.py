"""Cut-vote first stage with a fallback decoder.

Each cut parity (see clifford_clause.cut_literals) is a vote on the logical flip. If all cuts
agree, take their answer: that is a handful of XOR trees and one comparison, with no matching.
Otherwise hand the shot to `fallback`. Unanimous shots include the empty syndrome. The share of
shots settled this way falls as distance and error rate grow, so this is a fast path for the
easy shots and not a decoder that scales alone.
"""
import numpy as np
from .clifford_clause import cut_literals
from .unionfind import UnionFindDecoder


class CutVoteDecoder:
    def __init__(self, graph, fallback=None, predecode=None):
        self.g = graph; self.pre = predecode
        self.fallback = fallback if fallback is not None else UnionFindDecoder(graph)
        M, names = cut_literals(graph)
        self.C = M[[i for i, n in enumerate(names) if n.startswith("cut") and "^" not in n]]
        self.name = ("Pair cancel -> " if predecode is not None else "") + f"Cut vote -> {self.fallback.name}"
        self.fast = 0; self.total = 0

    def decode_batch(self, syndromes):
        S = np.atleast_2d(syndromes)
        R, done = (S, 0) if self.pre is None else self.pre.apply(S)
        V = (R.astype(np.int32) @ self.C.T) % 2
        unanimous = V.min(axis=1) == V.max(axis=1)
        pred = V[:, 0].astype(np.uint8) ^ done
        rest = np.flatnonzero(~unanimous)
        if len(rest):
            pred[rest] = self.fallback.decode_batch(S[rest])
        self.fast += int(unanimous.sum()); self.total += len(S)
        return pred

    def decode(self, syndrome):
        return int(self.decode_batch(np.asarray(syndrome)[None, :])[0])

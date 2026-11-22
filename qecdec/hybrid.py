"""Clifford-clause machine in front of exact matching, gated by vote margin.

The Tsetlin machine's output is a vote count: clauses for "flip" minus clauses against. A large
margin means many independent rules agree. Shots whose margin reaches `margin` are answered by
the rule machine (bitwise, batched, no graph search); the rest go to `fallback` (exact matching
by default). `margin` is a dial between the two: 0 is the rule machine alone, a margin above the
largest possible vote is the fallback alone.
"""
import numpy as np
from .mwpm import MWPMDecoder


class HybridDecoder:
    """predecode: an optional `PairCancel` layer. The rule machine then reads the residual
    syndrome (it must have been trained on residuals); the fallback always gets the original."""

    def __init__(self, rule_decoder, fallback=None, margin=None, predecode=None):
        margin = rule_decoder.tm.T + 1 if margin is None else margin   # just past the training target T
        self.rules_dec, self.margin, self.pre = rule_decoder, margin, predecode
        self.fallback = fallback if fallback is not None else MWPMDecoder(rule_decoder.g)
        self.name = ("Pair cancel -> " if predecode is not None else "") + f"Clifford-clause -> MWPM (margin {margin})"
        self.fast = 0; self.total = 0

    def decode_batch(self, syndromes):
        S = np.atleast_2d(syndromes)
        R, done = (S, 0) if self.pre is None else self.pre.apply(S)
        votes = self.rules_dec.tm.votes(self.rules_dec.features(R))
        confident = np.abs(votes) >= self.margin
        pred = (votes > 0).astype(np.uint8) ^ done
        rest = np.flatnonzero(~confident)
        if len(rest):
            pred[rest] = self.fallback.decode_batch(S[rest])
        self.fast += int(confident.sum()); self.total += len(S)
        return pred

    def decode(self, syndrome):
        return int(self.decode_batch(np.asarray(syndrome)[None, :])[0])

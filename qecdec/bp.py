"""Belief propagation decoder (normalized min-sum) with a fallback.

Variables are the error mechanisms (edges of the decoding graph), checks are the detectors.
Messages are log-likelihood ratios. Each iteration every check tells each neighbouring edge how
likely it is to be in error given the others, and every edge sends back its prior plus what its
other check said. All messages in an iteration are updated at once from the previous ones, for
every shot in the batch, which is the same schedule a parallel hardware implementation uses.

After the last iteration the edges with negative total LLR form the guess. If the guess
reproduces the syndrome it is used. Surface codes have many equal-weight corrections, so BP
often does not settle; those shots go to `fallback` (Union-Find by default).
"""
import numpy as np
from .unionfind import UnionFindDecoder


class BPDecoder:
    name = "Belief propagation + UF fallback"

    def __init__(self, graph, iterations=20, scale=0.75, fallback=None):
        self.g, self.iterations, self.scale = graph, iterations, scale
        self.fallback = fallback if fallback is not None else UnionFindDecoder(graph)
        E = len(graph.eu)
        self.prior = graph.ew.copy()
        # slot = (edge, side). side 0 is endpoint u, side 1 is endpoint v (absent if boundary).
        slots = [[] for _ in range(graph.n)]
        for e in range(E):
            slots[int(graph.eu[e])].append(2 * e)
            if not graph.is_boundary(int(graph.ev[e])):
                slots[int(graph.ev[e])].append(2 * e + 1)
        self.maxdeg = max(len(s) for s in slots)
        self.slot = np.full((graph.n, self.maxdeg), 2 * E, dtype=np.int64)  # 2E = padding slot
        for c, s in enumerate(slots):
            self.slot[c, : len(s)] = s
        self.pad = self.slot == 2 * E
        self.converged = 0; self.total = 0

    def decode_batch(self, syndromes):
        g = self.g; S = np.atleast_2d(syndromes)[:, : g.n].astype(np.int8); B = len(S); E = len(g.eu)
        m = np.zeros((B, 2 * E + 1))                       # check -> edge messages, per slot
        prior2 = np.repeat(self.prior, 2)
        sgn_s = (1 - 2 * S)[:, :, None]                    # (-1)^syndrome
        for _ in range(self.iterations):
            other = m[:, : 2 * E].reshape(B, E, 2)[:, :, ::-1].reshape(B, 2 * E)  # message from the other check
            v = np.concatenate([prior2[None, :] + other, np.full((B, 1), 1e9)], axis=1)  # edge -> check; pad is "certainly no error"
            vc = v[:, self.slot]                                   # (B, checks, maxdeg)
            a = np.abs(vc); sign = np.where(vc < 0, -1.0, 1.0)
            total_sign = sign.prod(axis=2, keepdims=True)
            order = np.argsort(a, axis=2)
            min1 = np.take_along_axis(a, order[:, :, :1], axis=2); min2 = np.take_along_axis(a, order[:, :, 1:2], axis=2)
            is_min = np.arange(self.maxdeg)[None, None, :] == order[:, :, :1]
            out = self.scale * sgn_s * total_sign * sign * np.where(is_min, min2, min1)
            m = np.zeros((B, 2 * E + 1))
            np.put_along_axis(m, np.broadcast_to(self.slot.reshape(1, -1), (B, self.slot.size)), out.reshape(B, -1), axis=1)
            m[:, 2 * E] = 0.0
        post = self.prior[None, :] + m[:, : 2 * E].reshape(B, E, 2).sum(axis=2)
        x = (post < 0).astype(np.int32)
        synd = np.zeros((B, g.num_nodes), np.int32)
        np.add.at(synd, (slice(None), g.eu), x); np.add.at(synd, (slice(None), g.ev), x)
        ok = ((synd[:, : g.n] % 2) == S).all(axis=1)
        pred = (x @ g.eobs.astype(np.int32)) % 2
        self.converged += int(ok.sum()); self.total += B
        for i in np.flatnonzero(~ok):
            pred[i] = self.fallback.decode(S[i])
        return pred.astype(np.uint8)

    def decode(self, syndrome):
        return int(self.decode_batch(np.asarray(syndrome)[None, :])[0])

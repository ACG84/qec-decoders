"""Greedy matching decoder: repeatedly commit the cheapest remaining pairing.

The cost of pairing defects i and j is min(dist(i, j), boundary(i) + boundary(j)): either they
are joined directly or each goes to the boundary. All pairs are sorted by that cost and accepted
while both defects are free; a single leftover defect goes to the boundary.
O(k^2 log k) per shot, no optimality guarantee, very little state.
"""
import numpy as np


def greedy_match(defects, dist, par, bdist, bpar):
    k = len(defects)
    cands = []
    for i in range(k):
        a = defects[i]
        for j in range(i + 1, k):
            b = defects[j]
            direct, via = dist[a, b], bdist[a] + bdist[b]
            if direct <= via:
                cands.append((direct, i, j, int(par[a, b])))
            else:
                cands.append((via, i, j, int(bpar[a]) ^ int(bpar[b])))
    cands.sort(key=lambda c: c[0])
    free = [True] * k; obs = 0; pairs = []
    for _, i, j, o in cands:
        if free[i] and free[j]:
            free[i] = free[j] = False; obs ^= o; pairs.append((i, j))
    for i in range(k):
        if free[i]:
            obs ^= int(bpar[defects[i]]); pairs.append((i, -1))
    return obs, pairs


class GreedyDecoder:
    name = "Greedy matching"

    def __init__(self, graph):
        self.g = graph
        self.dist, self.par, self.bdist, self.bpar = graph.shortest_paths()

    def decode(self, syndrome):
        defects = np.flatnonzero(syndrome[: self.g.n])
        return greedy_match(defects, self.dist, self.par, self.bdist, self.bpar)[0] if len(defects) else 0

    def decode_batch(self, syndromes):
        return np.array([self.decode(s) for s in syndromes], dtype=np.uint8)

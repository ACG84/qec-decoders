"""Round-synchronous local matching decoder.

A parallel relative of greedy matching. In each round every unmatched defect looks only within
radius r: if its nearest unmatched neighbour also chose it (a mutual nearest pair) and joining
them is cheaper than two boundary matches, they pair up; a defect whose boundary is within r, and
for which pairing with its nearest neighbour would cost more than two boundary matches, retires
to the boundary. Then r grows. Every decision in a round uses
only local information and all defects act at once, so a round is a single parallel step and the
number of rounds, not the number of defects, sets the latency.
"""
import numpy as np


class LocalMatchingDecoder:
    name = "Local matching (parallel rounds)"

    def __init__(self, graph, radius_steps=12):
        self.g = graph
        self.dist, self.par, self.bdist, self.bpar = graph.shortest_paths()
        finite = self.dist[np.isfinite(self.dist) & (self.dist > 0)]
        top = max(finite.max(), self.bdist.max())
        self.radii = np.linspace(finite.min(), top, radius_steps)
        self.rounds_used = []

    def decode(self, syndrome):
        defects = list(np.flatnonzero(syndrome[: self.g.n]))
        obs = 0; rounds = 0
        for r in self.radii:
            changed = True
            while changed and defects:
                rounds += 1; changed = False
                k = len(defects); D = self.dist[np.ix_(defects, defects)].copy(); np.fill_diagonal(D, np.inf)
                B = self.bdist[defects]
                nn = D.argmin(axis=1) if k > 1 else np.zeros(k, int)
                nd = D[np.arange(k), nn] if k > 1 else np.full(k, np.inf)
                gone = set()
                for i in range(k):                      # all decisions read the same snapshot
                    j = int(nn[i])
                    if k > 1 and nd[i] <= r and nn[j] == i and i < j and nd[i] <= B[i] + B[j]:
                        obs ^= int(self.par[defects[i], defects[j]]); gone.update((i, j))
                for i in range(k):
                    alone = k == 1 or nd[i] > B[i] + B[int(nn[i])]   # pairing would cost more than two boundary matches
                    if i not in gone and B[i] <= r and alone:
                        obs ^= int(self.bpar[defects[i]]); gone.add(i)
                if gone:
                    defects = [d for t, d in enumerate(defects) if t not in gone]; changed = True
            if not defects:
                break
        for d in defects:                                # anything left goes to the boundary
            obs ^= int(self.bpar[d])
        self.rounds_used.append(rounds)
        return obs

    def decode_batch(self, syndromes):
        return np.array([self.decode(s) for s in syndromes], dtype=np.uint8)

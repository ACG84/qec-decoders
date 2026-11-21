"""Minimum-weight perfect matching decoder, exact, by dynamic programming over defect subsets.

Each defect is paired with another defect or with the boundary so that the total path weight is
minimal. f(S) = min over the lowest defect i in S of
    boundary_dist[i] + f(S - {i})   and   dist[i][j] + f(S - {i, j}) for j in S.
This is O(2^k * k) in the number of defects k, exact, and needs no blossom algorithm.

To keep k small the defects are first split into independent groups: i and j can only share a
group if joining them directly is cheaper than sending both to the boundary. Pairs across groups
are never better than two boundary matches, so solving each group separately is still exact, and
the groups can be solved in parallel. A group larger than `max_defects` falls back to greedy
matching (counted in `fallbacks`).
"""
import numpy as np
from .greedy import greedy_match


class MWPMDecoder:
    name = "MWPM (exact, subset DP)"

    def __init__(self, graph, max_defects=14):
        self.g = graph
        self.max_defects = max_defects
        self.dist, self.par, self.bdist, self.bpar = graph.shortest_paths()
        self.fallbacks = 0

    def decode(self, syndrome):
        defects = np.flatnonzero(syndrome[: self.g.n])
        k = len(defects)
        if k == 0:
            return 0
        D = self.dist[np.ix_(defects, defects)]; P = self.par[np.ix_(defects, defects)]
        B = self.bdist[defects]; BP = self.bpar[defects]
        # independent groups: connect i, j when a direct pairing beats two boundary matches
        comp = list(range(k))
        def find(x):
            while comp[x] != x:
                comp[x] = comp[comp[x]]; x = comp[x]
            return x
        link = D < (B[:, None] + B[None, :])
        for a in range(k):
            for b in range(a + 1, k):
                if link[a, b]:
                    comp[find(a)] = find(b)
        groups = {}
        for a in range(k):
            groups.setdefault(find(a), []).append(a)
        obs = 0
        for idx in groups.values():
            obs ^= self._solve(idx, D, P, B, BP, defects)
        return obs

    def _solve(self, idx, D, P, B, BP, defects):
        k = len(idx)
        if k == 1:
            return int(BP[idx[0]])
        if k > self.max_defects:
            self.fallbacks += 1
            return greedy_match(defects[idx], self.dist, self.par, self.bdist, self.bpar)[0]
        full = (1 << k) - 1
        cost = [0.0] * (full + 1); obs = [0] * (full + 1)
        for mask in range(1, full + 1):
            i = (mask & -mask).bit_length() - 1
            rest = mask & ~(1 << i)
            best, bo = B[idx[i]] + cost[rest], int(BP[idx[i]]) ^ obs[rest]
            m = rest
            while m:
                j = (m & -m).bit_length() - 1
                m &= m - 1
                r2 = rest & ~(1 << j)
                c = D[idx[i], idx[j]] + cost[r2]
                if c < best:
                    best, bo = c, int(P[idx[i], idx[j]]) ^ obs[r2]
            cost[mask], obs[mask] = best, bo
        return obs[full]

    def decode_batch(self, syndromes):
        return np.array([self.decode(s) for s in syndromes], dtype=np.uint8)

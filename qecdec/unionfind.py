"""Union-Find decoder (Delfosse and Nickerson, 2017), unweighted growth.

1. Growth: every cluster that holds an odd number of defects and does not touch the boundary
   grows by half an edge per round. An edge grown from both ends (or twice from one) is added and
   its two clusters are merged with a union-find structure. Stop when no odd cluster is left.
2. Peeling: inside each cluster take a spanning tree of the grown edges (rooted at a boundary
   vertex if the cluster has one) and peel it from the leaves: a leaf holding a defect uses its
   tree edge, which moves the defect to its parent.
The correction's effect on the logical observable is the XOR of the used edges' labels.
Growth is local to each cluster, which is what makes the decoder suited to parallel hardware.
"""
from collections import deque
import numpy as np


class UnionFindDecoder:
    name = "Union-Find"

    def __init__(self, graph):
        self.g = graph

    def decode(self, syndrome):
        g = self.g
        defects = np.flatnonzero(syndrome[: g.n])
        if len(defects) == 0:
            return 0
        N = g.num_nodes
        parent = list(range(N))
        parity = [0] * N          # defects in the cluster, mod 2 (valid at roots)
        boundary = [False] * N    # cluster contains a boundary vertex (valid at roots)
        member = [False] * N      # node belongs to some cluster
        for d in defects:
            parity[d] = 1; member[d] = True

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        growth = np.zeros(len(g.eu), dtype=np.int8)
        full = np.zeros(len(g.eu), dtype=bool)
        eu, ev = g.eu, g.ev
        while True:
            active = set()
            for d in defects:
                r = find(d)
                if parity[r] and not boundary[r]:
                    active.add(r)
            if not active:
                break
            grew = []
            for e in np.flatnonzero(~full):
                u, v = int(eu[e]), int(ev[e])
                inc = (member[u] and find(u) in active) + (member[v] and find(v) in active)
                if inc:
                    growth[e] += inc
                    if growth[e] >= 2:
                        grew.append(e)
            for e in grew:
                full[e] = True
                u, v = int(eu[e]), int(ev[e])
                for x in (u, v):
                    if not member[x]:
                        member[x] = True
                        if g.is_boundary(x):
                            boundary[x] = True
                ru, rv = find(u), find(v)
                if ru != rv:
                    parent[ru] = rv
                    parity[rv] ^= parity[ru]
                    boundary[rv] = boundary[rv] or boundary[ru]
        # peeling on the forest of fully grown edges
        tree_adj = {}
        for e in np.flatnonzero(full):
            u, v = int(eu[e]), int(ev[e])
            tree_adj.setdefault(u, []).append((v, e)); tree_adj.setdefault(v, []).append((u, e))
        synd = {int(d): 1 for d in defects}
        seen = set(); obs = 0
        roots = sorted(tree_adj, key=lambda x: not g.is_boundary(x))  # boundary vertices first
        for root in roots:
            if root in seen:
                continue
            order = []; par_edge = {root: (None, None)}; q = deque([root]); seen.add(root)
            while q:
                u = q.popleft(); order.append(u)
                for v, e in tree_adj[u]:
                    if v not in seen:
                        seen.add(v); par_edge[v] = (u, e); q.append(v)
            for u in reversed(order):
                pu, e = par_edge[u]
                if pu is not None and synd.get(u, 0):
                    obs ^= int(g.eobs[e])
                    synd[u] = 0
                    if not g.is_boundary(pu):
                        synd[pu] = synd.get(pu, 0) ^ 1
        return obs

    def decode_batch(self, syndromes):
        return np.array([self.decode(s) for s in syndromes], dtype=np.uint8)

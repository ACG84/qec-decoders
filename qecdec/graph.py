"""Decoding graph built from a Stim detector error model.

Nodes are detectors. Every error mechanism that flips two detectors is an edge; one that flips a
single detector is an edge to its own private boundary vertex (so clusters never merge through a
shared boundary). Each edge carries its probability, weight log((1-p)/p) and whether it flips the
logical observable.
"""
import heapq
import math
import numpy as np
import stim


def surface_code_circuit(distance=3, rounds=None, p=0.005, basis="z"):
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}", distance=distance, rounds=rounds or distance,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)


def repetition_code_circuit(distance=5, rounds=None, p=0.01):
    return stim.Circuit.generated(
        "repetition_code:memory", distance=distance, rounds=rounds or distance,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)


def sample(circuit, shots, seed=0):
    d, o = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return d.astype(np.uint8), o[:, 0].astype(np.uint8)


class DecodingGraph:
    def __init__(self, circuit):
        self.circuit = circuit
        self.n = circuit.num_detectors
        merged = {}
        dem = circuit.detector_error_model(decompose_errors=True).flattened()
        for inst in dem:
            if inst.type != "error":
                continue
            p = inst.args_copy()[0]
            dets, obs, parts = [], False, []
            for t in inst.targets_copy():
                if t.is_separator():
                    parts.append((dets, obs)); dets, obs = [], False
                elif t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id() and t.val == 0:
                    obs = not obs
            parts.append((dets, obs))
            for dets, obs in parts:
                if len(dets) == 0 or len(dets) > 2:
                    continue  # nothing to match on (or not graph-like)
                key = (min(dets), max(dets) if len(dets) == 2 else -1, obs)
                q = merged.get(key, 0.0)
                merged[key] = q * (1 - p) + p * (1 - q)
        # keep the likelier observable label when two parallel edges disagree
        best = {}
        for (u, v, obs), p in merged.items():
            if (u, v) not in best or p > best[(u, v)][0]:
                best[(u, v)] = (p, obs)
        self.eu, self.ev, self.ep, self.eobs = [], [], [], []
        nb = 0
        for (u, v), (p, obs) in sorted(best.items()):
            if v == -1:
                v = self.n + nb; nb += 1
            self.eu.append(u); self.ev.append(v); self.ep.append(p); self.eobs.append(obs)
        self.num_boundary = nb
        self.num_nodes = self.n + nb
        self.eu = np.array(self.eu); self.ev = np.array(self.ev)
        self.ep = np.array(self.ep); self.eobs = np.array(self.eobs, dtype=np.uint8)
        self.ew = np.log((1 - self.ep) / self.ep)
        self.adj = [[] for _ in range(self.num_nodes)]
        for e, (u, v) in enumerate(zip(self.eu, self.ev)):
            self.adj[u].append((v, e)); self.adj[v].append((u, e))
        self._paths = None

    def is_boundary(self, node):
        return node >= self.n

    def shortest_paths(self):
        """All-pairs detector-to-detector distances that avoid boundary vertices, the distance of
        every detector to its nearest boundary, and the observable parity along each of those
        paths. Dijkstra from every detector; computed once and cached."""
        if self._paths is None:
            n = self.n
            dist = np.full((n, n), np.inf); par = np.zeros((n, n), np.uint8)
            bdist = np.full(n, np.inf); bpar = np.zeros(n, np.uint8)
            for s in range(n):
                d = {s: 0.0}; pr = {s: 0}; heap = [(0.0, s)]; done = set()
                while heap:
                    du, u = heapq.heappop(heap)
                    if u in done:
                        continue
                    done.add(u)
                    if self.is_boundary(u):
                        if du < bdist[s]:
                            bdist[s], bpar[s] = du, pr[u]
                        continue  # paths do not continue through the boundary
                    dist[s, u], par[s, u] = du, pr[u]
                    for v, e in self.adj[u]:
                        nd = du + self.ew[e]
                        if nd < d.get(v, math.inf):
                            d[v] = nd; pr[v] = pr[u] ^ int(self.eobs[e]); heapq.heappush(heap, (nd, v))
            self._paths = (dist, par, bdist, bpar)
        return self._paths

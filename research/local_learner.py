"""Noise-robust learning at high qubit counts, for codes with local checks.

Two changes from scalable_learner.py:
 1. Two copies per sample, not four. A Bell sample P of a code state satisfies
    <P, g> = (number of Y's in g) mod 2 for every code stabilizer g, whatever the logical state.
    So each stabilizer commutes with ALL samples or anticommutes with ALL samples: its commute
    fraction is 0 or 1, and anything else sits near 1/2. No differences needed.
 2. Local search. For each small neighbourhood of the (known) qubit layout, score every Pauli
    supported there and accept those whose commute fraction is far from 1/2. A weight-w check
    is only disturbed by noise on its own 2w qubit copies, so the score does not degrade with n.
Tested on rotated surface codes (weight-4 and weight-2 checks) with a random logical state per shot.
"""
import itertools, sys, time
import numpy as np
import stim
from scalable_learner import rank, same_span, noisy_measure

rng = np.random.default_rng(3)

def surface_code(d):
    """Rotated surface code on a d x d grid. Returns (generators as (x|z) rows, neighbourhoods)."""
    n = d * d; q = lambda r, c: r * d + c; gens, hoods = [], []
    for i in range(-1, d):
        for j in range(-1, d):
            cells = [q(r, c) for r in (i, i + 1) for c in (j, j + 1) if 0 <= r < d and 0 <= c < d]
            if len(cells) < 2: continue
            hoods.append(cells); typ = "X" if (i + j) % 2 == 0 else "Z"
            bulk = len(cells) == 4
            edge = len(cells) == 2 and ((typ == "X" and i in (-1, d - 1)) or (typ == "Z" and j in (-1, d - 1)))
            if bulk or edge:
                v = np.zeros(2 * n, np.uint8); v[[c + (0 if typ == "X" else n) for c in cells]] = 1; gens.append(v)
    return np.array(gens), hoods

class GenCode:
    def __init__(self, G, n):
        self.n, self.G = n, G
        ps = [stim.PauliString.from_numpy(xs=g[:n].astype(bool), zs=g[n:].astype(bool)) for g in G]
        self.T = stim.Tableau.from_stabilizers(ps, allow_underconstrained=True); self.k = n - len(G)
    def load(self, sim, block):
        qs = list(range(block * self.n, (block + 1) * self.n))
        sim.do_tableau(self._logical, qs[self.n - self.k:]); sim.do_tableau(self.T, qs)

def bell_sample(code, p):
    n = code.n; sim = stim.TableauSimulator(seed=int(rng.integers(2**31))); code._logical = stim.Tableau.random(code.k)
    code.load(sim, 0); code.load(sim, 1)
    if p: sim.depolarize1(*range(2 * n), p=p)
    A, B = list(range(n)), list(range(n, 2 * n))
    sim.cnot(*[x for pair in zip(A, B) for x in pair]); sim.h(*A)
    z = noisy_measure(sim, A, p); x = noisy_measure(sim, B, p)
    return np.concatenate([x, z])

def learn_local(D, n, hoods, sigmas=5.0):
    """Score every Pauli on every neighbourhood; accept |commute fraction - 1/2| above `sigmas`
    standard deviations of a fair coin over len(D) samples (5 sigma keeps false accepts
    negligible across the ~10^4-10^5 candidates tested)."""
    found = {}; gap = sigmas * 0.5 / np.sqrt(len(D))
    for cells in hoods:
        w = len(cells); cols = cells + [c + n for c in cells]
        Dr = D[:, cols].astype(np.int32)                       # samples restricted to the neighbourhood
        P = np.array(list(itertools.product((0, 1), repeat=2 * w)), dtype=np.int32)[1:]
        anti = (P[:, :w] @ Dr[:, w:].T + P[:, w:] @ Dr[:, :w].T) % 2   # (paulis, samples)
        frac = 1 - anti.mean(axis=1)
        for idx in np.flatnonzero(np.abs(frac - 0.5) > gap):
            v = np.zeros(2 * n, np.uint8); v[cols] = P[idx]; found[v.tobytes()] = (v, float(frac[idx]))
    return [v for v, _ in found.values()], [f for _, f in found.values()]

if __name__ == "__main__":
    print("Rotated surface code, random logical state per shot, 2 copies per sample")
    print("    d  qubits    p   samples | accepted | all true stabilizers | spans full group | weakest accepted score | time")
    for d in (3, 5, 7, 9, 11, 15):
        G, hoods = surface_code(d); n = d * d; code = GenCode(G, n)
        for p, m in ((0.0, 100), (0.01, 300), (0.03, 300), (0.05, 300), (0.05, 1000), (0.08, 1000)):
            t0 = time.time(); D = np.array([bell_sample(code, p) for _ in range(m)])
            L, F = learn_local(D, n, hoods); L = np.array(L) if L else np.zeros((0, 2 * n), np.uint8)
            genuine = len(L) > 0 and rank(np.vstack([G, L])) == rank(G)
            full = genuine and rank(L) == rank(G)
            weakest = min(abs(f - 0.5) for f in F) + 0.5 if F else float("nan")
            print(f"   {d:2d}  {n:4d}   {p:5.2f}  {m:5d}  | {len(L):5d}    | {str(genuine):5s}                | {str(full):5s}            | {weakest:.2f}                   | {time.time() - t0:5.1f}s", flush=True)

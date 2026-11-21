"""Scalable version of the Bell-sampling code learner: polynomial in n, any [[n, k]] code.

Random stabilizer code on n qubits with k logical qubits; each shot is a random logical
stabilizer state. Bell difference samples lie in the normalizer N(G) of the code's stabilizer
group G (dimension n + k as a GF(2) space). So:
    G = symplectic complement of span(samples)          -- Gaussian elimination, O(n^3)
No enumeration of Paulis. k is not given: sampling stops when the span has stopped growing.

With noise, a corrupted sample leaves N(G) and pollutes the span. `learn_noisy` handles that
by information-set search: draw small random subsets, keep one whose complement has the right
shape and whose generators commute with a clear majority of ALL samples. Its cost grows
exponentially with (noise rate x qubits), which is the learning-parity-with-noise barrier.
"""
import sys, time
import numpy as np
import stim

rng = np.random.default_rng(2)

# ---------- GF(2) linear algebra ----------
def rref(A):
    A = A.copy() % 2; piv = []; r = 0
    for c in range(A.shape[1]):
        if r == len(A): break
        rows = np.flatnonzero(A[r:, c])
        if len(rows) == 0: continue
        A[[r, r + rows[0]]] = A[[r + rows[0], r]]
        mask = A[:, c].astype(bool); mask[r] = False
        A[mask] ^= A[r]
        piv.append(c); r += 1
    return A[:r], piv

def nullspace(A):
    R, piv = rref(A); m = A.shape[1]; free = [c for c in range(m) if c not in piv]
    N = np.zeros((len(free), m), np.uint8)
    for i, f in enumerate(free):
        N[i, f] = 1
        for j, pc in enumerate(piv): N[i, pc] = R[j, f]
    return N

def rank(A): return len(rref(A)[0]) if len(A) else 0
def swap(V, n): return np.hstack([V[:, n:], V[:, :n]])
def sym_complement(B, n): return nullspace(swap(B, n))
def commute_matrix(A, B, n): return (A.astype(np.int32) @ swap(B, n).T.astype(np.int32)) % 2 == 0
def same_span(A, B): return rank(A) == rank(B) == rank(np.vstack([A, B]))

# ---------- simulation ----------
class Code:
    def __init__(self, n, k):
        self.n, self.k = n, k; self.T = stim.Tableau.random(n)
        self.G = np.array([np.concatenate(self.T.z_output(i).to_numpy()) for i in range(n - k)], dtype=np.uint8)
    def load(self, sim, block, cls):
        qs = list(range(block * self.n, (block + 1) * self.n))
        if cls == 1:
            if self._logical is not None: sim.do_tableau(self._logical, qs[self.n - self.k:])
            sim.do_tableau(self.T, qs)
        else: sim.do_tableau(self._rand, qs)
    def new_state(self):
        self._logical = stim.Tableau.random(self.k) if self.k else None; self._rand = stim.Tableau.random(self.n)

def noisy_measure(sim, qs, p):
    if p: sim.x_error(*qs, p=p)
    return np.array(sim.measure_many(*qs), dtype=np.uint8)

def bell_difference(code, p):
    n = code.n; sim = stim.TableauSimulator(seed=int(rng.integers(2**31))); code.new_state()
    for b in range(4): code.load(sim, b, 1)
    if p: sim.depolarize1(*range(4 * n), p=p)
    out = np.zeros(2 * n, np.uint8)
    for a, b in ((0, 1), (2, 3)):
        A = list(range(a * n, a * n + n)); B = list(range(b * n, b * n + n))
        sim.cnot(*[q for pair in zip(A, B) for q in pair]); sim.h(*A)
        z = noisy_measure(sim, A, p); x = noisy_measure(sim, B, p); out ^= np.concatenate([x, z])
    return out

# ---------- learners ----------
def learn_clean(code, patience=8, max_samples=2000):
    """Sample until the span stops growing. Returns (generators of G, samples used)."""
    S = np.zeros((0, 2 * code.n), np.uint8); r = 0; still = 0; m = 0
    while still < patience and m < max_samples:
        v = bell_difference(code, 0.0); m += 1
        cand = np.vstack([S, v]); rr = rank(cand)
        if rr > r: S, r, still = rref(cand)[0], rr, 0
        else: still += 1
    return sym_complement(S, code.n), m

def learn_noisy(D, n, k, tries=4000, margin=0.07):
    """Information-set search over noisy samples D. Returns (generators or None, tries used)."""
    best, best_score = None, 0.0
    for t in range(1, tries + 1):
        sub = D[rng.choice(len(D), n + k + 2, replace=False)]
        C = sym_complement(sub, n)
        if len(C) != n - k or not commute_matrix(C, C, n).all(): continue
        score = commute_matrix(C, D, n).mean(axis=1).min()
        if score > best_score: best, best_score = C, score
        if score > 2 / 3 + margin: return C, t, score
    return None, tries, best_score

def clause_test(code, Ggen, p, shots=300):
    """One-copy entangling clause from learned generators: (miss rate, false-accept rate)."""
    n = code.n; ps = [stim.PauliString.from_numpy(xs=g[:n].astype(bool), zs=g[n:].astype(bool)) for g in Ggen]
    Tinv = stim.Tableau.from_stabilizers(ps, allow_underconstrained=True).inverse(); r = len(ps)
    def run(cls, m):
        out = np.zeros((m, r), np.uint8)
        for j in range(m):
            sim = stim.TableauSimulator(seed=int(rng.integers(2**31))); code.new_state(); code.load(sim, 0, cls)
            if p: sim.depolarize1(*range(n), p=p)
            sim.do_tableau(Tinv, list(range(n))); out[j] = noisy_measure(sim, list(range(r)), p)
        return out
    target = (run(1, 30).mean(axis=0) > 0.5).astype(np.uint8)
    return float((~np.all(run(1, shots) == target, axis=1)).mean()), float(np.all(run(0, shots) == target, axis=1).mean())

if __name__ == "__main__":
    print("A. Noise-free scaling (5 random codes per size)")
    print("     n   k | samples used (mean) | copies | exact recovery | learn time | clause: miss / false accept")
    for n, k in ((5, 1), (10, 1), (20, 2), (40, 4), (80, 8), (120, 12)):
        used, ok, tt = [], [], []
        for _ in range(5):
            code = Code(n, k); t0 = time.time(); Gl, m = learn_clean(code); tt.append(time.time() - t0)
            used.append(m); ok.append(len(Gl) == n - k and same_span(Gl, code.G))
        miss, fa = clause_test(code, Gl, 0.0, shots=200) if ok[-1] else (float("nan"),) * 2
        print(f"   {n:4d} {k:3d} | {np.mean(used):8.1f}            | {4 * np.mean(used):6.0f} | {np.mean(ok):.2f}           | {np.mean(tt):6.2f}s    | {miss:.3f} / {fa:.3f}", flush=True)
    print("B. With noise: information-set search, 400 Bell-difference samples, up to 4000 subsets (5 codes per row)")
    print("     n   k    p   | exact recovery | subsets tried (median) | time   | clause: miss / false accept")
    for n, k, p in ((5, 1, 0.01), (5, 1, 0.03), (8, 1, 0.01), (12, 1, 0.005), (12, 1, 0.01), (16, 2, 0.005), (24, 2, 0.005)):
        ok, tries, tt = [], [], []; last = None
        for _ in range(5):
            code = Code(n, k); D = np.array([bell_difference(code, p) for _ in range(400)])
            t0 = time.time(); Gl, t, sc = learn_noisy(D, n, k); tt.append(time.time() - t0); tries.append(t)
            good = Gl is not None and same_span(Gl, code.G); ok.append(good)
            if good: last = (code, Gl)
        miss, fa = clause_test(*last, p, shots=300) if last else (float("nan"),) * 2
        print(f"   {n:4d} {k:3d} {p:6.3f} | {np.mean(ok):.2f}           | {int(np.median(tries)):6d}                 | {np.mean(tt):5.1f}s | {miss:.3f} / {fa:.3f}", flush=True)

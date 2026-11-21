"""Entangling Clifford clauses learned by Bell difference sampling, with noise.

Same task as pauli_clauses.py (five-qubit-code states vs random stabilizer states), now with
depolarizing noise on every qubit and readout flips, both with probability p.

Learning (code class only):
  1. Bell difference sampling. Four copies of the shot's state: Bell-measure copies (1,2) and
     (3,4) qubit by qubit; the XOR of the two outcome Paulis lies in the state's stabilizer group.
  2. Score every Pauli g by the fraction of differences it commutes with. Code stabilizers
     score ~1 (less under noise), logical operators 2/3 (they anticommute with the shot's
     logical stabilizer two times in three, on half the samples), everything else ~1/2.
     Accept g above the midpoint between the top scores and 2/3. Sign-free at this point.
  3. Build the Clifford circuit that maps four independent accepted stabilizers to Z1..Z4, run
     it on a few single copies and read off the constant outcome bits (this fixes the signs).
Clause = that circuit followed by AND(four measured bits match). It checks all four stabilizers
on ONE copy. The product-basis alternative needs one copy per check.
"""
import itertools, sys
import numpy as np
import stim

N = 5
GENS = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
LOGICALS = ["+XXXXX", "-XXXXX", "+ZZZZZ", "-ZZZZZ", "+YYYYY", "-YYYYY"]
PREPS = [stim.Tableau.from_stabilizers([stim.PauliString(g) for g in GENS] + [stim.PauliString(l)]) for l in LOGICALS]
rng = np.random.default_rng(1)

def prepare(sim, cls, blocks, p, state=None):
    """Put the same state on each block of N qubits, then apply independent noise."""
    t = (PREPS[rng.integers(6)] if cls == 1 else stim.Tableau.random(N)) if state is None else state
    for b in blocks:
        sim.do_tableau(t, list(range(b * N, b * N + N)))
    if p: sim.depolarize1(*range(len(blocks) * N), p=p)
    return t

def measure(sim, qubits, p):
    if p: sim.x_error(*qubits, p=p)
    return np.array(sim.measure_many(*qubits), dtype=np.uint8)

def bell_difference(cls, p):
    sim = stim.TableauSimulator(seed=int(rng.integers(2**31))); prepare(sim, cls, range(4), p)
    out = np.zeros(2 * N, np.uint8)
    for a, b in ((0, 1), (2, 3)):
        for i in range(N): sim.cnot(a * N + i, b * N + i); sim.h(a * N + i)
        z = measure(sim, [a * N + i for i in range(N)], p); x = measure(sim, [b * N + i for i in range(N)], p)
        out ^= np.concatenate([x, z])
    return out  # (x | z) of a Pauli, phase ignored

ALL = np.array([np.concatenate([[c in "XY" for c in s], [c in "ZY" for c in s]]) for s in map("".join, itertools.product("IXYZ", repeat=N))], dtype=np.uint8)[1:]
def to_str(v): return "".join("IXZY"[int(v[i]) + 2 * int(v[N + i])] for i in range(N))
def commutes(A, B): return ((A[:, :N] @ B[:, N:].T + A[:, N:] @ B[:, :N].T) % 2) == 0
TRUE = set()
for a in range(1, 16):
    ps = stim.PauliString(N)
    for i in range(4):
        if a >> i & 1: ps *= stim.PauliString(GENS[i])
    TRUE.add(str(ps)[1:].replace("_", "I"))

def learn_group(m, p):
    D = np.array([bell_difference(1, p) for _ in range(m)])
    frac = commutes(ALL, D).mean(axis=1)
    thr = (np.sort(frac)[-5:].mean() + 2 / 3) / 2
    return {to_str(v) for v in ALL[frac > thr]}

def is_valid_group(group):
    """Certificate that needs no ground truth: non-empty, mutually commuting, closed under products."""
    if not group: return False
    V = np.array([np.concatenate([[c in "XY" for c in s], [c in "ZY" for c in s]]) for s in sorted(group)], dtype=np.uint8)
    if not commutes(V, V).all(): return False
    return all(to_str(a ^ b) in group for i, a in enumerate(V) for b in V[i + 1:])

def independent_generators(group):
    basis, gens = [], []
    for s in sorted(group):
        v = np.concatenate([[c in "XY" for c in s], [c in "ZY" for c in s]]).astype(np.uint8); r = v.copy()
        for b in basis:
            piv = int(np.flatnonzero(b)[0])
            if r[piv]: r ^= b
        if r.any(): basis.append(r); gens.append(s)
    return gens

def build_clause(gens, p, n_sign=40):
    T = stim.Tableau.from_stabilizers([stim.PauliString(g) for g in gens], allow_underconstrained=True).inverse()
    k = len(gens)
    def run(cls, n):
        out = np.zeros((n, k), np.uint8)
        for j in range(n):
            sim = stim.TableauSimulator(seed=int(rng.integers(2**31))); prepare(sim, cls, [0], p)
            sim.do_tableau(T, list(range(N))); out[j] = measure(sim, list(range(k)), p)
        return out
    target = (run(1, n_sign).mean(axis=0) > 0.5).astype(np.uint8)
    return lambda cls, n: np.all(run(cls, n) == target, axis=1)

def product_clause(gens, p, n_sign=40):
    """One copy per check: measure each generator in its own product basis."""
    def run(cls, n):
        out = np.zeros((n, len(gens)), np.uint8)
        for j in range(n):
            sim = stim.TableauSimulator(seed=int(rng.integers(2**31))); prepare(sim, cls, range(len(gens)), p)
            for b, g in enumerate(gens):
                qs = []
                for i, c in enumerate(g):
                    q = b * N + i
                    if c == "X": sim.h(q)
                    elif c == "Y": sim.s_dag(q); sim.h(q)
                    if c != "I": qs.append(q)
                out[j, b] = measure(sim, qs, p).sum() % 2
        return out
    target = (run(1, n_sign).mean(axis=0) > 0.5).astype(np.uint8)
    return lambda cls, n: np.all(run(cls, n) == target, axis=1)

if __name__ == "__main__":
    print("A. Copies needed to recover the code's stabilizer group exactly (30 trials each; 4 copies per sample)")
    for p in (0.0, 0.01, 0.03):
        print(f"   p={p}: " + "  ".join(f"{m} samples: {np.mean([learn_group(m, p) == TRUE for _ in range(30)]):.2f}" for m in (8, 16, 32, 64, 128, 256)))
    print("B. Classification from the learned clause (256 Bell-difference samples; 3000 test shots per class)")
    NT = 3000
    for p in (0.0, 0.01, 0.03):
        G = learn_group(256, p); gens = independent_generators(G)
        line = f"   p={p}: learned {len(G)} elements, passes group certificate={is_valid_group(G)}, equals true group={G == TRUE}, generators {gens}\n"
        if not is_valid_group(G):
            print(line + "      learner output rejected by the certificate; no clause built"); continue
        for label, mk in (("entangling Clifford clause, 1 copy ", build_clause), ("product-basis clause, 4 copies     ", product_clause)):
            f = mk(gens, p); miss = (~f(1, NT)).mean(); fa = f(0, NT).mean()
            line += f"      {label} misses code state {miss:.3f} | accepts random state {fa:.3f} | balanced error {(miss + fa) / 2:.3f}\n"
        print(line, end="", flush=True)

"""Clifford clauses on QUANTUM data: letting each literal pick its Pauli basis.

Task: tell 5-qubit-code states (random logical state each shot) from random stabilizer states,
from single-shot measurements. A literal is a parity of outcomes after a single-qubit Clifford
on each qubit, i.e. a Pauli-product measurement (per qubit: X, Y or Z, and a sign).
The proposer is a GF(2) null-space search: for a given basis setting it returns every parity
(with sign) that holds on all training shots of the code class. A clause ANDs those parities.
  - "Z basis only" is what an ordinary Tsetlin machine sees: computational-basis bits.
  - "one learned basis" lets the clause choose single-qubit Cliffords (3 bases per qubit).
  - "k copies" measures different copies in different learned bases and ANDs all checks.
"""
import itertools, sys
import numpy as np
import stim

N = 5
GENS = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
LOGICALS = ["+XXXXX", "-XXXXX", "+ZZZZZ", "-ZZZZZ", "+YYYYY", "-YYYYY"]
PREPS = [stim.Tableau.from_stabilizers([stim.PauliString(g) for g in GENS] + [stim.PauliString(l)]).to_circuit() for l in LOGICALS]
rng = np.random.default_rng(0)

def shot(cls, basis):
    sim = stim.TableauSimulator(seed=int(rng.integers(2**31)))
    if cls == 1: sim.do_circuit(PREPS[rng.integers(6)])
    else: sim.do_tableau(stim.Tableau.random(N), list(range(N)))
    for q, b in enumerate(basis):
        if b == "X": sim.h(q)
        elif b == "Y": sim.s_dag(q); sim.h(q)
    return sim.measure_many(*range(N))

def sample(cls, basis, n): return np.array([shot(cls, basis) for _ in range(n)], dtype=np.uint8)

def affine_nullspace(B):
    """All independent (w, c) with B @ w = c (mod 2) on every row: parities that never vary."""
    A = np.hstack([B, np.ones((len(B), 1), np.uint8)]) % 2
    m = A.shape[1]; R = A.copy(); piv = []; r = 0
    for col in range(m):
        rows = np.flatnonzero(R[r:, col]) + r
        if len(rows) == 0: continue
        R[[r, rows[0]]] = R[[rows[0], r]]
        for i in range(len(R)):
            if i != r and R[i, col]: R[i] ^= R[r]
        piv.append(col); r += 1
        if r == len(R): break
    free = [c for c in range(m) if c not in piv]; out = []
    for f in free:
        w = np.zeros(m, np.uint8); w[f] = 1
        for i, pc in enumerate(piv): w[pc] = R[i, f]
        if w[:-1].any(): out.append((w[:-1].copy(), int(w[-1])))
    return out

def checks_pass(B, checks): return np.all([(B @ w) % 2 == c for w, c in checks], axis=0) if checks else np.ones(len(B), bool)
def name(basis, w, c): return ("-" if c else "+") + "".join(b if wi else "I" for b, wi in zip(basis, w))

NTRAIN, NTEST = 40, 3000
found = {}
for basis in itertools.product("XYZ", repeat=N):
    ch = affine_nullspace(sample(1, basis, NTRAIN))
    if ch: found["".join(basis)] = ch
def evaluate(label, bases):
    ok1 = np.ones(NTEST, bool); ok0 = np.ones(NTEST, bool); n = 0
    for b in bases:  # one fresh copy of the state per basis
        ch = found.get(b, []); n += len(ch)
        ok1 &= checks_pass(sample(1, b, NTEST), ch); ok0 &= checks_pass(sample(0, b, NTEST), ch)
    err = 0.5 * ((~ok1).mean() + ok0.mean())
    print(f"  {label:34s} checks {n} | misses code state {(~ok1).mean():.3f} | accepts random state {ok0.mean():.3f} | balanced error {err:.3f}")
print(f"5-qubit code vs random stabilizer states; {NTRAIN} training shots per basis, {NTEST} test shots per class")
print(f"  bases with at least one learned check: {len(found)} of 243; Z basis has {len(found.get('ZZZZZ', []))}")
evaluate("Z basis only (ordinary literals)", ["ZZZZZ"])
best = sorted(found, key=lambda b: -len(found[b]))
evaluate("one learned basis", best[:1])
chosen, span = [], []
for b in best:  # greedy: add bases whose checks are new Pauli strings
    names = {name(b, w, c)[1:] for w, c in found[b]}
    if not names <= set(span): chosen.append(b); span += list(names)
    if len(chosen) == 4: break
for k in (2, 3, 4): evaluate(f"{k} copies, {k} learned bases", chosen[:k])
print("  learned checks:", "; ".join(f"basis {b}: " + ", ".join(name(b, w, c) for w, c in found[b]) for b in chosen))
G = set()
for a in range(1, 16):
    p = stim.PauliString(N)
    for i in range(4):
        if a >> i & 1: p *= stim.PauliString(GENS[i])
    G.add(str(p))
allc = {name(b, w, c).replace("I", "_") for b in found for w, c in found[b]}
print(f"  distinct learned checks: {len(allc)}; all are true stabilizers or logical-free products: {allc <= G}")

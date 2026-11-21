"""Do learned Clifford (parity) literals help?  python clifford_experiment.py DISTANCE P"""
import sys, time, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier
from stm.tm import TsetlinMachine
from stm import qec, clifford
d, p = int(sys.argv[1]), float(sys.argv[2]); NTR, NTE = 30000, 100000 if d == 3 else 50000
c = qec.memory_circuit(d, d, p); Dtr, ytr = qec.sample(c, NTR, 11); Dte, yte = qec.sample(c, NTE, 22)
M, pnames = qec.parity_features(c); dn = qec.detector_names(c)
hand = {frozenset(np.flatnonzero(r).tolist()) for r in M}
print(f"d={d} p={p}: no decoding {yte.mean():.4f}, MWPM {(qec.mwpm_predict(c, Dte) != yte).mean():.4f}")
def evaluate(label, Xtr, Xte, cfg):
    tree = DecisionTreeClassifier(max_leaf_nodes=16, random_state=0).fit(Xtr, ytr)
    nv = NTR // 5; t0 = time.time()
    tm = TsetlinMachine(seed=1, **cfg).fit(Xtr[:-nv], ytr[:-nv], epochs=8, val=(Xtr[-nv:], ytr[-nv:]))
    print(f"  {label:44s} literals {Xtr.shape[1]:4d} | TM {(tm.predict(Xte) != yte).mean():.4f} | 16-leaf tree {(tree.predict(Xte) != yte).mean():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return tm
evaluate("raw detectors (propositional clauses)", Dtr, Dte, dict(n_clauses=300, T=15, s=15.0))
evaluate("hand-written stabilizer parities", qec.apply_parities(Dtr, M), qec.apply_parities(Dte, M), dict(n_clauses=40, T=5, s=4.0))
for R in (1, 2, d, d + 2):
    S = clifford.grow_parities(Dtr, rounds=R)
    found = len(hand & set(S)); sizes = np.bincount([len(s) for s in S])
    Xtr, Xte = clifford.apply_supports(Dtr, S), clifford.apply_supports(Dte, S)
    tm = evaluate(f"learned Clifford literals, {R} CNOT rounds", Xtr, Xte, dict(n_clauses=100, T=10, s=8.0))
    print(f"      parity sizes {dict(enumerate(sizes.tolist()))}; rediscovered {found}/{len(hand)} hand-written stabilizer parities")
    # learned literals only (drop the raw bits): is the Clifford layer alone enough?
    keep = [i for i, s in enumerate(S) if len(s) > 1]
    evaluate(f"   same, parity literals only", Xtr[:, keep], Xte[:, keep], dict(n_clauses=100, T=10, s=8.0))

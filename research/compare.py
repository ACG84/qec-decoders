"""Same data, other decoders: matching, and standard learned classifiers on both feature sets.
    python compare.py DISTANCE P"""
import sys, time, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from stm.tm import TsetlinMachine
from stm import qec
d, p = int(sys.argv[1]), float(sys.argv[2]); NTR, NTE = 30000, 100000 if d == 3 else 50000
c = qec.memory_circuit(d, d, p); Dtr, ytr = qec.sample(c, NTR, 11); Dte, yte = qec.sample(c, NTE, 22)
M, _ = qec.parity_features(c); Ptr, Pte = qec.apply_parities(Dtr, M), qec.apply_parities(Dte, M)
print(f"d={d} p={p}  ({Dtr.shape[1]} detectors, {Ptr.shape[1]} parity bits, {NTR} training shots)")
print(f"  {'no decoding':42s} {yte.mean():.4f}")
print(f"  {'MWPM (PyMatching, knows the noise model)':42s} {(qec.mwpm_predict(c, Dte) != yte).mean():.4f}")
models = {"logistic regression": lambda: LogisticRegression(max_iter=500),
          "decision tree, <=16 leaves (readable)": lambda: DecisionTreeClassifier(max_leaf_nodes=16, random_state=0),
          "decision tree, unlimited": lambda: DecisionTreeClassifier(random_state=0),
          "gradient-boosted trees": lambda: HistGradientBoostingClassifier(random_state=0),
          "MLP 2x128": lambda: MLPClassifier((128, 128), max_iter=60, random_state=0)}
for fname, (Xtr, Xte) in {"parity bits": (Ptr, Pte), "raw detectors": (Dtr, Dte), "both": (np.hstack([Ptr, Dtr]), np.hstack([Pte, Dte]))}.items():
    print(f"  -- features: {fname}")
    for name, mk in models.items():
        t0 = time.time(); m = mk().fit(Xtr, ytr); e = (m.predict(Xte) != yte).mean()
        extra = f"  leaves {m.get_n_leaves()}" if hasattr(m, "get_n_leaves") else ""
        print(f"     {name:39s} {e:.4f}  ({time.time() - t0:.0f}s){extra}", flush=True)
    if fname != "both":
        cfg = dict(n_clauses=40, T=5, s=4.0) if fname == "parity bits" else dict(n_clauses=300, T=15, s=15.0)
        tm = TsetlinMachine(seed=1, **cfg).fit(Xtr[:24000], ytr[:24000], epochs=8, val=(Xtr[24000:], ytr[24000:]))
        print(f"     {'Tsetlin machine (ours)':39s} {(tm.predict(Xte) != yte).mean():.4f}  rules {len(set((pl, tuple(l)) for pl, l in tm.rules([str(i) for i in range(Xtr.shape[1])])))}")

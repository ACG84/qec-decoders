"""Interpretable syndrome rules: a Tsetlin machine as a decoder for a surface-code memory.
    python run_experiment.py [distance] [train_shots] [test_shots]
Compares logical error rate of: no decoding, minimum-weight perfect matching (PyMatching),
a majority lookup table on stabilizer parities, and Tsetlin machines on three feature sets."""
import json, sys, time
import numpy as np
from stm.tm import TsetlinMachine
from stm import qec

d = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NTR = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
NTE = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
results = {}
for p in (0.002, 0.005, 0.01):
    c = qec.memory_circuit(d, d, p)
    Dtr, ytr = qec.sample(c, NTR, seed=11); Dte, yte = qec.sample(c, NTE, seed=22)
    M, pnames = qec.parity_features(c); dnames = qec.detector_names(c)
    Ptr, Pte = qec.apply_parities(Dtr, M), qec.apply_parities(Dte, M)
    row = {"no decoding": float(yte.mean()), "MWPM (PyMatching)": float((qec.mwpm_predict(c, Dte) != yte).mean()),
           "lookup table on parities": float((qec.lookup_fit_predict(Ptr, ytr, Pte) != yte).mean()) if Ptr.shape[1] <= 24 else float("nan")}
    sets = {"TM on raw detectors": (Dtr, Dte, dnames), "TM on stabilizer parities": (Ptr, Pte, pnames)}
    rules = {}
    for name, (Xtr, Xte, names) in sets.items():
        cfg = dict(n_clauses=300, T=15, s=15.0) if name == "TM on raw detectors" else dict(n_clauses=40, T=5, s=4.0)
        nv = len(ytr) // 5  # last fifth of the training shots picks the best epoch; test shots are never used
        t0 = time.time(); tm = TsetlinMachine(seed=1, **cfg).fit(Xtr[:-nv], ytr[:-nv], epochs=8, val=(Xtr[-nv:], ytr[-nv:]))
        row[name] = float((tm.predict(Xte) != yte).mean()); r = tm.rules(names); rules[name] = r
        row[name + " | clauses"] = len(r); row[name + " | mean literals"] = float(np.mean([len(x[1]) for x in r])) if r else 0.0
        print(f"p={p} {name}: err {row[name]:.4f}, {len(r)} clauses, {time.time() - t0:.0f}s", flush=True)
    results[str(p)] = {"errors": row, "rules": rules, "se": float(np.sqrt(row["MWPM (PyMatching)"] / NTE))}
    print(f"p={p}: " + ", ".join(f"{k} {v:.4f}" for k, v in row.items() if "|" not in k), flush=True)
    json.dump(results, open(f"results_d{d}.json", "w"), indent=1)

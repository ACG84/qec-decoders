"""Print the learned rules for one setting, with how often each fires and how often it is right.
    python show_rules.py [p]"""
import sys, numpy as np
from stm.tm import TsetlinMachine
from stm import qec
p = float(sys.argv[1]) if len(sys.argv) > 1 else 0.002
c = qec.memory_circuit(3, 3, p); Dtr, ytr = qec.sample(c, 30000, 11); Dte, yte = qec.sample(c, 100000, 22)
M, names = qec.parity_features(c); Ptr, Pte = qec.apply_parities(Dtr, M), qec.apply_parities(Dte, M)
tm = TsetlinMachine(40, T=5, s=4.0, seed=1).fit(Ptr[:24000], ytr[:24000], epochs=8, val=(Ptr[24000:], ytr[24000:]))
print(f"d=3, p={p}: TM logical error {(tm.predict(Pte) != yte).mean():.4f}; MWPM {(qec.mwpm_predict(c, Dte) != yte).mean():.4f}")
L = np.concatenate([Pte, ~Pte], axis=1); inc = tm.ta > tm.N; seen = {}
for j in range(tm.C):
    if not inc[j].any(): continue
    key = (int(tm.pol[j]), tuple(np.flatnonzero(inc[j])))
    seen[key] = seen.get(key, 0) + 1
F = len(names); rows = []
for (pol, idx), mult in seen.items():
    fire = ~np.any(~L[:, list(idx)], axis=1); txt = " AND ".join(names[k] if k < F else "NOT " + names[k - F] for k in idx)
    rows.append((pol, -fire.mean(), f"{'FLIP   ' if pol > 0 else 'NO FLIP'} x{mult}  if {txt}   [fires {fire.mean():.3%}, logical flip when fired {yte[fire].mean() if fire.any() else float('nan'):.1%}]"))
for r in sorted(rows, key=lambda r: (-r[0], r[1])): print("  " + r[2])
print(f"  {len(seen)} distinct rules from {sum(seen.values())} non-empty clauses")

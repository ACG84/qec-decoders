"""Clifford-clause machine in front of exact matching: accuracy vs share of shots the rules answer.
    python benchmarks/hybrid_tradeoff.py   -> benchmarks/hybrid_tradeoff.png, hybrid_tradeoff.json"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qecdec as qd
from qecdec import metrics

here = os.path.dirname(os.path.abspath(__file__)); out = {}
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for d, p, shots in ((5, 0.002, 20000), (7, 0.002, 10000), (5, 0.005, 20000)):
    c = qd.surface_code_circuit(d, p=p); g = qd.DecodingGraph(c)
    Str, Otr = qd.sample(c, 30000, seed=2); S, O = qd.sample(c, shots, seed=1)
    rules = qd.CliffordClauseDecoder(g).fit(Str, Otr); mw = qd.MWPMDecoder(g)
    t0 = time.perf_counter(); pm = mw.decode_batch(S); t_m = time.perf_counter() - t0
    t0 = time.perf_counter(); pr = rules.decode_batch(S); t_r = time.perf_counter() - t0
    votes = rules.tm.votes(rules.features(S)); rows = []
    for m in range(0, int(np.abs(votes).max()) + 2):
        conf = np.abs(votes) >= m; hyb = np.where(conf, pr, pm)
        rows.append((m, float(conf.mean()), float((hyb != O).mean())))
    key = f"d{d}_p{p}"
    out[key] = {"rows": rows, "matching_ler": float((pm != O).mean()), "rules_ler": float((pr != O).mean()),
                "empty_syndromes": float((S.sum(1) == 0).mean()), "matching_us_per_shot": 1e6 * t_m / shots, "rules_us_per_shot": 1e6 * t_r / shots}
    ax.plot([r[1] for r in rows], [max(r[2], 1e-5) for r in rows], marker="o", ms=3, label=f"d={d}, p={p}")
    ax.axhline(out[key]["matching_ler"], ls=":", lw=0.8, color=ax.lines[-1].get_color())
    best = [r for r in rows if r[2] <= 1.15 * out[key]["matching_ler"] + 1e-9]
    b = max(best, key=lambda r: r[1])
    print(f"{key}: matching {out[key]['matching_ler']:.5f} ({out[key]['matching_us_per_shot']:.0f} us/shot), rules alone {out[key]['rules_ler']:.5f} ({out[key]['rules_us_per_shot']:.1f} us/shot), "
          f"empty syndromes {out[key]['empty_syndromes']:.1%}; within 15% of matching: rules answer {b[1]:.1%} at margin {b[0]} (LER {b[2]:.5f})", flush=True)
ax.set_yscale("log"); ax.set_xlabel("share of shots answered by the rule machine"); ax.set_ylabel("logical error rate per shot")
ax.set_title("Clifford-clause machine in front of exact matching\n(dotted: matching alone; right end: rules alone)"); ax.legend(); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(here, "hybrid_tradeoff.png"), dpi=150)
json.dump(out, open(os.path.join(here, "hybrid_tradeoff.json"), "w"), indent=1)

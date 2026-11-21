"""Threshold scan: logical error rate vs physical error rate for several distances.
    python benchmarks/threshold_plot.py [shots]   -> benchmarks/threshold.png, threshold.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qecdec as qd
from qecdec import metrics

shots = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
ps = [0.002, 0.004, 0.006, 0.008, 0.010, 0.012]; ds = [3, 5, 7]
here = os.path.dirname(os.path.abspath(__file__))
res = metrics.threshold_scan(lambda d, p: qd.surface_code_circuit(d, p=p), lambda c: qd.UnionFindDecoder(qd.DecodingGraph(c)), ds, ps, shots=shots, seed=4)
fig, ax = plt.subplots(figsize=(6, 4.2))
for d in ds:
    y = [res[(d, p)]["logical_error_rate"] for p in ps]
    lo = [res[(d, p)]["logical_error_rate"] - res[(d, p)]["ci95"][0] for p in ps]; hi = [res[(d, p)]["ci95"][1] - res[(d, p)]["logical_error_rate"] for p in ps]
    ax.errorbar(ps, y, yerr=[lo, hi], marker="o", capsize=3, label=f"distance {d}")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("physical error rate p"); ax.set_ylabel("logical error rate per shot (d rounds)")
ax.set_title(f"Union-Find decoder, rotated surface code, {shots} shots per point"); ax.legend(); ax.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(os.path.join(here, "threshold.png"), dpi=150)
json.dump({f"d{d}_p{p}": res[(d, p)] for d in ds for p in ps}, open(os.path.join(here, "threshold.json"), "w"), indent=1)
for d in ds: print(d, [round(res[(d, p)]["logical_error_rate"], 4) for p in ps])

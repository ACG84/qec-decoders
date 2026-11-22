"""Rule machine in front of matching, with and without the pair-cancellation layer.

For each setting: train on 30,000 shots, pick the vote margin on 10,000 validation shots (the
smallest margin whose validation error is within 15% of matching's, or one logical error if that
is larger), then report on separate test shots.
    python benchmarks/hybrid_tradeoff.py   -> hybrid_tradeoff.png, hybrid_tradeoff.json, hybrid_tradeoff.txt
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymatching
import qecdec as qd

here = os.path.dirname(os.path.abspath(__file__)); out = {}; lines = []
SETTINGS = ((5, 0.002, 20000), (5, 0.005, 20000), (7, 0.002, 20000), (7, 0.004, 10000), (9, 0.002, 10000))


def curve(votes, pred, pm, O):
    rows = []
    for m in range(0, int(np.abs(votes).max()) + 2):
        conf = np.abs(votes) >= m
        rows.append((m, float(conf.mean()), float(np.mean(np.where(conf, pred, pm) != O))))
    return rows


def pick_margin(votes, pred, pm, O):
    base = np.mean(pm != O); allow = max(1.15 * base, base + 1.0 / len(O))
    for m, share, ler in curve(votes, pred, pm, O):
        if ler <= allow + 1e-12:
            return m
    return int(np.abs(votes).max()) + 1


fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
panel = {(5, 0.005): 0, (7, 0.002): 1, (7, 0.004): 2}
for d, p, shots in SETTINGS:
    c = qd.surface_code_circuit(d, p=p); g = qd.DecodingGraph(c)
    Str, Otr = qd.sample(c, 30000, seed=2); Sv, Ov = qd.sample(c, 10000, seed=3); S, O = qd.sample(c, shots, seed=1)
    mw = qd.FastMWPMDecoder(g); mw.decode_batch(S[:10])
    t0 = time.perf_counter(); pm = mw.decode_batch(S); t_m = time.perf_counter() - t0
    pmv = mw.decode_batch(Sv)
    ref = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True)).decode_batch(S)[:, 0]
    pre = qd.PairCancel(g)
    key = f"d{d}_p{p}"
    out[key] = {"matching_ler": float(np.mean(pm != O)), "pymatching_ler": float(np.mean(ref != O)), "matching_us": 1e6 * t_m / shots,
                "empty_syndromes": float((S.sum(1) == 0).mean())}
    lines.append(f"d={d}, p={p}, {shots} test shots: our matching {np.mean(pm != O):.5f}, PyMatching {np.mean(ref != O):.5f}, "
                 f"matching {1e6 * t_m / shots:.0f} us/shot (numba, {qd.fast.threads()} threads)")
    for layered in (False, True):
        tr, dtr = pre.apply(Str) if layered else (Str, np.zeros(len(Str), np.uint8))
        va, dva = pre.apply(Sv) if layered else (Sv, np.zeros(len(Sv), np.uint8))
        t0 = time.perf_counter()
        te, dte = pre.apply(S) if layered else (S, np.zeros(len(S), np.uint8))
        t_pre = time.perf_counter() - t0
        rules = qd.CliffordClauseDecoder(g).fit(tr, Otr ^ dtr)
        vv = rules.tm.votes(rules.features(va)); m = pick_margin(vv, (vv > 0).astype(np.uint8) ^ dva, pmv, Ov)
        t0 = time.perf_counter(); vt = rules.tm.votes(rules.features(te)); t_r = time.perf_counter() - t0
        pred = (vt > 0).astype(np.uint8) ^ dte
        conf = np.abs(vt) >= m; hyb = np.where(conf, pred, pm)
        name = "pair cancel + rules" if layered else "rules"
        out[key][name] = {"rules_alone_ler": float(np.mean(pred != O)), "margin": int(m), "share": float(conf.mean()),
                          "hybrid_ler": float(np.mean(hyb != O)), "rules_us": 1e6 * (t_r + t_pre) / shots,
                          "residual_empty": float((te.sum(1) == 0).mean()), "curve": curve(vt, pred, pm, O)}
        lines.append(f"   {name:20s}: alone {np.mean(pred != O):.5f}; margin {m} (chosen on validation) answers {conf.mean():.1%} of shots, "
                     f"hybrid LER {np.mean(hyb != O):.5f}; rule path {1e6 * (t_r + t_pre) / shots:.0f} us/shot")
        if (d, p) in panel:
            ax = axes[panel[(d, p)]]; rows = out[key][name]["curve"]
            ax.plot([r[1] for r in rows], [max(r[2], 2e-5) for r in rows], marker="o", ms=3, label=name,
                    color="#1f6fb2" if layered else "#999999")
            ax.plot([conf.mean()], [max(np.mean(hyb != O), 2e-5)], marker="*", ms=14, color="#1f6fb2" if layered else "#999999")
    if (d, p) in panel:
        ax = axes[panel[(d, p)]]
        ax.axhline(max(out[key]["matching_ler"], 2e-5), ls=":", lw=1, color="k", label="matching alone")
        ax.set_yscale("log"); ax.set_xlabel("share of shots answered without matching"); ax.set_title(f"d={d}, p={p}")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    print("\n".join(lines[-3:]), flush=True)
axes[0].set_ylabel("logical error rate per shot")
fig.suptitle("Rule machine in front of matching (star: margin chosen on validation shots)")
fig.tight_layout(); fig.savefig(os.path.join(here, "hybrid_tradeoff.png"), dpi=150)
json.dump(out, open(os.path.join(here, "hybrid_tradeoff.json"), "w"), indent=1)
open(os.path.join(here, "hybrid_tradeoff.txt"), "w").write("\n".join(lines) + "\n")

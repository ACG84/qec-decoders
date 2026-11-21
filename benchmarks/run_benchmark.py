"""Benchmark every decoder on the same syndromes.
    python benchmarks/run_benchmark.py [shots]
Rotated surface code memory, circuit-level depolarizing noise, rounds = distance."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qecdec as qd
from qecdec import metrics
import pymatching

shots = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
out = {}
for d, p in ((3, 0.002), (3, 0.005), (5, 0.002), (5, 0.005)):
    c = qd.surface_code_circuit(d, p=p); g = qd.DecodingGraph(c)
    S, O = qd.sample(c, shots, seed=1); Str, Otr = qd.sample(c, 30000, seed=2)
    decs = [qd.MWPMDecoder(g), qd.UnionFindDecoder(g), qd.BPDecoder(g), qd.LocalMatchingDecoder(g), qd.GreedyDecoder(g),
            qd.LookupDecoder(g, max_weight=2 if d == 3 else 1, fallback=qd.UnionFindDecoder(g)),
            qd.TsetlinDecoder(g).fit(Str, Otr), qd.CliffordClauseDecoder(g, literals="learned").fit(Str, Otr),
            qd.CliffordClauseDecoder(g, literals="cuts").fit(Str, Otr), qd.CutVoteDecoder(g, fallback=qd.MWPMDecoder(g))]
    decs.append(qd.HybridDecoder(decs[8], fallback=qd.MWPMDecoder(g)))
    rows = [metrics.evaluate(dec, S, O) for dec in decs]
    ref = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True)).decode_batch(S)[:, 0]
    print(f"\ndistance {d}, p = {p}, {shots} shots, {g.n} detectors, {len(g.eu)} edges; "
          f"no decoding {metrics.no_decoding_rate(O):.5f}; PyMatching (reference library) {(ref != O).mean():.5f}")
    print(metrics.format_table(rows))
    print(f"   MWPM fallbacks to greedy: {decs[0].fallbacks}; lookup misses: {decs[5].misses} of {shots}")
    print(f"   BP settled without fallback on {decs[2].converged / decs[2].total:.1%} of shots; "
          f"local matching used {sum(decs[3].rounds_used) / len(decs[3].rounds_used):.1f} rounds on average, {max(decs[3].rounds_used)} at most")
    print(f"   rules: Tsetlin {len(set((a, tuple(l)) for a, l in decs[6].rules()))}, Clifford-clause learned {len(set((a, tuple(l)) for a, l in decs[7].rules()))}, "
          f"Clifford-clause cuts {len(set((a, tuple(l)) for a, l in decs[8].rules()))} over {decs[8].n_literals()} literals; lookup table entries {len(decs[5].table)}")
    print(f"   cut vote settled {decs[9].fast / decs[9].total:.1%} of shots without matching; hybrid (margin {decs[10].margin}) settled {decs[10].fast / decs[10].total:.1%}")
    sweep = []
    for m in (4, 8, 11, 13, 15):
        h = qd.HybridDecoder(decs[8], fallback=qd.MWPMDecoder(g), margin=m); r = metrics.evaluate(h, S, O)
        sweep.append((m, round(h.fast / h.total, 3), round(r["logical_error_rate"], 5)))
    print("   hybrid margin sweep (margin, share answered by rules, LER):", sweep)
    out[f"d{d}_p{p}_hybrid_sweep"] = sweep
    out[f"d{d}_p{p}"] = rows
import time
c = qd.surface_code_circuit(5, p=0.005); g = qd.DecodingGraph(c); S, O = qd.sample(c, shots, seed=1); dec = qd.UnionFindDecoder(g)
t0 = time.perf_counter(); a = dec.decode_batch(S); t1 = time.perf_counter(); b2 = qd.parallel_decode(dec, S, workers=2); t2 = time.perf_counter()
print(f"\nShot-level parallelism, Union-Find, d=5, p=0.005: serial {t1 - t0:.2f}s, 2 processes {t2 - t1:.2f}s, speedup {(t1 - t0) / (t2 - t1):.2f}x, identical output: {(a == b2).all()}")
out["parallel"] = {"serial_s": t1 - t0, "two_process_s": t2 - t1}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "results.json"), "w"), indent=1)

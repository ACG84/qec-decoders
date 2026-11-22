"""Pair-cancellation layer in front of the rule machine and the cut vote: share of shots answered
without matching at matching-level accuracy.   PYTHONPATH=. python research/predecode_tm.py"""
import sys, time, numpy as np, qecdec as qd
from qecdec.fast import FastMWPMDecoder
from qecdec.clifford_clause import cut_literals

def predecode(g, S, rounds=1):
    dd = np.flatnonzero((g.eu < g.n) & (g.ev < g.n))
    u, v, eo = g.eu[dd], g.ev[dd], g.eobs[dd].astype(np.int32)
    A = np.zeros((g.n, g.n), np.int32); A[u, v] = 1; A[v, u] = 1
    R = S.copy(); obs = np.zeros(len(S), np.int32)
    for _ in range(rounds):
        c = R.astype(np.int32) @ A
        iso = (R[:, u] == 1) & (R[:, v] == 1) & (c[:, u] == 1) & (c[:, v] == 1)
        rows, cols = np.nonzero(iso)
        R[rows, u[cols]] = 0; R[rows, v[cols]] = 0
        obs ^= (iso.astype(np.int32) @ eo) % 2
    return R, obs.astype(np.uint8)

def best_share(votes, pred, pm, O, tol=1.15):
    base = np.mean(pm != O); best = (0.0, base, None)
    for m in range(0, int(np.abs(votes).max()) + 2):
        conf = np.abs(votes) >= m; ler = np.mean(np.where(conf, pred, pm) != O)
        if ler <= tol * base + 1e-9 and conf.mean() > best[0]:
            best = (conf.mean(), ler, m)
    return best

for d, p, shots in ((5, 0.002, 20000), (5, 0.005, 20000), (7, 0.002, 20000), (7, 0.004, 10000)):
    c = qd.surface_code_circuit(d, p=p); g = qd.DecodingGraph(c)
    Str, Otr = qd.sample(c, 30000, seed=2); S, O = qd.sample(c, shots, seed=1)
    pm = FastMWPMDecoder(g).decode_batch(S); base = np.mean(pm != O)
    M, names = cut_literals(g); C = M[[i for i, n in enumerate(names) if n.startswith("cut") and "^" not in n]]
    out = [f"d={d} p={p} matching {base:.5f}"]
    for rounds in (0, 1, 2):
        R, ob = (S, np.zeros(len(S), np.uint8)) if rounds == 0 else predecode(g, S, rounds)
        Rtr, obtr = (Str, np.zeros(len(Str), np.uint8)) if rounds == 0 else predecode(g, Str, rounds)
        V = (R.astype(np.int32) @ C.T) % 2; un = V.min(1) == V.max(1)
        cut = np.where(un, V[:, 0] ^ ob, pm)
        t = time.perf_counter(); rules = qd.CliffordClauseDecoder(g).fit(Rtr, Otr ^ obtr); tt = time.perf_counter() - t
        votes = rules.tm.votes(rules.features(R)); pred = (votes > 0).astype(np.uint8) ^ ob
        sh, ler, m = best_share(votes, pred, pm, O)
        out.append(f"  layers={rounds}: cut vote settles {un.mean():.1%} (LER {np.mean(cut != O):.5f}) | rule machine alone {np.mean(pred != O):.5f}; "
                   f"within 15% of matching it answers {sh:.1%} (margin {m}, LER {ler:.5f}) | train {tt:.0f}s")
    print("\n".join(out), flush=True)

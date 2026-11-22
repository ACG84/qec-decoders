import time, sys, numpy as np, qecdec as qd
from qecdec.fast import FastMWPMDecoder
from qecdec.clifford_clause import cut_literals

def predecode(g, S):
    dd = np.flatnonzero((g.eu < g.n) & (g.ev < g.n))
    u, v, eo = g.eu[dd], g.ev[dd], g.eobs[dd]
    A = np.zeros((g.n, g.n), np.int32); A[u, v] = 1; A[v, u] = 1
    S32 = S.astype(np.int32)
    c = S32 @ A
    iso = (S[:, u] == 1) & (S[:, v] == 1) & (c[:, u] == 1) & (c[:, v] == 1)     # shots x edges
    R = S.copy()
    rows, cols = np.nonzero(iso)
    R[rows, u[cols]] = 0; R[rows, v[cols]] = 0
    obs = (iso.astype(np.int32) @ eo.astype(np.int32)) % 2
    return R, obs.astype(np.uint8), iso.sum(1)

for d, p, shots in ((5,0.002,20000),(5,0.005,20000),(7,0.002,20000),(7,0.004,10000),(9,0.002,8000)):
    c = qd.surface_code_circuit(d, p=p); g = qd.DecodingGraph(c); S, O = qd.sample(c, shots, seed=1)
    mw = FastMWPMDecoder(g, max_defects=20); pm = mw.decode_batch(S)
    M, names = cut_literals(g); C = M[[i for i, n in enumerate(names) if n.startswith("cut") and "^" not in n]]
    V = (S.astype(np.int32) @ C.T) % 2; un = V.min(1) == V.max(1)
    plain = np.where(un, V[:, 0], pm)
    R, ob, npairs = predecode(g, S)
    V2 = (R.astype(np.int32) @ C.T) % 2; un2 = V2.min(1) == V2.max(1)
    pre = np.where(un2, V2[:, 0] ^ ob, pm)
    empty = (S.sum(1) == 0).mean(); rempty = (R.sum(1) == 0).mean()
    print(f"d={d} p={p} K={len(C)} matching {np.mean(pm!=O):.5f} | plain cut vote: {un.mean():.1%} settled, LER {np.mean(plain!=O):.5f}, err among settled {np.mean((V[:,0]!=O)[un]):.5f}"
          f" | predecode+cut vote: {un2.mean():.1%} settled, LER {np.mean(pre!=O):.5f}, err among settled {np.mean(((V2[:,0]^ob)!=O)[un2]):.5f} | empty {empty:.1%} -> residual empty {rempty:.1%}; mean defects {S.sum(1).mean():.1f} -> {R.sum(1).mean():.1f}", flush=True)

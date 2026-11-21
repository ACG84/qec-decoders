import sys, time, itertools, numpy as np
from multiprocessing import Pool
from stm.tm import TsetlinMachine
from stm import qec
c = qec.memory_circuit(3, 3, 0.005); Dtr, ytr = qec.sample(c, 30000, 11); Dva, yva = qec.sample(c, 50000, 33)
M, _ = qec.parity_features(c); Ptr, Pva = qec.apply_parities(Dtr, M), qec.apply_parities(Dva, M)
def job(a):
    feat, C, T, s = a; Xtr, Xva = (Ptr, Pva) if feat == "par" else (Dtr, Dva)
    t0 = time.time(); tm = TsetlinMachine(C, T=T, s=s, seed=1).fit(Xtr, ytr, epochs=6)
    return feat, C, T, s, float((tm.predict(Xva) != yva).mean()), round(time.time() - t0)
if __name__ == "__main__":
    grid = [("par", C, T, s) for C in (40, 100) for T in (5, 15, 40) for s in (2.0, 4.0, 8.0)] + [("det", C, T, s) for C in (100, 300) for T in (15, 40) for s in (5.0, 15.0)]
    with Pool(2) as p:
        res = sorted(p.imap_unordered(job, grid), key=lambda r: (r[0], r[4]))
    for r in res: print(r)
    print("refs: lookup", (qec.lookup_fit_predict(Ptr, ytr, Pva) != yva).mean(), "mwpm", (qec.mwpm_predict(c, Dva) != yva).mean())

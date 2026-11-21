"""Clifford clauses on syndrome data.

On classical bits a Clifford circuit of CNOTs is an invertible GF(2)-linear map, and measuring
its outputs gives parities of the inputs. A clause that ANDs such outcomes is the indicator of
an affine GF(2) subspace: the classical shadow of a stabilizer projector. So a "Clifford clause"
= AND over parity literals, where each parity literal is the XOR of a learned subset of bits.

`grow_parities` is the proposer. It uses no knowledge of the code or noise model: starting from
single bits, each round it adds the XOR of pairs of current literals that fire together far more
often than chance (one CNOT deeper per round). Paired firings are what a single fault produces,
so XORing them cancels that fault mechanism.
"""
import numpy as np


def grow_parities(X, rounds=3, per_round=None, min_corr=0.03, max_pool=400):
    """X: (shots, bits) bool. Returns supports (list of frozensets of bit indices)."""
    X = np.asarray(X, dtype=bool)
    supports = [frozenset([i]) for i in range(X.shape[1])]
    V = X.copy()
    per_round = per_round or X.shape[1]
    seen = set(supports)
    for _ in range(rounds):
        C = np.corrcoef(V.T.astype(np.float32))
        np.fill_diagonal(C, 0)
        iu = np.triu_indices(len(supports), 1)
        order = np.argsort(-C[iu])
        new = []
        for k in order:
            a, b = iu[0][k], iu[1][k]
            if C[a, b] < min_corr or len(new) >= per_round:
                break
            s = supports[a] ^ supports[b]
            if s and s not in seen:
                seen.add(s); new.append(s)
        if not new or len(supports) + len(new) > max_pool:
            new = new[: max(0, max_pool - len(supports))]
        if not new:
            break
        supports += new
        V = np.hstack([V, np.stack([X[:, sorted(s)].sum(1) % 2 == 1 for s in new], axis=1)])
    return supports


def apply_supports(X, supports):
    X = np.asarray(X, dtype=bool)
    return np.stack([X[:, sorted(s)].sum(1) % 2 == 1 for s in supports], axis=1)


def support_names(supports, bit_names):
    return ["^".join(bit_names[i] for i in sorted(s)) for s in supports]

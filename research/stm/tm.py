"""Minimal two-class Tsetlin machine (Granmo 2018), written from scratch in NumPy.

A clause is an AND over included literals; literals are the input bits and their negations.
Half the clauses vote for class 1, half against. Prediction: sum of votes >= 0.
"""
import numpy as np


class TsetlinMachine:
    def __init__(self, n_clauses=40, T=10, s=3.0, n_states=100, seed=0):
        assert n_clauses % 2 == 0
        self.C, self.T, self.s, self.N = n_clauses, T, s, n_states
        self.rng = np.random.default_rng(seed)
        self.pol = np.where(np.arange(n_clauses) % 2 == 0, 1, -1)
        self.ta = None  # automaton states, shape (clauses, 2*features); include iff state > N

    def _lits(self, X):
        X = np.asarray(X, dtype=bool)
        return np.concatenate([X, ~X], axis=1)

    def _clause_out(self, lit, training):
        inc = self.ta > self.N
        out = ~np.any(inc & ~lit[None, :], axis=1)
        if not training:
            out &= inc.any(axis=1)  # an empty clause does not vote at inference
        return out

    def fit(self, X, y, epochs=10, val=None):
        """val=(Xv, yv): keep the automaton states from the epoch with the lowest validation
        error (training is stochastic, so the last epoch is not always the best)."""
        L = self._lits(X)
        y = np.asarray(y).astype(int).ravel()
        if self.ta is None:
            self.ta = self.rng.integers(self.N, self.N + 2, size=(self.C, L.shape[1]))
        best, best_ta = None, None
        for _ in range(epochs):
            for i in self.rng.permutation(len(y)):
                self._update(L[i], y[i])
            if val is not None:
                e = float((self.predict(val[0]) != np.asarray(val[1]).ravel()).mean())
                if best is None or e < best:
                    best, best_ta = e, self.ta.copy()
        if best_ta is not None:
            self.ta = best_ta
        return self

    def _update(self, lit, y):
        out = self._clause_out(lit, True)
        v = int(np.clip((self.pol * out).sum(), -self.T, self.T))
        p = (self.T - v) / (2 * self.T) if y == 1 else (self.T + v) / (2 * self.T)
        chosen = self.rng.random(self.C) < p
        type1 = chosen & ((self.pol == 1) == (y == 1))  # reinforce the pattern
        type2 = chosen & ~((self.pol == 1) == (y == 1))  # block a false vote
        if type1.any():
            r = self.rng.random((int(type1.sum()), len(lit)))
            o = out[type1][:, None]
            up = o & lit[None, :] & (r < (self.s - 1) / self.s)
            down = (~(o & lit[None, :])) & (r < 1 / self.s)
            self.ta[type1] += up.astype(int) - down.astype(int)
        if type2.any():
            excl = self.ta[type2] <= self.N
            self.ta[type2] += (out[type2][:, None] & ~lit[None, :] & excl).astype(int)
        np.clip(self.ta, 1, 2 * self.N, out=self.ta)

    def votes(self, X):
        L = self._lits(X)
        inc = self.ta > self.N
        viol = (~L).astype(np.int32) @ inc.T.astype(np.int32)  # (samples, clauses)
        out = (viol == 0) & inc.any(axis=1)[None, :]
        return out @ self.pol

    def predict(self, X):
        return (self.votes(X) > 0).astype(int)

    def rules(self, names):
        """Readable clauses: list of (polarity, [literal strings]), empty clauses dropped."""
        F = len(names)
        out = []
        for j in range(self.C):
            inc = np.flatnonzero(self.ta[j] > self.N)
            if len(inc):
                out.append((int(self.pol[j]), [names[k] if k < F else "NOT " + names[k - F] for k in inc]))
        return out

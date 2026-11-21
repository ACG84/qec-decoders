"""Tsetlin-machine decoder with stabilizer-parity literals (the novel decoder in this repo).

A Tsetlin machine learns a small set of AND-rules over Boolean literals. Here the literals are
GF(2) parities taken from the code structure: for each stabilizer, the XOR of its detection
events over all rounds (a measurement error fires the same stabilizer twice and cancels; a data
error survives). The trained decoder is a list of human-readable rules, and inference is a few
bitwise operations per rule, evaluated for a whole batch of shots at once. It learns from
samples and needs no noise model. It is competitive only on small codes: see the README.
"""
import numpy as np
from .tsetlin_core import TsetlinMachine


def stabilizer_parities(circuit):
    co = circuit.get_detector_coordinates()
    groups = {}
    for i in range(circuit.num_detectors):
        groups.setdefault((int(co[i][0]), int(co[i][1])), []).append(i)
    M = np.zeros((len(groups), circuit.num_detectors), dtype=np.int32)
    names = []
    for r, ((x, y), idx) in enumerate(sorted(groups.items())):
        M[r, idx] = 1; names.append(f"S[x{x},y{y}]")
    return M, names


class TsetlinDecoder:
    name = "Tsetlin machine (parity literals)"

    def __init__(self, graph, n_clauses=40, T=5, s=4.0, seed=1):
        self.g = graph
        self.M, self.names = stabilizer_parities(graph.circuit)
        self.tm = TsetlinMachine(n_clauses=n_clauses, T=T, s=s, seed=seed)

    def features(self, syndromes):
        return (np.atleast_2d(syndromes).astype(np.int32) @ self.M.T) % 2 == 1

    def fit(self, syndromes, observables, epochs=8, val_fraction=0.2):
        X = self.features(syndromes); nv = int(len(X) * val_fraction)
        self.tm.fit(X[:-nv], observables[:-nv], epochs=epochs, val=(X[-nv:], observables[-nv:]))
        return self

    def decode(self, syndrome):
        return int(self.tm.predict(self.features(syndrome))[0])

    def decode_batch(self, syndromes):
        return self.tm.predict(self.features(syndromes)).astype(np.uint8)

    def rules(self):
        return self.tm.rules(self.names)

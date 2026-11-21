"""Stim memory experiments, stabilizer-parity features and decoder baselines."""
import numpy as np
import stim
import pymatching


def memory_circuit(distance=3, rounds=3, p=0.005, basis="z"):
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}", distance=distance, rounds=rounds,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
        after_reset_flip_probability=p, before_round_data_depolarization=p)


def sample(circuit, shots, seed):
    d, o = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return d.astype(bool), o[:, 0].astype(int)


def detector_names(circuit):
    co = circuit.get_detector_coordinates()
    return [f"D[x{int(co[i][0])},y{int(co[i][1])},t{int(co[i][2])}]" for i in range(circuit.num_detectors)]


def parity_features(circuit):
    """GF(2) parity literals built from the code structure: for every stabilizer (same x, y),
    the XOR of its detection events over all rounds. A measurement error fires the same
    stabilizer twice in time and cancels; a data error leaves the parity set.
    Returns (matrix M with rows = features over detectors, names)."""
    co = circuit.get_detector_coordinates()
    groups = {}
    for i in range(circuit.num_detectors):
        groups.setdefault((int(co[i][0]), int(co[i][1])), []).append(i)
    M, names = [], []
    for (x, y), idx in sorted(groups.items()):
        row = np.zeros(circuit.num_detectors, dtype=bool)
        row[idx] = True
        M.append(row)
        names.append(f"S[x{x},y{y}]")
    return np.array(M), names


def apply_parities(D, M):
    return (D.astype(np.int32) @ M.T.astype(np.int32)) % 2 == 1


def mwpm_predict(circuit, D):
    m = pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))
    return m.decode_batch(D)[:, 0].astype(int)


def lookup_fit_predict(Xtr, ytr, Xte):
    """Majority-vote table on the full feature pattern: the best any rule set over these
    features could do given this much data (unseen patterns predict 0)."""
    w = 1 << np.arange(Xtr.shape[1], dtype=np.int64)
    ktr, kte = Xtr.astype(np.int64) @ w, Xte.astype(np.int64) @ w
    keys, inv = np.unique(ktr, return_inverse=True)
    ones = np.bincount(inv, weights=ytr, minlength=len(keys)); tot = np.bincount(inv, minlength=len(keys))
    table = dict(zip(keys.tolist(), (ones * 2 > tot).astype(int).tolist()))
    return np.array([table.get(k, 0) for k in kte.tolist()])

"""Decoder metrics: logical error rate with a confidence interval, speed, and threshold scans."""
import math
import time
import numpy as np
from .graph import sample


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def evaluate(decoder, syndromes, observables):
    """Logical error rate (per shot), 95% Wilson interval and microseconds per shot."""
    t0 = time.perf_counter()
    pred = decoder.decode_batch(syndromes)
    dt = time.perf_counter() - t0
    k = int((pred != observables).sum()); n = len(observables)
    lo, hi = wilson_interval(k, n)
    return {"decoder": decoder.name, "shots": n, "errors": k, "logical_error_rate": k / n,
            "ci95": (lo, hi), "us_per_shot": 1e6 * dt / n}


def no_decoding_rate(observables):
    return float(np.mean(observables))


def threshold_scan(make_circuit, make_decoder, distances, ps, shots=5000, seed=0):
    """Logical error rate for each (distance, p). Curves for different distances cross near the
    threshold: below it larger codes do better, above it worse."""
    out = {}
    for d in distances:
        for p in ps:
            c = make_circuit(d, p)
            S, O = sample(c, shots, seed=seed)
            out[(d, p)] = evaluate(make_decoder(c), S, O)
    return out


def format_table(rows):
    lines = [f"{'decoder':36s} {'LER':>9s} {'95% CI':>19s} {'us/shot':>9s}"]
    for r in rows:
        lines.append(f"{r['decoder']:36s} {r['logical_error_rate']:9.5f} "
                     f"[{r['ci95'][0]:.5f}, {r['ci95'][1]:.5f}] {r['us_per_shot']:9.1f}")
    return "\n".join(lines)

"""Correctness and metric tests. Run with: pytest -q   (or: python tests/test_decoders.py)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qecdec as qd
from qecdec import metrics


def graph(d=3, p=0.005):
    return qd.DecodingGraph(qd.surface_code_circuit(d, p=p))


def graph_decoders(g):
    return [qd.MWPMDecoder(g), qd.GreedyDecoder(g), qd.UnionFindDecoder(g), qd.LookupDecoder(g, max_weight=1),
            qd.BPDecoder(g), qd.LocalMatchingDecoder(g)]


def test_empty_syndrome_means_no_flip():
    g = graph()
    for dec in graph_decoders(g):
        assert dec.decode(np.zeros(g.n, np.uint8)) == 0, dec.name


def test_every_single_fault_is_corrected():
    """Each error mechanism alone must be decoded to its own logical effect (distance >= 3)."""
    for g in (graph(3), graph(5)):
        for dec in graph_decoders(g):
            for e in range(len(g.eu)):
                s = np.zeros(g.n, np.uint8)
                for x in (int(g.eu[e]), int(g.ev[e])):
                    if not g.is_boundary(x):
                        s[x] ^= 1
                assert dec.decode(s) == int(g.eobs[e]), (dec.name, e)


def test_mwpm_matches_reference_matching():
    """Our exact matching should agree with PyMatching's logical error rate within noise."""
    import pymatching
    c = qd.surface_code_circuit(3, p=0.005); g = qd.DecodingGraph(c); S, O = qd.sample(c, 20000, seed=5)
    ours = qd.MWPMDecoder(g).decode_batch(S)
    ref = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True)).decode_batch(S)[:, 0]
    assert (ours == ref).mean() > 0.995
    assert abs((ours != O).mean() - (ref != O).mean()) < 0.003


def test_decoders_beat_no_decoding_and_order_sensibly():
    c = qd.surface_code_circuit(3, p=0.003); g = qd.DecodingGraph(c); S, O = qd.sample(c, 20000, seed=7)
    base = metrics.no_decoding_rate(O)
    ler = {d.name: metrics.evaluate(d, S, O)["logical_error_rate"] for d in (qd.MWPMDecoder(g), qd.GreedyDecoder(g), qd.UnionFindDecoder(g))}
    assert all(v < base / 3 for v in ler.values()), (base, ler)
    assert ler["MWPM (exact, subset DP)"] <= min(ler.values()) + 0.002


def test_larger_distance_helps_below_threshold():
    r = metrics.threshold_scan(lambda d, p: qd.surface_code_circuit(d, p=p), lambda c: qd.UnionFindDecoder(qd.DecodingGraph(c)), [3, 5], [0.001], shots=20000)
    assert r[(5, 0.001)]["logical_error_rate"] < r[(3, 0.001)]["logical_error_rate"]


def test_repetition_code():
    c = qd.repetition_code_circuit(5, p=0.02); g = qd.DecodingGraph(c); S, O = qd.sample(c, 10000, seed=3)
    for dec in (qd.MWPMDecoder(g), qd.UnionFindDecoder(g)):
        assert metrics.evaluate(dec, S, O)["logical_error_rate"] < 0.02, dec.name


def test_tsetlin_decoder_learns():
    c = qd.surface_code_circuit(3, p=0.002); g = qd.DecodingGraph(c)
    Str, Otr = qd.sample(c, 20000, seed=11); S, O = qd.sample(c, 20000, seed=22)
    dec = qd.TsetlinDecoder(g).fit(Str, Otr)
    assert metrics.evaluate(dec, S, O)["logical_error_rate"] < metrics.no_decoding_rate(O) / 3
    assert len(dec.rules()) > 0


def test_clifford_clause_decoder_learns_without_code_layout():
    c = qd.surface_code_circuit(3, p=0.002); g = qd.DecodingGraph(c)
    Str, Otr = qd.sample(c, 20000, seed=11); S, O = qd.sample(c, 20000, seed=22)
    dec = qd.CliffordClauseDecoder(g, literals="learned").fit(Str, Otr)
    assert metrics.evaluate(dec, S, O)["logical_error_rate"] < metrics.no_decoding_rate(O) / 2
    assert max(len(s) for s in dec.supports) >= 3   # it built parities deeper than one CNOT


def test_cut_literals_vote_on_the_logical_flip():
    """Every fault on the observable boundary flips every cut; no other single fault flips more than one."""
    from qecdec.clifford_clause import cut_literals
    g = graph(5); M, names = cut_literals(g); cuts = [i for i, n in enumerate(names) if n.startswith("cut") and "^" not in n]
    for e in range(len(g.eu)):
        s = np.zeros(g.n, np.int32)
        for x in (int(g.eu[e]), int(g.ev[e])):
            if not g.is_boundary(x):
                s[x] ^= 1
        votes = (M[cuts] @ s) % 2
        if g.eobs[e]:
            assert votes.all(), e
        else:
            assert votes.sum() <= 1, e


def test_cut_literal_decoder_beats_plain_parities_at_distance_5():
    c = qd.surface_code_circuit(5, p=0.002); g = qd.DecodingGraph(c)
    Str, Otr = qd.sample(c, 30000, seed=2); S, O = qd.sample(c, 20000, seed=1)
    cut = metrics.evaluate(qd.CliffordClauseDecoder(g).fit(Str, Otr), S, O)["logical_error_rate"]
    plain = metrics.evaluate(qd.TsetlinDecoder(g).fit(Str, Otr), S, O)["logical_error_rate"]
    assert cut < plain / 2 and cut < metrics.no_decoding_rate(O) / 5, (cut, plain)


def test_cut_vote_first_stage_keeps_matching_accuracy():
    c = qd.surface_code_circuit(5, p=0.002); g = qd.DecodingGraph(c); S, O = qd.sample(c, 20000, seed=1)
    two = qd.CutVoteDecoder(g, fallback=qd.MWPMDecoder(g)); a = metrics.evaluate(two, S, O)["logical_error_rate"]
    b = metrics.evaluate(qd.MWPMDecoder(g), S, O)["logical_error_rate"]
    assert a < b + 0.001 and two.fast / two.total > 0.4, (a, b, two.fast / two.total)


def test_hybrid_margin_trades_speed_for_accuracy():
    c = qd.surface_code_circuit(5, p=0.002); g = qd.DecodingGraph(c)
    Str, Otr = qd.sample(c, 30000, seed=2); S, O = qd.sample(c, 20000, seed=1)
    rules = qd.CliffordClauseDecoder(g).fit(Str, Otr)
    lo = qd.HybridDecoder(rules, margin=4); hi = qd.HybridDecoder(rules)
    e_lo = metrics.evaluate(lo, S, O)["logical_error_rate"]; e_hi = metrics.evaluate(hi, S, O)["logical_error_rate"]
    assert hi.fast < lo.fast and e_hi < e_lo            # stricter margin: fewer shots by rules, fewer errors
    assert e_hi < metrics.evaluate(rules, S, O)["logical_error_rate"] / 2 and hi.fast / hi.total > 0.5


def test_bp_and_local_matching_beat_no_decoding():
    c = qd.surface_code_circuit(3, p=0.003); g = qd.DecodingGraph(c); S, O = qd.sample(c, 10000, seed=9)
    for dec in (qd.BPDecoder(g), qd.LocalMatchingDecoder(g)):
        assert metrics.evaluate(dec, S, O)["logical_error_rate"] < metrics.no_decoding_rate(O) / 3, dec.name


def test_parallel_decode_matches_serial():
    c = qd.surface_code_circuit(3, p=0.005); g = qd.DecodingGraph(c); S, _ = qd.sample(c, 4000, seed=2)
    dec = qd.UnionFindDecoder(g)
    assert (qd.parallel_decode(dec, S, workers=2) == dec.decode_batch(S)).all()


def test_wilson_interval_contains_estimate():
    lo, hi = metrics.wilson_interval(30, 1000)
    assert lo < 0.03 < hi


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time(); fn(); print(f"PASS {name} ({time.time() - t:.1f}s)")


def test_numba_decoders_match_the_reference_shot_for_shot():
    for d, p in ((3, 0.005), (5, 0.005)):
        c = qd.surface_code_circuit(d, p=p); g = qd.DecodingGraph(c); S, O = qd.sample(c, 1500, seed=11)
        assert (qd.FastMWPMDecoder(g, large="greedy").decode_batch(S) == qd.MWPMDecoder(g).decode_batch(S)).all()
        assert (qd.FastUnionFindDecoder(g).decode_batch(S) == qd.UnionFindDecoder(g).decode_batch(S)).all()


def test_window_search_agrees_with_reference_matching_at_distance_7():
    """Large defect groups (window search) must stay close to PyMatching where greedy does not."""
    import pymatching
    c = qd.surface_code_circuit(7, p=0.004); g = qd.DecodingGraph(c); S, O = qd.sample(c, 3000, seed=4)
    ref = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True)).decode_batch(S)[:, 0]
    lns = qd.FastMWPMDecoder(g).decode_batch(S); greedy = qd.FastMWPMDecoder(g, large="greedy").decode_batch(S)
    assert (lns == ref).mean() > 0.995
    assert (lns != O).mean() < (greedy != O).mean()


def test_pair_cancel_clears_every_isolated_two_detector_fault():
    for g in (graph(3), graph(5)):
        pre = qd.PairCancel(g)
        for e in range(len(g.eu)):
            u, v = int(g.eu[e]), int(g.ev[e])
            if g.is_boundary(u) or g.is_boundary(v):
                continue
            s = np.zeros(g.n, np.uint8); s[u] = s[v] = 1
            R, obs = pre.apply(s)
            assert R.sum() == 0 and obs[0] == g.eobs[e], e
        s = np.zeros(g.n, np.uint8); s[0] = 1                    # a lone detector is left alone
        R, obs = pre.apply(s)
        assert R.sum() == 1 and obs[0] == 0


def test_pair_cancel_layer_helps_the_rule_machine():
    c = qd.surface_code_circuit(5, p=0.003); g = qd.DecodingGraph(c)
    Str, Otr = qd.sample(c, 12000, seed=2); S, O = qd.sample(c, 6000, seed=1)
    pre = qd.PairCancel(g); Rtr, dtr = pre.apply(Str); R, done = pre.apply(S)
    plain = qd.CliffordClauseDecoder(g).fit(Str, Otr, epochs=4)
    layered = qd.CliffordClauseDecoder(g).fit(Rtr, Otr ^ dtr, epochs=4)
    e_plain = (plain.decode_batch(S) != O).mean(); e_layered = ((layered.decode_batch(R) ^ done) != O).mean()
    assert e_layered < e_plain
    hyb = qd.HybridDecoder(layered, fallback=qd.FastMWPMDecoder(g), predecode=pre)
    assert (hyb.decode_batch(S) != O).mean() < e_layered and hyb.fast / hyb.total > 0.5

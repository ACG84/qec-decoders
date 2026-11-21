# 10-minute talk outline

1. (1 min) The decoding problem: syndrome in, logical correction out, fast. The decoding graph.
2. (2 min) Matching: why minimum weight is the right objective; our exact subset-DP and the
   independent-group split that keeps it tractable; it reproduces PyMatching.
3. (2 min) Union-Find: grow and peel, why growth is local and hardware-friendly; accuracy cost.
4. (1 min) BP, local matching, greedy and lookup: one line each on the idea and where it breaks.
5. (3 min) Clifford-clause decoder, our own idea. A lookup table is one rule per syndrome; an AND
   of parities covers a whole subspace. Cut parities: every logical-boundary fault flips all
   cuts, any other single fault flips at most one, so cuts are votes. Alone: 6.5x better than
   plain parities at distance 5, still behind Union-Find, gets worse with distance (say so).
   In front of exact matching: 88% of shots answered by bitwise rules at matching-level accuracy.
6. (1 min) Results table and the accuracy-versus-speed picture.
7. (1 min) The threshold plot, the 2x shot-parallel measurement, and what we would build next.

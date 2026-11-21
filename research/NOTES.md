# Interpretable syndrome rules: a Tsetlin machine decoder with stabilizer-parity literals

Written from scratch on Nov 21, 2026 (MidQWest day 1). No code from the earlier Syndrome
Tsetlin Machine project is included; the idea of using stabilizer/GF(2) structure as the
literals comes from that project and should be disclosed as prior work.

## What it does

Stim simulates a rotated surface-code memory experiment under circuit-level depolarizing noise.
A two-class Tsetlin machine (TM) learns to predict the logical flip from the syndrome. Its clauses
are ANDs of literals, so the trained decoder is a list of rules a person can read.

Two literal sets are compared:
- raw detection events (one bit per stabilizer per round);
- stabilizer-parity literals: for each stabilizer, the XOR of its detection events over all
  rounds. This is a GF(2) feature read off the code structure. A measurement error fires the
  same stabilizer twice in time and cancels, a data error survives.

Baselines: no decoding, minimum-weight perfect matching (PyMatching), and a majority lookup
table on the parity bits (the ceiling for any rule set on those features with this much data).

## Results (logical error rate per shot; rounds = distance; 30,000 training shots)

Distance 3, 100,000 test shots:

| p | No decoding | MWPM | Lookup on parities | TM, raw detectors | TM, parity literals |
|---|---|---|---|---|---|
| 0.002 | 0.0440 | 0.0031 | 0.0054 | 0.0316 | 0.0072 |
| 0.005 | 0.1044 | 0.0171 | 0.0263 | 0.0711 | 0.0401 |
| 0.010 | 0.1882 | 0.0593 | 0.0826 | 0.1635 | 0.1190 |

Distance 5, 50,000 test shots:

| p | No decoding | MWPM | Lookup on parities | TM, raw detectors | TM, parity literals |
|---|---|---|---|---|---|
| 0.002 | 0.1077 | 0.0010 | 0.0355 | 0.1087 | 0.0670 |
| 0.005 | 0.2307 | 0.0138 | 0.1552 | 0.2399 | 0.1965 |
| 0.010 | 0.3507 | 0.0834 | 0.3316 | 0.3642 | 0.3342 |

## What the numbers say

- The literal set decides everything. On raw detectors the TM barely beats no decoding; with
  parity literals at distance 3 and p = 0.002 it cuts the logical error 6x and reaches within
  2.3x of matching, using 25 distinct rules over 8 bits.
- It is not competitive with matching, and it does not scale. At distance 5 the parity-literal TM
  is 67x worse than matching at p = 0.002. Collapsing over time discards the information matching
  uses, and the lookup-table ceiling on these features is itself far above matching.
- So the claim this supports is narrow: a small, readable rule set recovers most of the decoding
  gain on a distance-3 code once the literals respect the stabilizer structure. It is not a
  claim of decoder accuracy.

Single training seed, one split; TM settings were tuned at p = 0.005 on a separate validation
sample (`tune.py`). Error bars from shot noise are about ±0.0002 at the lowest rates.

## Run

    pip install stim pymatching numpy
    python run_experiment.py 3 30000 100000     # table for distance 3 -> results_d3.json
    python run_experiment.py 5 30000 50000
    python show_rules.py 0.002                  # the learned rules, with firing rate and accuracy

## Obvious next steps for the team

1. Richer GF(2) literals for distance 5: windowed time parities and pairwise space parities,
   chosen by a proposer instead of fixed, to raise the feature ceiling.
2. Prune and deduplicate rules after training (several are redundant or weak), then report
   accuracy vs number of rules.
3. Rule-level comparison with matching: which syndromes do the two disagree on.
4. Hardware: a distance-3 repetition or surface patch on IBM, training on real syndromes.
5. Cite Bhatia, "Quantum-Logic Tsetlin Machines" (arXiv:2608.18659), which already links
   Pauli-projector clauses to syndrome semantics.

Layout: `stm/tm.py` Tsetlin machine, `stm/qec.py` Stim circuits, parity literals, baselines.

## Comparison with other decoders on the same data (`compare.py`, p = 0.002)

Logical error rate; all learned models get the same 30,000 training shots.

| Decoder | Distance 3 | Distance 5 |
|---|---|---|
| MWPM (PyMatching, given the noise model) | 0.0031 | 0.0010 |
| Decision tree, 16 leaves, parity bits | 0.0050 | 0.0287 |
| MLP 2x128, parity bits | 0.0050 | 0.0110 |
| Gradient-boosted trees, raw detectors | 0.0058 | 0.0239 |
| Tsetlin machine, parity bits (this project) | 0.0072 | 0.0670 |
| Logistic regression, parity bits | 0.0085 | 0.0694 |
| Tsetlin machine, raw detectors | 0.0316 | 0.1087 |

The Tsetlin machine is currently the weakest non-linear learner here. A 16-leaf decision tree on
the same parity bits is as readable, more accurate, and trains in under a second. The result that
does hold across every model is that the stabilizer-parity literals help: they make small models
work that fail on raw detectors. Full output in `comparison_p0.002.txt`.

## Clifford clauses with learned parity literals (`stm/clifford.py`, `clifford_experiment.py`)

On syndrome bits a CNOT circuit is a GF(2)-linear map, so a clause that ANDs measured outputs of
a Clifford circuit is an AND of parity checks: the indicator of an affine GF(2) subspace, the
classical image of a stabilizer projector. The proposer learns the parity literals from data
alone (no code layout, no noise model): each round it XORs pairs of current literals that fire
together far more often than chance.

Logical error rate at p = 0.002 (16-leaf tree shown as a second rule learner on the same literals):

| Literals | d=3 TM | d=3 tree | d=5 TM | d=5 tree |
|---|---|---|---|---|
| Raw detectors (propositional clauses) | 0.0316 | 0.0265 | 0.1087 | 0.1035 |
| Learned Clifford literals, 5 CNOT rounds | 0.0124 | 0.0053 | 0.0926 | 0.0858 |
| Hand-written stabilizer parities | 0.0072 | 0.0050 | 0.0670 | 0.0287 |
| MWPM for reference | 0.0031 | | 0.0010 | |

- Distance 3: learned Clifford literals cut the TM's error 2.5x and the tree's 5x relative to raw
  detectors, and the tree matches the hand-written parities. The proposer rediscovered 6 of the 8
  stabilizer time-parities without being told the code.
- Distance 5: the proposer stalls. It finds pair parities but does not grow the longer chains
  (0 of 24 stabilizer parities recovered), so the gain is small. A better proposer is the open
  problem; the hand-written row shows what a correct one would reach.
- The TM is less able than the tree to exploit a large literal pool (it gets worse as the pool
  grows from 8 to 144 literals).
- No quantum advantage is claimed or implied here: on classical syndrome bits every parity literal
  is classically computable. On hardware the clause is a parity-check circuit (CNOT fan-in to an
  ancilla), which matters when the input is a quantum state and not a bit string.

Full output in `clifford_results_p0.002.txt`.

## Clifford clauses on quantum data (`pauli_clauses.py`)

Task: tell five-qubit-code states (random logical state per shot) from random stabilizer states
using single-shot measurements. Each literal may choose a single-qubit Clifford per qubit, which
amounts to choosing a Pauli basis (X, Y or Z) and a sign; a GF(2) null-space proposer returns
every parity that never varies on 40 training shots of the code class.

| Clause type | Checks | Accepts a random state | Balanced error |
|---|---|---|---|
| Z basis only (ordinary literals) | 0 | 100% | 0.500 |
| One learned basis | 1 | 52% | 0.259 |
| 2 copies, 2 learned bases | 2 | 25% | 0.126 |
| 4 copies, 4 learned bases | 4 | 6.4% | 0.032 |

Code states are never rejected. Across the 243 basis settings the proposer recovered all 15
non-identity elements of the code's stabilizer group and nothing else. Ordinary literals see
nothing, because the code has no Z-only stabilizer. A product basis can measure only one
independent check of this code per copy; an entangling Clifford before measurement (the code's
syndrome circuit) would measure all four on one copy, which is the next step.
Noise-free simulation, one code, 5 qubits.

## Entangling Clifford clauses learned by Bell difference sampling, with noise (`bell_clauses.py`)

Same task, with depolarizing noise on every qubit and readout flips, both at probability p.
Learning uses the code class only: Bell difference samples (4 copies each) score every Pauli by
how often it commutes with the samples; code stabilizers score near 1, logical operators 2/3,
everything else 1/2. The accepted set must pass a certificate (mutually commuting and closed
under products) or no clause is built. The clause is the Clifford circuit that maps four learned
generators to Z1..Z4, followed by an AND on four measured bits: all checks on one copy.

Probability of recovering the stabilizer group exactly (30 trials):

| p | 16 samples | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| 0 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.01 | 0.10 | 0.17 | 0.47 | 0.70 | 0.97 |
| 0.03 | 0 | 0 | 0 | 0 | 0 |

Classification with the learned clause (3,000 test shots per class):

| p | Clause | Copies per decision | Misses code state | Accepts random state | Balanced error |
|---|---|---|---|---|---|
| 0 | entangling Clifford | 1 | 0.000 | 0.058 | 0.029 |
| 0 | product basis | 4 | 0.000 | 0.055 | 0.028 |
| 0.01 | entangling Clifford | 1 | 0.082 | 0.067 | 0.074 |
| 0.01 | product basis | 4 | 0.238 | 0.055 | 0.147 |

- The entangling clause reads all four checks from one copy and accepts random states at about
  1/16, the floor for four checks. Under 1% noise it halves the error of the product-basis clause
  while using a quarter of the copies, because fewer noisy qubits are involved.
- At 3% noise the learner fails and the certificate rejects its output. A stabilizer's score falls
  to about 0.68 there (20 noisy qubits per sample), which is indistinguishable from the 2/3 of a
  logical operator. More samples do not fix this; a different statistic or a fixed logical state would.
- Scoring enumerates all 4^n Paulis, fine for 5 qubits and not scalable. The scalable route is
  the GF(2) span of the samples and its symplectic complement, not implemented here.
- One code, 5 qubits, simulation only. Numbers vary by a few points between runs.

## Scalable learner (`scalable_learner.py`)

Replaces the enumeration of all 4^n Paulis. Bell difference samples lie in the normalizer N(G)
of the code's stabilizer group, an (n + k)-dimensional GF(2) space, so G is the symplectic
complement of the span of the samples: Gaussian elimination, O(n^3), no knowledge of k (sampling
stops once the span has not grown for 8 samples). Tested on random [[n, k]] stabilizer codes.

Noise-free, 5 random codes per size:

| n | k | Samples used | Copies (4 per sample) | Exact recovery | Learn time |
|---|---|---|---|---|---|
| 5 | 1 | 15 | 60 | 5/5 | 0.01 s |
| 20 | 2 | 32 | 127 | 5/5 | 0.04 s |
| 40 | 4 | 55 | 218 | 5/5 | 0.17 s |
| 80 | 8 | 98 | 394 | 5/5 | 0.8 s |
| 120 | 12 | 142 | 568 | 5/5 | 2.1 s |

Samples needed are n + k plus the stopping margin: linear in n. The one-copy clause built from
the learned generators never rejected a code state and never accepted a random state at n >= 10
(200 shots each; the expected false-accept rate is 2^-(n-k)).

With noise the span method breaks, because one corrupted sample pollutes the span. The fallback
is information-set search: random subsets of n + k + 2 samples, keeping one whose complement has
the right dimension, commutes with itself, and commutes with a clear majority of all 400 samples.

| n | k | p | Exact recovery (5 codes) |
|---|---|---|---|
| 5 | 1 | 0.01 | 5/5 |
| 8 | 1 | 0.01 | 4/5 |
| 12 | 1 | 0.005 | 5/5 |
| 5 | 1 | 0.03 | 0/5 |
| 12 | 1 | 0.01 | 0/5 |
| 16 | 2 | 0.005 | 0/5 |
| 24 | 2 | 0.005 | 0/5 |

It works while (noise rate x qubits) is small and fails beyond, within the 4,000-subset budget.
This is expected: finding a Pauli that commutes with most of a set of noisy samples is a
learning-parity-with-noise problem, and the subset search costs exponentially in noise x size.
Scaling the noisy case needs either error-mitigated samples or structure (for example local
generators), neither attempted here. Results vary run to run; `scalable_results.txt` has one run.

## Noise-robust learning at high qubit counts (`local_learner.py`)

Two changes. (1) Two copies per sample: a Bell sample P of a code state has a fixed commutation
sign with every code stabilizer g (set by the number of Y's in g), whatever the logical state, so
a stabilizer's commute fraction is 0 or 1 and everything else sits near 1/2. (2) Local search:
score every Pauli on each small neighbourhood of the known qubit layout and accept those more
than 5 standard deviations from 1/2. A weight-w check is disturbed only by noise on its own 2w
qubit copies, so its score does not fall as the device grows.

Rotated surface code, random logical state per shot:

| Qubits | p = 1%, 300 samples | p = 3%, 300 samples | p = 5%, 1,000 samples | p = 8%, 1,000 samples |
|---|---|---|---|---|
| 9 | full group | full group | full group | partial |
| 49 | full group | full group | full group | partial |
| 121 | full group | full group | full group | partial |
| 225 | full group | full group | full group | partial |

"Full group" means the accepted checks span the code's whole stabilizer group. No false check was
accepted in any run, including the partial ones. The weakest accepted score depends on the noise
rate and not on size (about 0.86 at 1%, 0.71 at 3%, 0.63 at 5%, from 9 to 225 qubits). 225 qubits
with 1,000 samples took 7 s including simulation.

Assumptions that carry this result: the checks are local (weight <= 4 here), the layout is
known so neighbourhoods can be listed, and the noise is independent per qubit. For random codes
with high-weight checks the limits in the previous section still apply. One run; `local_results.txt`.

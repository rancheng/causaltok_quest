# Quest — From InfoTok to CausalTok

## One-line challenge

**Find the shortest discrete language sufficient to predict how a physical world responds to every admissible action.**

You are not given object, contact, depth, semantic, geometry, force, or task labels. If such concepts are useful, they must emerge because prediction requires them.

---

## 1. Warm-up — From surprisal to code length

Let a source draw symbols `x` with probability `p(x)`. A tokenizer emits a prefix-free variable-length string over an alphabet of size `C`.

### Math

Derive a lower bound on expected code length and construct a code within a constant of it. Explain when an individual sample should receive approximately

`N(x) ~= -log_C p(x)`

tokens.

### Code

```python
def optimal_prefix_code(probabilities, alphabet_size):
    """Return one C-ary codeword, as a list of integers, for each symbol."""
```

Hidden binary instances may contain up to `10^6` symbols. The evaluator checks prefix-freeness, exact expected length, optimality, and runtime.

---

## 2. The nuisance paradox

Let an observation be `X=(S,N)`, where `S` determines future physical consequences and `N` is independent high-entropy nuisance.

### Math

Compare the minimum rate for exact reconstruction of `X` with the minimum rate required to preserve all intervention-conditioned future consequences. Construct a counterexample where raw surprisal allocates more code to an irrelevant distinction than to a physically decisive one.

### Code — Fresh Nuisance Attack

The public kit provides `FreshNuisanceStream`. Repeated observation of the same physical state returns a stable `sensor` payload and **fresh random `metadata` bytes on every visit**. Hidden tests sweep nuisance entropy over several orders of magnitude.

A strong solution should make the increase in causal rate approach zero as nuisance entropy grows.

---

## 3. What is a physical state?

For a history `h` and future action string `u`, let `Y(h,u)` denote the complete future consequence trace.

Define the strongest equivalence relation under which two histories may share one token state whenever no admissible future intervention can distinguish them.

Prove:

1. it is an equivalence relation;
2. every exactly sufficient deterministic tokenizer refines this partition;
3. encoding the equivalence class itself is sufficient;
4. if there are `K` classes, exact fixed-length binary representation requires at least `ceil(log2 K)` bits in the worst case;
5. derive the expected C-ary prefix-code bound for the class distribution.

---

## 4. Exact coding challenge — Minimize a controlled physical machine

The organizer supplies a deterministic Mealy-style world through:

```python
FiniteWorld(
    transitions=...,
    consequences=...,
    observations=...,
    probabilities=...,
)
```

For state `s` and action `a`, the world emits `consequences[s][a]` and enters `transitions[s][a]`.

### Required API

```python
def causal_partition(world: FiniteWorld) -> list[int]:
    """Return canonical class_id[s] for every state s."""
```

Canonical IDs are consecutive `0..K-1`, ordered by the smallest raw-state index in each class.

### Hidden constraints

- Bronze: `n <= 500`, `m <= 8`
- Silver: `n <= 2e4`, `m <= 16`
- Gold: `n <= 1e6`, `m <= 8`

Naive pairwise comparison is intentionally too slow for Gold.

### Public vs hidden verification

The public checker only searches for **short distinguishing action strings** up to a declared horizon. Passing it does **not** prove exact soundness or minimality.

The hidden judge separately checks exact soundness and independently computes the coarsest valid partition.

---

## 5. Emit the shortest causal code

Given canonical `class_id[s]` and state probability `p_s`, aggregate class probability

`P(c) = sum_{s in c} p_s`.

Implement:

```python
def causal_code(class_id, state_probability, alphabet_size):
    """Return one C-ary prefix codeword for each causal class."""
```

The evaluator computes the true expected code length and checks exact finite-instance optimality. Report redundancy relative to C-ary entropy.

---

## 6. Public Boss Worlds

The public package exposes representative generators; hidden parameters and hidden generators differ.

- **Random Texture Explosion** — many nuisance-expanded raw states share a much smaller physical machine.
- **One-Bit Contact** — a tiny distinction changes an action consequence and must never be merged.
- **Delayed Distinction** — states are identical under short action strings and differ only at a long depth.
- **Huge Duplicate World** — very many raw nodes are nuisance copies of a tiny machine.
- **Rare Critical State** — the decisive causal class has total probability approximately `1e-6`, but exact sufficiency still forbids merging it away.
- **Fresh Nuisance Stream** — every visit generates new random nuisance bytes while the physical state stays fixed.

---

## 7. Approximate stochastic world — Rate–distortion

For stochastic dynamics, the public kit defines `StochasticWorld`, where each `(state, action)` has a distribution over `(consequence, next_state)`.

For the executable track, the organizer uses a deterministic partition `Z=f(S)` and scores

`J = H_B(Z) + beta * D`,

where `D` is the state/action averaged KL divergence between each state's distribution over `(consequence, next_class)` and the corresponding class-conditional mixture distribution.

### Math

Study the information–distortion tradeoff, including at least four of: monotonicity, convexity/counterexamples, zero-distortion limit, large-distortion limit, deterministic vs stochastic encoders, Lagrangian form, relation to classical rate–distortion, relation to exact causal states.

### Code

```python
def approximate_partition(world: StochasticWorld, beta: float) -> list[int]:
    ...
```

For small hidden worlds the organizer computes the global optimum independently. The public `score_approximate_partition` only scores your proposal; it does not optimize it.

---

## 8. Online extension — When should a new token be emitted?

Long stretches of unchanged predictive state should cost almost no communication; a rare physical event may require a new token.

To prevent side channels, encoder and decoder are evaluated as separate objects:

```python
class StreamingEncoder:
    def reset(self): ...
    def observe(self, observation, previous_action) -> list[int]: ...

class StreamingDecoder:
    def reset(self): ...
    def consume(self, symbols): ...
    def predict(self, query_action): ...
```

Only emitted symbols may pass from encoder to decoder. Hidden score is prediction error plus a rate penalty; RAM is capped and metered.

The public kit contains `public_stream`, with continuously changing nuisance and a rare physical-state transition.

---

## 9. Optional Final Boss — From finite worlds to robot video

You are given frozen dense video features and action-labelled robot trajectories. Learn a variable-rate bottleneck `Z_t` that predicts future frozen features conditioned on future actions.

Conceptual objective:

`future predictive loss + beta * representation rate`.

No supervised semantic, object, depth, contact, optical-flow, geometry, or reconstruction objective may be added. Those concepts may only be used after training as probes.

The goal is to move the rate–future-predictability frontier left.

---

## 10. Submission package

```text
solution.pdf
src/
  exact.py
  coding.py
  approximate.py
  streaming.py
README.md
counterexample.json
complexity.md
```

The Final Boss may add `video/` training code and results.

Every theorem must be marked **proved**, **assumed**, or **conjectured**. Every complexity claim must state the data structure that makes it achievable.

---

## 11. Scoring

- Info-theoretic derivation — 15
- Causal-state theorem — 20
- Exact partition algorithm — 25
- Algorithmic efficiency — 15
- Shortest causal code — 10
- Approximate extension — 10
- Originality / new theorem — 5

**100 points + uncapped research bonus.**

Automatic elimination includes exact-soundness failure on decisive hidden cases, future leakage, side channels, hard-coded public parameters, and irreproducible results.

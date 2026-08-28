# Quest — From InfoTok to CausalTok

## The challenge

Suppose an agent observes a world, takes actions, and sees consequences. You want to construct the **shortest discrete representation of its past that preserves everything necessary to predict the consequences of future interventions**.

You are not optimizing pixel reconstruction. You are not given semantic, object, depth, contact, force, or geometry labels.

The target is a representation in which useful physical concepts emerge because they are necessary for prediction.

---

## Part A — From surprisal to code length

Let a source emit symbols `x` with probabilities `p(x)`. Consider a prefix-free code over an alphabet of size `B`.

1. Derive a lower bound on the expected code length.
2. Characterize a near-optimal code length for each symbol.
3. Explain the relation between code length and surprisal.
4. Implement a binary prefix-code constructor and compare its expected length with entropy.

Your code will be tested on distributions with up to `10^6` symbols and highly skewed probabilities.

---

## Part B — The nuisance paradox

An observation is

```text
X = (S, N)
```

where `S` contains a few bits that determine all future physical consequences, while `N` contains hundreds or thousands of random nuisance bits independent of dynamics.

1. Compare the optimal reconstruction rate with the optimal predictive rate.
2. Construct the smallest counterexample showing why raw observation surprisal can be the wrong quantity for a robot.
3. State what quantity should replace it if the goal is to predict intervention-conditioned futures.

Your explanation must be accompanied by an executable example.

---

## Part C — Exact finite physical universe

A finite deterministic world has states `s`, actions `a`, physical consequences `y`, and successor states `s'`:

```text
(s, a) -> (y, s')
```

You receive the transition and consequence tables but no semantic state labels.

A history/state representation is **exactly sufficient** if states assigned the same code can never be distinguished by any future action sequence through their consequence traces.

### Mathematical tasks

1. Define the correct equivalence relation on states (or histories).
2. Prove that it is an equivalence relation.
3. Prove a lower bound on the number of internal representation states required by any exact deterministic encoder.
4. Characterize when the lower bound is attainable.
5. Derive worst-case and expected description-length bounds.

### Coding task

Implement:

```python
def minimize_world(world) -> list[int]:
    ...
```

The output contains one class ID per raw state.

Hidden tests measure:

- **soundness**: merged states really are intervention-indistinguishable;
- **minimality**: no unnecessary distinctions remain;
- **runtime** and **memory**;
- behavior on large nuisance-expanded worlds;
- long delayed distinctions that defeat fixed-horizon heuristics.

Official limits may include up to roughly `10^6` raw states and a small action alphabet.

---

## Part D — Public adversarial families

The public package includes representative generators. Hidden generators will not be identical.

### 1. Random texture trap

Many raw states differ only by irrelevant nuisance IDs. The physical machine is much smaller than the observation space.

### 2. One-bit contact trap

A tiny binary distinction changes the consequence of an action.

### 3. Delayed distinction

Two states look identical under short tests and differ only after a long intervention suffix.

### 4. Huge duplicate world

A very large raw machine is formed from many nuisance copies of a much smaller physical machine.

### 5. Rare critical state

A low-probability state can be physically decisive. Probability alone must not justify merging it away in the exact track.

---

## Part E — Minimal code for the discovered physical states

After computing your partition, aggregate state probabilities into class probabilities `p(c)` and construct a prefix-free code.

Report:

```text
entropy of causal classes
expected code length
expected code length - entropy
raw-state entropy
compression ratio relative to raw-state coding
```

The evaluator checks the actual codewords, not just reported lengths.

---

## Part F — Approximate physical equivalence

Exact equality is unrealistic in stochastic and continuous worlds.

Let a representation be `Z=f(H)`, where `H` is observation/action history. Propose a distortion based on the difference between future consequence distributions under interventions.

Study an information-constrained objective of the form

```text
minimize information rate
subject to predictive distortion <= epsilon
```

Tasks:

1. define the distortion precisely;
2. prove basic properties of the optimal rate as a function of `epsilon`;
3. identify the zero-distortion limit;
4. design a finite-sample estimator;
5. implement it on a stochastic extension of the public worlds.

No particular known formalism is required; justify your choices from first principles.

---

## Part G — Streaming token emission

Remove the assumption that one time step must emit one token.

Design an online encoder that decides when a new token is necessary. Long periods in which the predictive state does not change should be representable with very low rate; rare physical events may trigger new tokens.

Your memory usage may not grow linearly with stream duration.

Give both:

- a mathematical token-boundary criterion;
- a working streaming implementation.

---

## Final Boss — Toward robot video

You are given dense features from a frozen pretrained video representation model and action-labelled robot trajectories.

Design a variable-rate bottleneck

```text
past video features -> compact tokens
compact tokens + future actions -> future video features
```

with the conceptual objective

```text
future predictive loss + beta * representation rate
```

Do **not** add supervised semantic/contact/depth/geometry objectives. Those concepts are evaluation probes only.

The goal is to move the rate–future-predictability curve left: fewer tokens at equal action-conditioned predictive quality.

An exceptional result should additionally show that object permanence, contact, geometry, affordance, or other physical concepts become simply decodable despite never being direct training targets.

---

## Deliverables

1. `solution.pdf` — definitions, proofs, bounds, complexity, counterexamples.
2. `solution.py` / C++ equivalent — exact finite-world solver.
3. Prefix-code constructor.
4. Runtime and memory report.
5. At least one original adversarial world that breaks a naive method.
6. Optional approximate/streaming/final-boss extensions.

## What we are deliberately not telling you

We are not asking for a named theorem, a particular automaton algorithm, a particular information-theoretic formalism, or a neural-network architecture.

The strongest solution should make a huge observation space disappear and expose the smallest predictive physical machine underneath it.

# CausalTok Quest

> **What is the shortest discrete language sufficient to predict how a physical world responds to intervention?**

This is the **public candidate kit**. It contains executable interfaces, public adversarial generators, public scorers, starter files, and tests. It intentionally does **not** contain reference minimizers, organizer-only generators, exact hidden verifiers, optimal partitions, or model answers.

## Quick start

```bash
python -m pip install -e .[dev]
pytest -q
python -m causaltok.cli list-worlds
python -m causaltok.cli export-world texture --out /tmp/texture.json
python -m causaltok.cli inspect /tmp/texture.json
```

Read [`QUEST.md`](QUEST.md), then implement the files under [`starter/`](starter/).

## Canonical public APIs

### Exact causal partition

```python
def causal_partition(world: FiniteWorld) -> list[int]:
    """Return canonical class_id[s] for every raw state s."""
```

Class IDs must be consecutive `0..K-1`, ordered by the smallest raw-state index contained in each class.

### C-ary prefix code

```python
def optimal_prefix_code(
    probabilities: list[float],
    alphabet_size: int,
) -> list[list[int]]:
    """Return one prefix codeword per source symbol."""
```

Each code symbol must lie in `0..alphabet_size-1`.

### Shortest causal code

```python
def causal_code(
    class_id: list[int],
    state_probability: list[float],
    alphabet_size: int,
) -> list[list[int]]:
    ...
```

### Approximate stochastic track

```python
def approximate_partition(world: StochasticWorld, beta: float) -> list[int]:
    ...
```

The public scorer evaluates `J = H_B(Z) + beta * D`; it does not search for the optimum.

### Streaming track

```python
class StreamingEncoder:
    def reset(self): ...
    def observe(self, observation, previous_action) -> list[int]: ...

class StreamingDecoder:
    def reset(self): ...
    def consume(self, symbols): ...
    def predict(self, query_action): ...
```

The evaluator instantiates encoder and decoder separately. Only emitted symbols may pass between them.

## Public tools vs hidden judge

The public checker deliberately searches only for **bounded-horizon counterexamples**. Passing it is not a proof of exact soundness or minimality. The hidden judge uses a separate exact verifier.

The public package also includes:

- deterministic adversarial worlds;
- fresh nuisance observations whose metadata is regenerated on every visit;
- C-ary prefix-code validation and rate scoring;
- a stochastic-world rate–distortion scorer;
- a simple streaming benchmark generator.

## Repository layout

```text
QUEST.md
causaltok/
  world.py
  public_worlds.py
  public_check.py
  coding.py
  nuisance.py
  stochastic.py
  streaming.py
  cli.py
starter/
  exact.py
  coding.py
  approximate.py
  streaming.py
tests/
  test_public.py
```

## Design principle

A visually complicated observation can be physically irrelevant. A one-bit distinction can completely change the consequence of an action. Spend bits on the second kind of information, not the first.

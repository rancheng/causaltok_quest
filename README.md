# CausalTok Quest

> **What is the shortest discrete language sufficient to predict how a physical world responds to intervention?**

CausalTok is a math + coding quest for strong algorithmic and mathematical students. It starts from a deliberately small, exact setting where every claim can be checked by code, then asks you to generalize toward stochastic worlds, streaming representations, and robot video.

This public repository contains the **candidate materials only**. It intentionally does **not** contain reference minimizers, organizer-only generators, optimal partitions, hidden tests, or model answers.

## Quick start

```bash
python -m pip install -e .[dev]
pytest -q
python -m causaltok.cli list-worlds
python -m causaltok.cli export-world texture --out /tmp/texture.json
python -m causaltok.cli inspect examples/public_world.json
```

Start with [`QUEST.md`](QUEST.md), then implement [`starter/solution.py`](starter/solution.py).

## Submission API

Your exact finite-world solver must implement:

```python
def minimize_world(world) -> list[int]:
    """Return one integer class id for each raw state."""
```

Class IDs do not need to be consecutive. Only the induced partition matters.

For the coding track, implement:

```python
def build_prefix_code(class_probabilities: list[float]) -> dict[int, str]:
    """Return a binary prefix code for class indices 0..K-1."""
```

The public checker can search for **short counterexamples** and validate **prefix-freeness**, but it does not prove soundness, compute minimality, or reveal the hidden optimum.

## Repository layout

```text
QUEST.md                    full candidate specification
causaltok/world.py          finite controlled-world format
causaltok/public_worlds.py  representative public generators
causaltok/public_check.py   finite-horizon public counterexample search
causaltok/coding.py         entropy + code validation/scoring only
causaltok/cli.py            public CLI tools
starter/solution.py         deliberately trivial baseline
examples/public_world.json  small public example
```

## Design principle

A visually complicated observation can be physically irrelevant. A one-bit distinction can completely change the consequence of an action. Your representation should spend bits on the second kind of information, not the first.

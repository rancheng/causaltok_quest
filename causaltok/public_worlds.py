from __future__ import annotations
import random
from .world import FiniteWorld


def _duplicate_base(base_t, base_y, duplicates: int, seed: int = 0, rare_state: int | None = None):
    if duplicates < 1:
        raise ValueError("duplicates must be >= 1")
    rng = random.Random(seed)
    k, m = len(base_t), len(base_t[0])
    n = k * duplicates
    transitions, consequences, observations = [], [], []
    for causal_state in range(k):
        for nuisance_copy in range(duplicates):
            row_t = []
            for action in range(m):
                next_causal = base_t[causal_state][action]
                next_copy = rng.randrange(duplicates)
                row_t.append(next_causal * duplicates + next_copy)
            transitions.append(tuple(row_t))
            consequences.append(tuple(base_y[causal_state]))
            observations.append({
                "texture_id": rng.getrandbits(64),
                "lighting_id": rng.getrandbits(32),
                "copy_index": nuisance_copy,
            })
    if rare_state is None:
        probabilities = [1.0 / n] * n
    else:
        probabilities = [1.0] * n
        for d in range(duplicates):
            probabilities[rare_state * duplicates + d] = 1e-4
        z = sum(probabilities)
        probabilities = [p / z for p in probabilities]
    return FiniteWorld(tuple(transitions), tuple(consequences), tuple(observations), tuple(probabilities))


def random_texture_trap(duplicates: int = 64, seed: int = 0) -> FiniteWorld:
    base_t = ((0, 1), (2, 1), (2, 0))
    base_y = (("idle", "touch"), ("slide", "hold"), ("fall", "reset"))
    return _duplicate_base(base_t, base_y, duplicates, seed)


def one_bit_contact(duplicates: int = 32, seed: int = 1) -> FiniteWorld:
    base_t = ((0, 1), (1, 0))
    base_y = (("object_stays", "approach"), ("object_moves", "release"))
    return _duplicate_base(base_t, base_y, duplicates, seed)


def delayed_distinction(depth: int = 12) -> FiniteWorld:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    transitions, consequences = [], []
    for branch in range(2):
        for i in range(depth + 1):
            idx = branch * (depth + 1) + i
            nxt = branch * (depth + 1) + min(i + 1, depth)
            transitions.append((nxt, idx))
            if i == depth:
                consequences.append(("wait", f"terminal_{branch}"))
            else:
                consequences.append(("wait", "same"))
    return FiniteWorld(tuple(transitions), tuple(consequences))


def duplicate_world(causal_states: int = 50, duplicates: int = 20, actions: int = 3, seed: int = 2) -> FiniteWorld:
    if causal_states < 1 or actions < 1:
        raise ValueError("causal_states and actions must be >= 1")
    rng = random.Random(seed)
    base_t, base_y = [], []
    for state in range(causal_states):
        base_t.append(tuple(rng.randrange(causal_states) for _ in range(actions)))
        base_y.append(tuple((a, (state * 131 + a * 17) % max(3, causal_states // 2 + 1)) for a in range(actions)))
    return _duplicate_base(tuple(base_t), tuple(base_y), duplicates, seed)


def rare_critical_state(duplicates: int = 16, seed: int = 3) -> FiniteWorld:
    base_t = ((0, 1), (1, 2), (2, 0))
    base_y = (("normal", "normal"), ("normal", "normal"), ("normal", "catastrophic"))
    return _duplicate_base(base_t, base_y, duplicates, seed, rare_state=2)


PUBLIC_WORLDS = {
    "texture": random_texture_trap,
    "contact": one_bit_contact,
    "delayed": delayed_distinction,
    "duplicate": duplicate_world,
    "rare": rare_critical_state,
}

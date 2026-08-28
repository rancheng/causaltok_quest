"""Candidate starter for the exact finite-world track.

This baseline is deliberately conservative: it keeps every raw state separate.
"""


def causal_partition(world):
    return list(range(world.n_states))

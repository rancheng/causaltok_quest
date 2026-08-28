"""Candidate starter implementation.

The baseline below is intentionally trivial: every raw state is kept separate.
It is always conservative, but it does not solve the minimization problem.
"""


def minimize_world(world):
    return list(range(world.n_states))


def build_prefix_code(class_probabilities):
    raise NotImplementedError("Implement your own prefix-code constructor")

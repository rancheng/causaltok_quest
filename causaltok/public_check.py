from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from .world import FiniteWorld


@dataclass(frozen=True)
class PublicCheckReport:
    passed: bool
    n_classes: int
    message: str
    counterexample: tuple[int, int, tuple[int, ...]] | None = None


def _canonicalize(labels):
    remap = {}
    out = []
    for value in labels:
        if value not in remap:
            remap[value] = len(remap)
        out.append(remap[value])
    return out


def check_partition_public(world: FiniteWorld, labels, max_horizon: int = 4) -> PublicCheckReport:
    """Search for short distinguishing action strings inside a proposed class.

    Passing this public checker is NOT a proof of soundness or minimality.
    Hidden evaluation may use arbitrarily longer distinguishing sequences and
    a separate exact verifier.
    """
    if len(labels) != world.n_states:
        return PublicCheckReport(False, -1, "partition length mismatch")
    labels = _canonicalize(labels)
    members: dict[int, list[int]] = {}
    for state, cls in enumerate(labels):
        members.setdefault(cls, []).append(state)

    for horizon in range(1, max_horizon + 1):
        for actions in product(range(world.n_actions), repeat=horizon):
            for states in members.values():
                if len(states) < 2:
                    continue
                base = states[0]
                base_trace = world.trace(base, actions)
                for state in states[1:]:
                    if world.trace(state, actions) != base_trace:
                        return PublicCheckReport(
                            False,
                            len(members),
                            f"found a distinguishing action string of length {horizon}",
                            (base, state, tuple(actions)),
                        )
    return PublicCheckReport(
        True,
        len(members),
        f"no counterexample found up to horizon {max_horizon}; hidden exact verification is stronger",
    )

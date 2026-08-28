from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import math
from typing import Hashable

Outcome = tuple[float, Hashable, int]


@dataclass(frozen=True)
class StochasticWorld:
    """Small stochastic controlled world for the approximate track.

    outcomes[s][a] is a tuple of (probability, consequence, next_state).
    """

    outcomes: tuple[tuple[tuple[Outcome, ...], ...], ...]
    state_probability: tuple[float, ...]

    def __post_init__(self):
        n = len(self.outcomes)
        if n == 0 or len(self.state_probability) != n:
            raise ValueError("invalid world size")
        if any(p < 0 for p in self.state_probability) or abs(sum(self.state_probability) - 1.0) > 1e-9:
            raise ValueError("state_probability must sum to 1")
        m = len(self.outcomes[0])
        if m == 0:
            raise ValueError("at least one action is required")
        for s, row in enumerate(self.outcomes):
            if len(row) != m:
                raise ValueError("ragged action table")
            for action_outcomes in row:
                if not action_outcomes:
                    raise ValueError("each state/action needs at least one outcome")
                if abs(sum(p for p, _, _ in action_outcomes) - 1.0) > 1e-9:
                    raise ValueError("outcome probabilities must sum to 1")
                for p, _, next_state in action_outcomes:
                    if p < 0 or not 0 <= next_state < n:
                        raise ValueError(f"invalid stochastic outcome at state {s}")

    @property
    def n_states(self) -> int:
        return len(self.outcomes)

    @property
    def n_actions(self) -> int:
        return len(self.outcomes[0])


def _entropy_base(probabilities, alphabet_size: int) -> float:
    return -sum(p * math.log(p, alphabet_size) for p in probabilities if p > 0)


def score_approximate_partition(world: StochasticWorld, class_id, beta: float, alphabet_size: int = 2) -> dict[str, float]:
    """Score a deterministic partition using J = H_B(Z) + beta * D.

    D is the state/action averaged KL divergence between each state's outcome
    distribution over (consequence, next_class) and the class-conditional
    mixture distribution. This scorer evaluates a proposal; it does not search
    for the optimum.
    """
    labels = list(class_id)
    if len(labels) != world.n_states:
        raise ValueError("partition length mismatch")
    if min(labels) != 0 or set(labels) != set(range(max(labels) + 1)):
        raise ValueError("class IDs must be consecutive 0..K-1")
    if alphabet_size < 2:
        raise ValueError("alphabet_size must be >= 2")

    k = max(labels) + 1
    class_prob = [0.0] * k
    for s, p in enumerate(world.state_probability):
        class_prob[labels[s]] += p
    rate = _entropy_base(class_prob, alphabet_size)

    mixtures: dict[tuple[int, int], dict[tuple[Hashable, int], float]] = {}
    for c in range(k):
        states = [s for s in range(world.n_states) if labels[s] == c]
        mass = class_prob[c]
        for a in range(world.n_actions):
            q = defaultdict(float)
            if mass > 0:
                for s in states:
                    weight_s = world.state_probability[s] / mass
                    for p, y, sp in world.outcomes[s][a]:
                        q[(y, labels[sp])] += weight_s * p
            mixtures[(c, a)] = dict(q)

    distortion = 0.0
    for s, ps in enumerate(world.state_probability):
        c = labels[s]
        for a in range(world.n_actions):
            p_dist = defaultdict(float)
            for p, y, sp in world.outcomes[s][a]:
                p_dist[(y, labels[sp])] += p
            q = mixtures[(c, a)]
            kl = 0.0
            for key, p in p_dist.items():
                if p == 0:
                    continue
                qv = q.get(key, 0.0)
                if qv <= 0:
                    kl = math.inf
                    break
                kl += p * math.log(p / qv)
            distortion += ps * kl / world.n_actions

    return {"rate": rate, "distortion": distortion, "objective": rate + beta * distortion}


def tiny_stochastic_world() -> StochasticWorld:
    return StochasticWorld(
        outcomes=(
            (((0.9, "stay", 0), (0.1, "move", 1)),),
            (((0.2, "stay", 1), (0.8, "move", 0)),),
        ),
        state_probability=(0.5, 0.5),
    )

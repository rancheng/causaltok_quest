from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Hashable, Sequence

Consequence = Hashable


@dataclass(frozen=True)
class FiniteWorld:
    """Finite deterministic controlled world.

    For each state s and action a:
      consequences[s][a] is emitted, then
      transitions[s][a] is the next state.
    """

    transitions: tuple[tuple[int, ...], ...]
    consequences: tuple[tuple[Consequence, ...], ...]
    observations: tuple[object, ...] | None = None
    probabilities: tuple[float, ...] | None = None

    def __post_init__(self):
        n = len(self.transitions)
        if n == 0:
            raise ValueError("world must contain at least one state")
        if len(self.consequences) != n:
            raise ValueError("transition/consequence row count mismatch")
        m = len(self.transitions[0])
        if m == 0:
            raise ValueError("world must contain at least one action")
        for s, (tr, out) in enumerate(zip(self.transitions, self.consequences)):
            if len(tr) != m or len(out) != m:
                raise ValueError(f"ragged action table at state {s}")
            if any(t < 0 or t >= n for t in tr):
                raise ValueError(f"invalid successor at state {s}")
        if self.observations is not None and len(self.observations) != n:
            raise ValueError("observation count mismatch")
        if self.probabilities is not None:
            if len(self.probabilities) != n or any(p < 0 for p in self.probabilities):
                raise ValueError("invalid probabilities")
            if abs(sum(self.probabilities) - 1.0) > 1e-9:
                raise ValueError("probabilities must sum to 1")

    @property
    def n_states(self) -> int:
        return len(self.transitions)

    @property
    def n_actions(self) -> int:
        return len(self.transitions[0])

    def step(self, state: int, action: int):
        return self.transitions[state][action], self.consequences[state][action]

    def trace(self, state: int, actions: Sequence[int]) -> tuple[Consequence, ...]:
        out = []
        for action in actions:
            state, consequence = self.step(state, action)
            out.append(consequence)
        return tuple(out)

    def to_dict(self) -> dict:
        return {
            "transitions": [list(r) for r in self.transitions],
            "consequences": [list(r) for r in self.consequences],
            "observations": list(self.observations) if self.observations is not None else None,
            "probabilities": list(self.probabilities) if self.probabilities is not None else None,
        }

    def to_json(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "FiniteWorld":
        return cls(
            tuple(tuple(map(int, r)) for r in data["transitions"]),
            tuple(tuple(r) for r in data["consequences"]),
            tuple(data["observations"]) if data.get("observations") is not None else None,
            tuple(map(float, data["probabilities"])) if data.get("probabilities") is not None else None,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "FiniteWorld":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

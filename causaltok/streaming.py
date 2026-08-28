from __future__ import annotations
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class StreamingStep:
    observation: dict[str, int]
    previous_action: int | None
    query_action: int
    target_consequence: int


def public_stream(length: int = 100, event_step: int = 70, seed: int = 0):
    """A tiny public stream: nuisance changes constantly, physical mode changes once."""
    if not 0 <= event_step < length:
        raise ValueError("event_step must lie inside the stream")
    rng = random.Random(seed)
    mode = 0
    previous_action = None
    for t in range(length):
        if t == event_step:
            mode = 1
        query_action = t % 2
        target = mode ^ query_action
        yield StreamingStep(
            observation={"weak_signal": mode, "nuisance": rng.getrandbits(32)},
            previous_action=previous_action,
            query_action=query_action,
            target_consequence=target,
        )
        previous_action = query_action


def count_emitted_symbols(symbol_batches) -> int:
    return sum(len(batch) for batch in symbol_batches)

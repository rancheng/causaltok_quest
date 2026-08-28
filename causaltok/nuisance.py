from __future__ import annotations
import hashlib
import json
import random
from .world import FiniteWorld


class FreshNuisanceStream:
    """Generate a fresh nuisance payload on every observation of the same state.

    `sensor` is stable for a raw state. `metadata` is fresh random bytes whose
    length is controlled by nuisance_bits. Hidden evaluation can vary this rate.
    """

    def __init__(self, world: FiniteWorld, seed: int = 0):
        self.world = world
        self._rng = random.Random(seed)

    def _stable_sensor(self, state: int) -> bytes:
        if self.world.observations is None:
            payload = {"state_slot": state}
        else:
            payload = self.world.observations[state]
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).digest()[:16]

    def observe(self, state: int, nuisance_bits: int) -> dict[str, bytes]:
        if state < 0 or state >= self.world.n_states:
            raise ValueError("invalid state")
        if nuisance_bits < 0:
            raise ValueError("nuisance_bits must be >= 0")
        nbytes = (nuisance_bits + 7) // 8
        metadata = bytes(self._rng.getrandbits(8) for _ in range(nbytes))
        if nuisance_bits % 8 and metadata:
            keep = nuisance_bits % 8
            metadata = metadata[:-1] + bytes([metadata[-1] & ((1 << keep) - 1)])
        return {"sensor": self._stable_sensor(state), "metadata": metadata}

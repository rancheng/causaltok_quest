from __future__ import annotations
import math
from collections.abc import Sequence

Code = Sequence[Sequence[int]]


def _validate_probabilities(probabilities) -> list[float]:
    probs = [float(p) for p in probabilities]
    if not probs:
        raise ValueError("empty probability vector")
    if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-9:
        raise ValueError("probabilities must be nonnegative and sum to 1")
    return probs


def entropy(probabilities, alphabet_size: int = 2) -> float:
    probs = _validate_probabilities(probabilities)
    if alphabet_size < 2:
        raise ValueError("alphabet_size must be >= 2")
    return -sum(p * math.log(p, alphabet_size) for p in probs if p > 0)


def is_prefix_free(code: Code, alphabet_size: int) -> bool:
    if alphabet_size < 2:
        return False
    words = [tuple(word) for word in code]
    if any(len(word) == 0 for word in words):
        return False
    if any(any((not isinstance(symbol, int)) or symbol < 0 or symbol >= alphabet_size for symbol in word) for word in words):
        return False
    if len(words) != len(set(words)):
        return False
    return all(not (len(a) <= len(b) and b[:len(a)] == a) for i, a in enumerate(words) for j, b in enumerate(words) if i != j)


def expected_length(probabilities, code: Code, alphabet_size: int) -> float:
    probs = _validate_probabilities(probabilities)
    if len(code) != len(probs):
        raise ValueError("one codeword is required for each symbol")
    if not is_prefix_free(code, alphabet_size):
        raise ValueError("code is not prefix-free over the declared alphabet")
    return sum(p * len(codeword) for p, codeword in zip(probs, code))


def aggregate_class_probabilities(class_id, state_probability) -> list[float]:
    labels = list(class_id)
    probs = _validate_probabilities(state_probability)
    if len(labels) != len(probs):
        raise ValueError("partition/probability length mismatch")
    if not labels or min(labels) != 0 or set(labels) != set(range(max(labels) + 1)):
        raise ValueError("class IDs must be consecutive 0..K-1")
    class_probs = [0.0] * (max(labels) + 1)
    for label, p in zip(labels, probs):
        class_probs[label] += p
    return class_probs

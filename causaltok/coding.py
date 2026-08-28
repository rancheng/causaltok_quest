from __future__ import annotations


def is_prefix_free(code: dict[int, str]) -> bool:
    words = list(code.values())
    if any(not w or any(ch not in "01" for ch in w) for w in words):
        return False
    if len(words) != len(set(words)):
        return False
    return all(not b.startswith(a) for i, a in enumerate(words) for j, b in enumerate(words) if i != j)


def expected_length(probs, code: dict[int, str]) -> float:
    if len(probs) == 0:
        raise ValueError("empty probability vector")
    if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-9:
        raise ValueError("invalid probability vector")
    if set(code) != set(range(len(probs))):
        raise ValueError("code must contain exactly class indices 0..K-1")
    if not is_prefix_free(code):
        raise ValueError("code is not prefix-free")
    return sum(p * len(code[i]) for i, p in enumerate(probs))

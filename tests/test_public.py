from causaltok.coding import expected_length, is_prefix_free
from causaltok.public_check import check_partition_public
from causaltok.public_worlds import random_texture_trap, one_bit_contact, delayed_distinction
from causaltok.world import FiniteWorld


def test_world_roundtrip(tmp_path):
    world = one_bit_contact(duplicates=3)
    path = tmp_path / "world.json"
    world.to_json(path)
    restored = FiniteWorld.from_json(path)
    assert restored.transitions == world.transitions
    assert restored.consequences == world.consequences


def test_public_families_construct():
    assert random_texture_trap(duplicates=4).n_states == 12
    assert one_bit_contact(duplicates=5).n_states == 10
    assert delayed_distinction(depth=7).n_states == 16


def test_unique_partition_passes_public_check():
    world = random_texture_trap(duplicates=4)
    labels = list(range(world.n_states))
    report = check_partition_public(world, labels, max_horizon=3)
    assert report.passed
    assert report.n_classes == world.n_states


def test_obvious_bad_merge_is_rejected():
    world = one_bit_contact(duplicates=1)
    report = check_partition_public(world, [0, 0], max_horizon=1)
    assert not report.passed
    assert report.counterexample is not None


def test_prefix_scoring():
    probs = [0.5, 0.25, 0.25]
    code = {0: "0", 1: "10", 2: "11"}
    assert is_prefix_free(code)
    assert expected_length(probs, code) == 1.5

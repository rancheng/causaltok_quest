from causaltok.coding import aggregate_class_probabilities, entropy, expected_length, is_prefix_free
from causaltok.nuisance import FreshNuisanceStream
from causaltok.public_check import check_partition_public, canonicalize_partition, is_canonical_partition
from causaltok.public_worlds import random_texture_trap, one_bit_contact, delayed_distinction, rare_critical_state
from causaltok.stochastic import score_approximate_partition, tiny_stochastic_world
from causaltok.streaming import public_stream
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


def test_canonical_partition_format():
    labels = [9, 9, 4, 4, 9]
    assert canonicalize_partition(labels) == [0, 0, 1, 1, 0]
    assert is_canonical_partition([0, 0, 1, 1, 0])


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


def test_c_ary_prefix_scoring():
    probs = [0.5, 0.25, 0.25]
    binary = [[0], [1, 0], [1, 1]]
    ternary = [[0], [1], [2]]
    assert is_prefix_free(binary, 2)
    assert is_prefix_free(ternary, 3)
    assert expected_length(probs, binary, 2) == 1.5
    assert expected_length(probs, ternary, 3) == 1.0
    assert entropy(probs, 2) > 0


def test_class_probability_aggregation():
    assert aggregate_class_probabilities([0, 0, 1], [0.2, 0.3, 0.5]) == [0.5, 0.5]


def test_fresh_nuisance_changes_without_sensor_change():
    world = random_texture_trap(duplicates=1)
    stream = FreshNuisanceStream(world, seed=0)
    a = stream.observe(0, 256)
    b = stream.observe(0, 256)
    assert a["sensor"] == b["sensor"]
    assert a["metadata"] != b["metadata"]
    assert len(a["metadata"]) == 32


def test_rare_class_probability_is_one_e_minus_six():
    world = rare_critical_state(duplicates=10)
    rare_mass = sum(world.probabilities[-10:])
    assert abs(rare_mass - 1e-6) < 1e-12


def test_approximate_scorer_evaluates_but_does_not_optimize():
    world = tiny_stochastic_world()
    separate = score_approximate_partition(world, [0, 1], beta=1.0)
    merged = score_approximate_partition(world, [0, 0], beta=1.0)
    assert separate["distortion"] == 0.0
    assert merged["rate"] == 0.0
    assert merged["distortion"] > 0.0


def test_public_stream_has_fresh_nuisance_and_one_event():
    steps = list(public_stream(length=8, event_step=5, seed=0))
    assert steps[0].observation["weak_signal"] == 0
    assert steps[5].observation["weak_signal"] == 1
    assert len({s.observation["nuisance"] for s in steps}) == len(steps)

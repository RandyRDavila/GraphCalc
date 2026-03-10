import math

import pytest

from graphcalc.additive_combinatorics.ambient_groups import FiniteAbelianGroup
from graphcalc.additive_combinatorics.dataset_generators import (
    additive_dataset_column_definitions,
    additive_set_to_record,
    generate_additive_set_dataset,
    medium_additive_snapshot,
    small_additive_snapshot,
)
from graphcalc.additive_combinatorics.generators import subgroup_from_generators
from graphcalc.additive_combinatorics.sets import AdditiveSet


def test_column_definitions_contains_expected_keys():
    defs = additive_dataset_column_definitions()

    expected = {
        "ambient_moduli",
        "ambient_rank",
        "ambient_order",
        "elements",
        "cardinality",
        "sumset_size",
        "diffset_size",
        "doubling_constant",
        "difference_constant",
        "tripling_constant",
        "additive_energy",
        "max_sum_representation_count",
        "max_difference_representation_count",
        "stabilizer_size",
        "stabilizer_size_of_sumset",
        "contains_zero",
        "is_symmetric",
        "is_subgroup",
        "is_coset",
        "is_sum_free",
        "is_sidon",
        "has_small_doubling_2",
        "is_periodic",
        "is_aperiodic",
        "construction",
        "construction_parameters",
    }

    assert expected.issubset(defs.keys())


def test_column_definitions_have_type_and_description():
    defs = additive_dataset_column_definitions()

    for meta in defs.values():
        assert "type" in meta
        assert "description" in meta
        assert isinstance(meta["description"], str)
        assert meta["description"]


def test_additive_set_to_record_for_nonempty_example():
    G = FiniteAbelianGroup((7,))
    A = AdditiveSet([(0,), (1,), (3,)], group=G)

    row = additive_set_to_record(
        A,
        construction="manual",
        construction_parameters={"example": True},
    )

    assert row["ambient_moduli"] == (7,)
    assert row["ambient_rank"] == 1
    assert row["ambient_order"] == 7
    assert row["elements"] == ((0,), (1,), (3,))
    assert row["cardinality"] == 3
    assert row["sumset_size"] == 6
    assert row["diffset_size"] == 7
    assert math.isclose(row["doubling_constant"], 2.0)
    assert math.isclose(row["difference_constant"], 7 / 3)
    assert math.isclose(row["tripling_constant"], 7 / 3)
    assert row["additive_energy"] == 15
    assert row["max_sum_representation_count"] == 2
    assert row["max_difference_representation_count"] == 3
    assert row["stabilizer_size"] == 1
    assert row["stabilizer_size_of_sumset"] == 1
    assert row["contains_zero"] is True
    assert row["is_symmetric"] is False
    assert row["is_subgroup"] is False
    assert row["is_coset"] is False
    assert row["is_sum_free"] is False
    assert row["is_sidon"] is True
    assert row["has_small_doubling_2"] is True
    assert row["is_periodic"] is False
    assert row["is_aperiodic"] is True
    assert row["construction"] == "manual"
    assert row["construction_parameters"] == {"example": True}


def test_additive_set_to_record_for_empty_set_uses_none_for_undefined_ratios():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([], group=G)

    row = additive_set_to_record(A)

    assert row["ambient_moduli"] == (5,)
    assert row["ambient_rank"] == 1
    assert row["ambient_order"] == 5
    assert row["elements"] == ()
    assert row["cardinality"] == 0
    assert row["sumset_size"] == 0
    assert row["diffset_size"] == 0
    assert row["doubling_constant"] is None
    assert row["difference_constant"] is None
    assert row["tripling_constant"] is None
    assert row["additive_energy"] == 0
    assert row["max_sum_representation_count"] == 0
    assert row["max_difference_representation_count"] == 0
    assert row["stabilizer_size"] == 5
    assert row["stabilizer_size_of_sumset"] == 5
    assert row["contains_zero"] is False
    assert row["is_symmetric"] is True
    assert row["is_subgroup"] is False
    assert row["is_coset"] is False
    assert row["is_sum_free"] is True
    assert row["is_sidon"] is True
    assert row["has_small_doubling_2"] is True
    assert row["is_periodic"] is True
    assert row["is_aperiodic"] is False
    assert row["construction"] is None
    assert row["construction_parameters"] is None


def test_additive_set_to_record_for_subgroup_example():
    G = FiniteAbelianGroup((6,))
    H = subgroup_from_generators(G, [(2,)])

    row = additive_set_to_record(H)

    assert row["elements"] == ((0,), (2,), (4,))
    assert row["cardinality"] == 3
    assert row["sumset_size"] == 3
    assert row["diffset_size"] == 3
    assert math.isclose(row["doubling_constant"], 1.0)
    assert math.isclose(row["difference_constant"], 1.0)
    assert math.isclose(row["tripling_constant"], 1.0)
    assert row["is_subgroup"] is True
    assert row["is_coset"] is True
    assert row["is_periodic"] is True
    assert row["is_aperiodic"] is False


def test_generate_additive_set_dataset_accepts_plain_sets():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(0,), (1,)], group=G)
    B = AdditiveSet([(2,), (3,)], group=G)

    rows = generate_additive_set_dataset([A, B])

    assert len(rows) == 2
    assert rows[0]["elements"] == A.elements
    assert rows[1]["elements"] == B.elements


def test_generate_additive_set_dataset_accepts_pairs_and_triples():
    G = FiniteAbelianGroup((5,))
    A = AdditiveSet([(0,), (1,)], group=G)
    B = AdditiveSet([(2,), (3,)], group=G)

    rows = generate_additive_set_dataset([
        (A, "first"),
        (B, "second", {"tag": 2}),
    ])

    assert len(rows) == 2
    assert rows[0]["construction"] == "first"
    assert rows[0]["construction_parameters"] is None
    assert rows[1]["construction"] == "second"
    assert rows[1]["construction_parameters"] == {"tag": 2}


def test_generate_additive_set_dataset_rejects_invalid_entry():
    with pytest.raises(TypeError, match="Each dataset entry must be"):
        generate_additive_set_dataset([123])  # type: ignore[list-item]


def test_generate_additive_set_dataset_rejects_tuple_without_additive_set():
    with pytest.raises(TypeError, match="must begin with an AdditiveSet"):
        generate_additive_set_dataset([("not a set", "bad")])  # type: ignore[list-item]


def test_small_snapshot_is_nonempty_and_schema_consistent():
    rows = small_additive_snapshot()

    assert rows
    keys = set(rows[0].keys())
    for row in rows[1:]:
        assert set(row.keys()) == keys


def test_small_snapshot_is_deterministic():
    rows1 = small_additive_snapshot()
    rows2 = small_additive_snapshot()

    assert rows1 == rows2


def test_medium_snapshot_is_nonempty_and_schema_consistent():
    rows = medium_additive_snapshot()

    assert rows
    keys = set(rows[0].keys())
    for row in rows[1:]:
        assert set(row.keys()) == keys


def test_medium_snapshot_is_deterministic():
    rows1 = medium_additive_snapshot()
    rows2 = medium_additive_snapshot()

    assert rows1 == rows2


def test_medium_snapshot_is_larger_than_small_snapshot():
    small_rows = small_additive_snapshot()
    medium_rows = medium_additive_snapshot()

    assert len(medium_rows) > len(small_rows)

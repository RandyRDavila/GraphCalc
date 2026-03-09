import pytest

from graphcalc.hypergraphs.core.basics import Hypergraph
from graphcalc.hypergraphs.invariants.transversals import (
    fractional_transversal_number,
    minimum_transversal,
    transversal_number,
)


def test_minimum_transversal_empty_hypergraph():
    H = Hypergraph()
    assert minimum_transversal(H) == set()
    assert transversal_number(H) == 0


def test_minimum_transversal_on_simple_example():
    H = Hypergraph(E=[{1, 2}, {2, 3}])
    T = minimum_transversal(H)
    assert len(T) == 1
    assert T == {2}
    assert transversal_number(H) == 1


def test_minimum_transversal_with_weights():
    H = Hypergraph(E=[{1, 2}, {2, 3}])
    weights = {1: 1.0, 2: 10.0, 3: 1.0}
    T = minimum_transversal(H, weights=weights)
    assert T == {1, 3}
    assert transversal_number(H, weights=weights) == pytest.approx(2.0)


def test_transversal_raises_on_empty_edge():
    H = Hypergraph(allow_empty=True, allow_singletons=True, E=[[], [1]])
    with pytest.raises(ValueError, match="no transversal exists"):
        minimum_transversal(H)
    with pytest.raises(ValueError, match="no fractional transversal exists"):
        fractional_transversal_number(H)


def test_fractional_transversal_number_simple():
    H = Hypergraph(E=[{1, 2}, {2, 3}])
    val = fractional_transversal_number(H)
    assert val == pytest.approx(1.0)


def test_fractional_transversal_number_empty_hypergraph():
    H = Hypergraph()
    assert fractional_transversal_number(H) == 0.0

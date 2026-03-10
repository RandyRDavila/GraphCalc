import pytest

from graphcalc.hypergraphs.core.basics import Hypergraph
from graphcalc.hypergraphs.invariants.independence import (
    independence_number,
    maximum_independent_set,
)


def test_maximum_independent_set_no_edges():
    H = Hypergraph(V=[1, 2, 3], allow_singletons=True)
    I = maximum_independent_set(H)
    assert I == {1, 2, 3}
    assert independence_number(H) == 3


def test_maximum_independent_set_graph_like_example():
    H = Hypergraph(E=[{1, 2}, {2, 3}])
    I = maximum_independent_set(H)
    assert len(I) == 2
    assert I in ({1, 3},)


def test_independence_number_graph_like_example():
    H = Hypergraph(E=[{1, 2}, {2, 3}])
    assert independence_number(H) == 2


def test_independence_raises_on_empty_edge():
    H = Hypergraph(allow_empty=True, E=[[]])
    with pytest.raises(ValueError, match="independence set is undefined"):
        maximum_independent_set(H)

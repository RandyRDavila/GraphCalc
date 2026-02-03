from __future__ import annotations
from typing import Iterable, Mapping, Hashable, Optional, Tuple, List, Dict, Sequence, Union
import matplotlib.pyplot as plt
import networkx as nx
from graphcalc import GraphLike


__all__ = [
    "draw_vertex_set",
    "draw_vertices",
]

def _ensure_nx(G: GraphLike) -> nx.Graph:
    if isinstance(G, nx.Graph):
        return G
    raise TypeError("G must be a NetworkX Graph (or subclass).")

def _default_palette(k: int) -> List[str]:
    base = [f"C{i}" for i in range(20)]
    if k <= len(base):
        return base[:k]
    out = []
    while len(out) < k:
        out.extend(base)
    return out[:k]

def draw_vertex_set(
    G: GraphLike,
    subset: Iterable[Hashable],
    *,
    layout: str = "spring",
    pos: Optional[dict] = None,
    seed: int = 42,
    highlight_size: int = 500,
    base_size: int = 200,
    highlight_color: str = "tab:red",
    other_color: str = "lightgray",
    with_labels: bool = True,
    title: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
    legend: bool = True,
) -> Tuple["plt.Figure", "plt.Axes"]:
    r"""
    Visualize a distinguished subset of vertices in a graph.

    This function highlights a specified subset of vertices (e.g., an independent set
    or dominating set) in a given graph :math:`G`. Highlighted nodes are drawn larger
    and in a distinct color, while all other nodes are rendered with a baseline size
    and muted color. Edges are drawn uniformly.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    subset : iterable of hashable
        A collection of nodes in :math:`G` to highlight.
    layout : {"spring", "kamada_kawai", "planar", "circular", "spectral"}, optional
        Layout algorithm used to compute node positions. Ignored if `pos` is given
        (default is "spring").
    pos : dict, optional
        Precomputed node positions as a dictionary mapping nodes to coordinate pairs.
        If provided, `layout` is ignored.
    seed : int, optional
        Random seed used by layout algorithms that are stochastic (default is 42).
    highlight_size : int, optional
        Size of nodes in `subset` (default is 500).
    base_size : int, optional
        Size of nodes not in `subset` (default is 200).
    highlight_color : str, optional
        Color used for nodes in `subset` (default is "tab:red").
    other_color : str, optional
        Color used for other nodes (default is "lightgray").
    with_labels : bool, optional
        Whether to display node labels (default is True).
    title : str, optional
        Title of the plot. If None, no title is shown (default is None).
    ax : matplotlib.axes.Axes, optional
        Axes object to draw on. If None, a new figure and axes are created (default is None).
    legend : bool, optional
        Whether to include a legend distinguishing highlighted vs. other nodes
        (default is True).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object where the graph is drawn.

    Examples
    --------
    Highlight a maximum independent set:

    >>> import graphcalc as gc
    >>> from graphcalc.viz import draw_vertex_set
    >>> G = gc.cycle_graph(6)
    >>> S = gc.maximum_independent_set(G)
    >>> fig, ax = draw_vertex_set(G, S,
    ...                 highlight_color="tab:blue",
    ...                 other_color="lightgray",
    ...                 title="Maximum Independent Set")
    """

    Gnx = _ensure_nx(G)
    S = set(subset)

    if pos is None:
        pos = nx.spring_layout(Gnx, seed=seed)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    else:
        fig = ax.figure

    # edges
    nx.draw_networkx_edges(Gnx, pos, ax=ax, width=1.5, alpha=0.7)

    # others
    others = [v for v in Gnx.nodes if v not in S]
    nx.draw_networkx_nodes(
        Gnx, pos, nodelist=others, node_size=base_size,
        node_color=other_color, ax=ax, alpha=0.9
    )

    # highlights
    nx.draw_networkx_nodes(
        Gnx, pos, nodelist=S, node_size=highlight_size,
        node_color=highlight_color, ax=ax
    )

    if with_labels:
        nx.draw_networkx_labels(Gnx, pos, ax=ax, font_size=9)

    ax.set_axis_off()
    if title: ax.set_title(title)
    return fig, ax

# ---------- Normalizers ----------
def _normalize_vertex_classes(
    G: GraphLike,
    classes: Union[Iterable[Hashable], Mapping[int, Iterable[Hashable]], Sequence[Iterable[Hashable]]],
) -> List[List[Hashable]]:
    """Return a list of disjoint vertex lists covering ONLY provided vertices."""
    if isinstance(classes, Mapping):
        parts = [list(vs) for _, vs in sorted(classes.items(), key=lambda kv: kv[0])]
    elif isinstance(classes, Iterable) and classes and not isinstance(next(iter(classes)), (list, set, tuple)):
        # Single target set -> 2-coloring: target vs complement
        S = set(classes)
        parts = [list(S), [v for v in G.nodes if v not in S]]
    else:
        # Sequence of iterables (k-coloring)
        parts = [list(vs) for vs in classes]  # type: ignore[arg-type]
    # keep only nodes that exist
    return [[v for v in part if v in G] for part in parts if part]


# ---------- Public APIs ----------
def draw_vertices(
    G: GraphLike,
    classes: Union[
        Iterable[Hashable],                        # target set -> auto 2-coloring
        Mapping[int, Iterable[Hashable]],          # dict color_index -> nodes
        Sequence[Iterable[Hashable]],              # list/tuple of node sets
    ],
    *,
    pos: Optional[dict] = None,
    layout: str = "spring",
    seed: int = 42,
    palette: Optional[List[str]] = None,
    node_size: int = 420,
    edge_color: str = "lightgray",
    with_labels: bool = True,
    title: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
    show_legend: bool = True,
    legend_labels: Optional[List[str]] = None,
) -> Tuple["plt.Figure", "plt.Axes"]:
    r"""
    Visualize vertex partitions or highlighted subsets in a graph.

    This function generalizes vertex visualization by supporting several input formats:
    a single subset (rendered as a 2-coloring), a dictionary encoding a k-coloring,
    or multiple disjoint subsets. Each class of vertices is drawn in a distinct color
    using either a provided palette or a default matplotlib cycle.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    classes : iterable, mapping, or sequence
        Specification of vertex classes to highlight. Supported formats include:

        - ``iterable of nodes``: A single target subset, automatically complemented
          by the rest of the vertices (2-coloring).
        - ``dict {color_index: iterable of nodes}``: A k-coloring where each key
          corresponds to a color class.
        - ``sequence of iterables``: Multiple vertex subsets to display, assigned
          colors sequentially.

    pos : dict, optional
        Precomputed node positions as a dictionary mapping nodes to coordinate pairs.
        If None, positions are computed using the algorithm specified by `layout`.
    layout : {"spring", "kamada_kawai", "planar", "circular", "spectral"}, optional
        Layout algorithm used to compute node positions (default is "spring").
    seed : int, optional
        Random seed for layout algorithms that are stochastic (default is 42).
    palette : list of str, optional
        List of colors to cycle through for different vertex classes. If None, a
        default matplotlib color cycle is used.
    node_size : int, optional
        Size of nodes (default is 420).
    edge_color : str, optional
        Color of edges (default is "lightgray").
    with_labels : bool, optional
        Whether to display node labels (default is True).
    title : str, optional
        Title of the plot (default is None).
    ax : matplotlib.axes.Axes, optional
        Axes object to draw on. If None, a new figure and axes are created (default is None).
    show_legend : bool, optional
        Whether to include a legend distinguishing vertex classes (default is True).
    legend_labels : list of str, optional
        Custom labels for the legend. If None, labels are generated automatically
        as ``"class i"``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object where the graph is drawn.

    Examples
    --------
    Highlight a maximum independent set as a 2-coloring:

    >>> import graphcalc as gc
    >>> from graphcalc.viz import draw_vertices
    >>> G = gc.cycle_graph(6)
    >>> S = gc.maximum_independent_set(G)
    >>> _ = draw_vertices(G, S,
    ...               palette=["tab:blue", "lightgray"],
    ...               legend_labels=["independent set", "other vertices"],
    ...               title="Independent Set as 2-coloring")

    Visualize an optimal proper coloring:

    >>> coloring = gc.optimal_proper_coloring(G)
    >>> _ = draw_vertices(G, coloring, title="Optimal Proper Coloring")
    """

    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a NetworkX Graph (or subclass).")

    parts = _normalize_vertex_classes(G, classes)
    k = len(parts)
    if palette is None:
        palette = _default_palette(k)

    # positions
    if pos is None:
        if layout == "spring":       pos = nx.spring_layout(G, seed=seed)
        elif layout == "kamada_kawai": pos = nx.kamada_kawai_layout(G)
        elif layout == "planar":     pos = nx.planar_layout(G)
        elif layout == "circular":   pos = nx.circular_layout(G)
        elif layout == "spectral":   pos = nx.spectral_layout(G)
        else: raise ValueError(f"Unknown layout '{layout}'")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=150)
    else:
        fig = ax.figure

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, width=1.4, alpha=0.85)

    handles = []
    for i, nodes in enumerate(parts):
        color = palette[i % len(palette)]
        coll = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=node_size, node_color=color, ax=ax)
        coll.set_edgecolor("k")
        coll.set_linewidth(0.5)
        handles.append(coll)

    if with_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)

    ax.set_axis_off()
    if title: ax.set_title(title)

    if show_legend and handles:
        from matplotlib.lines import Line2D
        labels = legend_labels or [f"class {i}" for i in range(k)]
        legend_handles = [
            Line2D([0],[0], marker="o", linestyle="None", markersize=8,
                   markerfacecolor=h.get_facecolor()[0], markeredgecolor="k", label=lab)
            for h, lab in zip(handles, labels)
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    fig.tight_layout()
    return fig, ax

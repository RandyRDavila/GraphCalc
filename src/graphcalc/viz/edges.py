from __future__ import annotations
from typing import Iterable, Mapping, Hashable, Optional, Tuple, List, Dict, Sequence, Union
import matplotlib.pyplot as plt
import networkx as nx
from graphcalc import GraphLike

__all__ = [
    "draw_edge_set",
    "draw_coloring",
    "draw_edges",
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

def draw_edge_set(
    G: GraphLike,
    edge_set: Iterable[Tuple[Hashable, Hashable]],
    *,
    layout: str = "spring",
    pos: Optional[dict] = None,
    seed: int = 42,
    width_selected: float = 2.8,
    width_other: float = 1.2,
    edge_color_selected: str = "tab:red",
    edge_color_other: str = "lightgray",
    node_color: str = "lightblue",
    with_labels: bool = True,
    title: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
    legend: bool = True,
) -> Tuple["plt.Figure", "plt.Axes"]:
    r"""
    Visualize a distinguished set of edges in a graph.

    This function highlights a subset of edges (e.g., a matching or cut) in a given
    graph :math:`G`. Selected edges are emphasized with distinct color and width,
    while the remaining edges are drawn in a muted style. Nodes are drawn uniformly.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    edge_set : iterable of tuple
        A collection of edges to highlight. Each edge must be a 2-tuple of hashable
        nodes present in :math:`G`.
    layout : {"spring", "kamada_kawai", "planar", "circular", "spectral"}, optional
        Layout algorithm used to compute node positions. Ignored if `pos` is given
        (default is "spring").
    pos : dict, optional
        Precomputed node positions as a dictionary mapping nodes to coordinate pairs.
        If provided, `layout` is ignored.
    seed : int, optional
        Random seed used by layout algorithms that are stochastic (default is 42).
    width_selected : float, optional
        Line width for edges in `edge_set` (default is 2.8).
    width_other : float, optional
        Line width for non-selected edges (default is 1.2).
    edge_color_selected : str, optional
        Color for edges in `edge_set` (default is "tab:red").
    edge_color_other : str, optional
        Color for non-selected edges (default is "lightgray").
    node_color : str or list, optional
        Color(s) used for nodes (default is "lightblue").
    with_labels : bool, optional
        Whether to display node labels (default is True).
    title : str, optional
        Title of the plot. If None, no title is shown (default is None).
    ax : matplotlib.axes.Axes, optional
        Axes object to draw on. If None, a new figure and axes are created (default is None).
    legend : bool, optional
        Whether to include a legend distinguishing selected and non-selected edges
        (default is True).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object where the graph is drawn.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.viz import draw_edge_set
    >>> G = gc.diamond_necklace(4)
    >>> M = gc.maximum_matching(G)
    >>> fig, ax = draw_edge_set(G, M,
    ...               edge_color_selected="tab:blue",
    ...               edge_color_other="gray",
    ...               node_color="lightyellow",
    ...               title="Maximum Matching")
    """

    Gnx = _ensure_nx(G)
    E_sel_set = {frozenset(e) for e in edge_set}
    E_sel = [e for e in Gnx.edges if frozenset(e) in E_sel_set]
    E_oth = [e for e in Gnx.edges if frozenset(e) not in E_sel_set]

    if pos is None:
        pos = nx.spring_layout(Gnx, seed=seed)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    else:
        fig = ax.figure

    if E_oth:
        nx.draw_networkx_edges(Gnx, pos, edgelist=E_oth, ax=ax,
                               width=width_other, edge_color=edge_color_other)
    if E_sel:
        nx.draw_networkx_edges(Gnx, pos, edgelist=E_sel, ax=ax,
                               width=width_selected, edge_color=edge_color_selected)

    nx.draw_networkx_nodes(Gnx, pos, ax=ax, node_size=220, node_color=node_color, alpha=0.9)

    if with_labels:
        nx.draw_networkx_labels(Gnx, pos, ax=ax, font_size=9)

    ax.set_axis_off()
    if title: ax.set_title(title)
    return fig, ax

def draw_coloring(
    G: nx.Graph,
    coloring: Dict[int, Iterable[Hashable]],
    *,
    layout: str = "spring",
    pos: Optional[dict] = None,
    seed: int = 42,
    palette: Optional[List[str]] = None,
    node_size: int = 420,
    edge_color: str = "lightgray",
    with_labels: bool = True,
    title: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
    show_legend: bool = True,
) -> Tuple["plt.Figure", "plt.Axes"]:
    r"""
    Visualize a proper coloring returned by ``optimal_proper_coloring``.

    Parameters
    ----------
    G : networkx.Graph or SimpleGraph
        The graph to visualize.
    coloring : dict[int, iterable]
        Mapping from color index to list of nodes assigned that color.
    layout : {"spring","kamada_kawai","planar","circular","spectral"}, optional
        Layout algorithm (default "spring").
    pos : dict, optional
        Node positions; if None, computed by the chosen layout.
    seed : int, optional
        Random seed for layouts with randomness (default 42).
    palette : list of str, optional
        List of color strings for classes. If None, uses a default cycle
        (e.g., Matplotlib's tab10 extended).
    node_size : int, optional
        Size of nodes (default 420).
    edge_color : str, optional
        Color for edges (default "lightgray").
    with_labels : bool, optional
        Whether to draw node labels (default True).
    title : str, optional
        Figure title.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, a new figure is created.
    show_legend : bool, optional
        Whether to draw a legend with color class labels (default True).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes
        The axes.

    Notes
    -----
    - Ignores empty color classes.
    - If there are more classes than palette entries, colors repeat.

    Examples
    --------
    >>> import graphcalc as gc
    >>> from graphcalc.viz import draw_coloring
    >>> G = gc.cycle_graph(6)
    >>> C = gc.optimal_proper_coloring(G)
    >>> fig, ax = draw_coloring(G, C, title=f"χ(G) = {len([v for v in C.values() if v])}")
    """
    # Normalize: drop empty classes, sort by color id
    classes = [(c, [v for v in nodes]) for c, nodes in sorted(coloring.items()) if nodes]
    k = len(classes)

    if palette is None:
        # Use a long, pleasant palette (tab10 + tab20 fallback)
        tab10 = [f"C{i}" for i in range(10)]
        tab20 = [f"C{i}" for i in range(20)]
        palette = (tab10 + tab20)[0:max(1, k)]

    # Compute / reuse positions
    if pos is None:
        if layout == "spring":
            pos = nx.spring_layout(G, seed=seed)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "planar":
            pos = nx.planar_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        else:
            raise ValueError(f"Unknown layout '{layout}'")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=150)
    else:
        fig = ax.figure

    # Edges first
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color, width=1.4, alpha=0.85)

    # Draw each color class
    handles = []
    for i, (c, nodes) in enumerate(classes):
        color = palette[i % len(palette)]
        coll = nx.draw_networkx_nodes(
            G, pos, nodelist=nodes, node_size=node_size, node_color=color, ax=ax
        )
        coll.set_edgecolor("k")
        coll.set_linewidth(0.5)
        handles.append((c, coll))

    if with_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)

    ax.set_axis_off()
    if title:
        ax.set_title(title)

    if show_legend and handles:
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="None", markersize=8,
                   markerfacecolor=h.get_facecolor()[0], markeredgecolor="k",
                   label=f"color {c}")
            for c, h in handles
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    fig.tight_layout()
    return fig, ax

def _as_edge_list_from_pairmap(pairmap: Mapping[Hashable, Hashable]) -> List[Tuple[Hashable, Hashable]]:
    """Convert a symmetric node->partner dict into an undirected edge list (u<v style)."""
    seen = set()
    edges = []
    for u, v in pairmap.items():
        if v is None:
            continue
        a, b = (u, v)
        key = frozenset((a, b))
        if key in seen:  # skip the reverse duplicate
            continue
        seen.add(key)
        edges.append((a, b))
    return edges

def _sanitize_edge_tuples(G: GraphLike, elist: Iterable[Tuple[Hashable, Hashable]]) -> List[Tuple[Hashable, Hashable]]:
    """Keep only real edges of G, ignoring orientation."""
    Eset = {frozenset(e) for e in G.edges()}
    out = []
    for e in elist:
        if not (isinstance(e, tuple) and len(e) == 2):
            continue
        a, b = e
        if frozenset((a, b)) in Eset:
            out.append((a, b))
    return out

def _normalize_edge_classes(
    G: GraphLike,
    classes: Union[
        Iterable[Tuple[Hashable, Hashable]],                   # a single target set
        Mapping[int, Iterable[Tuple[Hashable, Hashable]]],     # k-edge-coloring dict
        Sequence[Iterable[Tuple[Hashable, Hashable]]],         # list of edge sets
        Mapping[Hashable, Hashable],                            # node->partner dict (matching)
        nx.classes.reportviews.EdgeView,
    ],
) -> List[List[Tuple[Hashable, Hashable]]]:
    """Return list of edge lists. If input is a single set/dict -> [target, complement]."""
    # Case A: node->partner dict (e.g., nx.maximum_matching)
    if isinstance(classes, Mapping) and classes and not all(isinstance(k, int) for k in classes.keys()):
        target = _sanitize_edge_tuples(G, _as_edge_list_from_pairmap(classes))  # target edges
        selected = {frozenset(e) for e in target}
        complement = [e for e in G.edges() if frozenset(e) not in selected]
        return [target, complement]

    # Case B: dict color_index -> elist
    if isinstance(classes, Mapping):
        parts = []
        for _, elist in sorted(classes.items(), key=lambda kv: kv[0]):
            parts.append(_sanitize_edge_tuples(G, elist))
        return [p for p in parts if p]

    # Case C: EdgeView
    if isinstance(classes, nx.classes.reportviews.EdgeView):
        target = _sanitize_edge_tuples(G, list(classes))
        selected = {frozenset(e) for e in target}
        complement = [e for e in G.edges() if frozenset(e) not in selected]
        return [target, complement]

    # Case D: iterable of 2-tuples (single class) → auto 2-coloring
    try:
        sample = next(iter(classes))  # type: ignore[arg-type]
    except StopIteration:
        return [[]]  # empty
    except TypeError:
        # not iterable? give up gracefully
        return [[]]

    if isinstance(sample, tuple) and len(sample) == 2:
        target = _sanitize_edge_tuples(G, classes)  # type: ignore[arg-type]
        selected = {frozenset(e) for e in target}
        complement = [e for e in G.edges() if frozenset(e) not in selected]
        return [target, complement]

    # Case E: sequence of elists
    parts = []
    for elist in classes:  # type: ignore[assignment]
        parts.append(_sanitize_edge_tuples(G, elist))
    return [p for p in parts if p]

def draw_edges(
    G: GraphLike,
    classes: Union[
        Iterable[Tuple[Hashable, Hashable]],
        Mapping[int, Iterable[Tuple[Hashable, Hashable]]],
        Sequence[Iterable[Tuple[Hashable, Hashable]]],
        Mapping[Hashable, Hashable],
        nx.classes.reportviews.EdgeView,
    ],
    *,
    pos: Optional[dict] = None,
    layout: str = "spring",
    seed: int = 42,
    palette: Optional[List[str]] = None,
    width: float = 2.8,
    node_color: str = "white",
    with_labels: bool = True,
    title: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
    show_legend: bool = True,
    legend_labels: Optional[List[str]] = None,
):
    r"""
    Visualize edge partitions or highlighted sets in a graph.

    This function generalizes edge visualization by supporting several input formats:
    a single edge set, a node-to-partner mapping (matchings), a dictionary encoding
    a k-edge-coloring, or multiple disjoint edge sets. Selected classes of edges are
    drawn with distinct colors, while all nodes are displayed uniformly.

    Parameters
    ----------
    G : networkx.Graph or graphcalc.SimpleGraph
        An undirected simple graph.
    classes : iterable, mapping, or EdgeView
        Specification of edge classes to highlight. Supported formats include:

        - ``iterable of 2-tuples``: A single edge set, automatically complemented
          by the rest of the edges.
        - ``dict {node: partner}``: A matching, where edges are inferred from the
          mapping.
        - ``dict {color_index: iterable of edges}``: A k-edge-coloring.
        - ``sequence of iterables``: Multiple edge sets to display.
        - ``EdgeView``: A subset of edges returned by ``G.edges(data=False)``.

    pos : dict, optional
        Precomputed node positions as a dictionary mapping nodes to coordinate pairs.
        If None, positions are computed using the algorithm specified by `layout`.
    layout : {"spring", "kamada_kawai", "planar", "circular", "spectral"}, optional
        Layout algorithm used to compute node positions (default is "spring").
    seed : int, optional
        Random seed for layout algorithms that are stochastic (default is 42).
    palette : list of str, optional
        List of colors to cycle through for different edge classes. If None, a
        default matplotlib color cycle is used.
    width : float, optional
        Line width for drawn edges (default is 2.8).
    node_color : str or list, optional
        Color(s) used for nodes (default is "white").
    with_labels : bool, optional
        Whether to display node labels (default is True).
    title : str, optional
        Title of the plot (default is None).
    ax : matplotlib.axes.Axes, optional
        Axes object to draw on. If None, a new figure and axes are created (default is None).
    show_legend : bool, optional
        Whether to include a legend distinguishing edge classes (default is True).
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
    Highlight a maximum matching:

    >>> import graphcalc as gc
    >>> from graphcalc.viz import draw_edges
    >>> G = gc.diamond_necklace(5)
    >>> M = gc.maximum_matching(G)
    >>> _ = draw_edges(G, M,
    ...            palette=["tab:blue", "lightgray"],
    ...            legend_labels=["matching", "other edges"],
    ...            node_color="lightyellow",
    ...            title="Maximum Matching")

    Visualize a 3-edge-coloring:

    >>> coloring = {
    ...     0: [(0, 1), (2, 3)],
    ...     1: [(1, 2)],
    ...     2: [(3, 0)]
    ... }
    >>> _ = draw_edges(G, coloring, palette=["red", "green", "blue"], title="3-edge-coloring")
    """

    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a NetworkX Graph (or subclass).")

    parts = _normalize_edge_classes(G, classes)
    k = len(parts)
    if palette is None:
        palette = _default_palette(k)

    # positions
    if pos is None:
        if layout == "spring":
            pos = nx.spring_layout(G, seed=seed)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "planar":
            pos = nx.planar_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        else:
            raise ValueError(f"Unknown layout '{layout}'")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=150)
    else:
        fig = ax.figure

    # draw nodes first
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=220, node_color=node_color,
        edgecolors="k", linewidths=0.5, alpha=0.95
    )

    # draw edge classes
    handles = []
    for i, elist in enumerate(parts):
        if not elist:
            continue
        color = palette[i % len(palette)]
        coll = nx.draw_networkx_edges(G, pos, edgelist=elist, ax=ax, width=width, edge_color=color)
        handles.append((color, i))

    if with_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)

    ax.set_axis_off()
    if title:
        ax.set_title(title)

    if show_legend and handles:
        from matplotlib.lines import Line2D
        labels = legend_labels or [f"class {i}" for _, i in handles]
        legend_handles = [
            Line2D([0], [0], color=palette[i % len(palette)], lw=width, label=lab)
            for (_, i), lab in zip(handles, labels)
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    fig.tight_layout()
    return fig, ax

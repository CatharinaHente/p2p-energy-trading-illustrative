"""Trading graph construction for the P2P market.

Trading edges connect prosumer indices, not physical bus indices.

Supported topologies:
    cyclic — ring graph
    knn    — k-nearest neighbours by radial feeder distance
"""

import networkx as nx

from config import TRADING_TOPOLOGY, K_NEIGHBOURS
from network import load_network


def cyclic_edges(n):
    """Construct a cyclic trading graph."""

    if n < 3:
        raise ValueError("Cyclic graph requires at least 3 prosumers")

    return [(i, i + 1) for i in range(n - 1)] + [(0, n - 1)]


def knn_edges(agent_bus, k=3):
    """Construct a connected, undirected kNN trading graph.

    Distance is the cumulative physical line length along
    the radial feeder between two prosumer buses.
    """

    n = len(agent_bus)

    if not 1 <= k < n:
        raise ValueError(f"Invalid k={k} for N={n}")

    grid = load_network()

    # ── Physical feeder graph ──────────────────────────

    feeder = nx.Graph()

    for (u, v), length in zip(
        grid['branches'], grid['length_km']
    ):
        feeder.add_edge(u, v, weight=float(length))

    if not nx.is_tree(feeder):
        raise ValueError("Physical feeder must be a radial tree")

    # ── Pairwise feeder distances ─────────────────────

    distances = {}

    for i, bus in enumerate(agent_bus):

        lengths = nx.single_source_dijkstra_path_length(
            feeder, bus, weight='weight'
        )

        for j, other_bus in enumerate(agent_bus):
            if i != j:
                distances[i, j] = lengths[other_bus]

    # ── Select k nearest neighbours ───────────────────

    trading = nx.Graph()
    trading.add_nodes_from(range(n))

    for i in range(n):

        neighbours = sorted(
            (j for j in range(n) if j != i),
            key=lambda j: (distances[i, j], j)
        )[:k]

        for j in neighbours:
            trading.add_edge(i, j)

    # ── Ensure global connectivity ───────────────────

    while not nx.is_connected(trading):

        components = [
            sorted(c)
            for c in nx.connected_components(trading)
        ]

        _, i, j = min(
            (distances[i, j], i, j)
            for a, component_a in enumerate(components)
            for component_b in components[a + 1:]
            for i in component_a
            for j in component_b
        )

        trading.add_edge(i, j)

    return sorted(
        (min(i, j), max(i, j))
        for i, j in trading.edges()
    )


def make_trading_edges(agent_bus):
    """Select the configured trading topology."""

    if TRADING_TOPOLOGY == 'cyclic':
        return cyclic_edges(len(agent_bus))

    if TRADING_TOPOLOGY == 'knn':
        return knn_edges(agent_bus, k=K_NEIGHBOURS)

    raise ValueError(
        f"Unknown trading topology: {TRADING_TOPOLOGY}"
    )

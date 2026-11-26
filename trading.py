"""Trading graph construction for the P2P market.

Trading edges connect prosumer indices, not physical bus indices.

Supported topologies (config.TRADING_TOPOLOGY):
    cyclic        — ring graph over prosumer index order
    knn           — k-nearest neighbours by radial feeder distance
    sparse_hetero — sparse connected random graph with heterogeneous degrees
"""
import random

import networkx as nx

from config import (TRADING_TOPOLOGY, K_NEIGHBOURS,
                    SPARSE_MEAN_DEGREE, TRADING_SEED)
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


def sparse_hetero_edges(n, mean_degree=2.5, seed=42):
    """Sparse, connected trading graph with heterogeneous degrees.

    1. Preferential-attachment tree (Barabási–Albert, m = 1): every prosumer
       joins by linking to an existing one with probability ∝ degree, which
       gives a few well-connected hubs and many single-link prosumers.
    2. Extra edges up to round(mean_degree · n / 2), again chosen with
       probability ∝ degree product, so the hubs stay hubs.

    The result is connected, has mean degree ≈ mean_degree (capped by the
    complete graph) and is fully determined by (n, mean_degree, seed). It does
    not depend on location; the trading graph is a social structure here.
    """
    if n < 2:
        raise ValueError("Need at least 2 prosumers")
    rng = random.Random(f"sparse_hetero-{n}-{seed}")
    order = list(range(n))
    rng.shuffle(order)                  # hub identity is random, not index 0
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edge(order[0], order[1])
    for new in order[2:]:
        nodes = [v for v in graph.nodes if graph.degree(v) > 0]
        weights = [graph.degree(v) for v in nodes]
        graph.add_edge(new, rng.choices(nodes, weights=weights)[0])

    target = min(round(mean_degree * n / 2), n * (n - 1) // 2)
    while graph.number_of_edges() < target:
        missing = [(i, j) for i in range(n) for j in range(i + 1, n)
                   if not graph.has_edge(i, j)]
        weights = [graph.degree(i) * graph.degree(j) for i, j in missing]
        graph.add_edge(*rng.choices(missing, weights=weights)[0])

    return sorted((min(i, j), max(i, j)) for i, j in graph.edges())


def make_trading_edges(agent_bus):
    """Select the configured trading topology."""

    if TRADING_TOPOLOGY == 'cyclic':
        return cyclic_edges(len(agent_bus))

    if TRADING_TOPOLOGY == 'knn':
        return knn_edges(agent_bus, k=K_NEIGHBOURS)

    if TRADING_TOPOLOGY == 'sparse_hetero':
        return sparse_hetero_edges(len(agent_bus), SPARSE_MEAN_DEGREE,
                                   TRADING_SEED)

    raise ValueError(
        f"Unknown trading topology: {TRADING_TOPOLOGY}"
    )

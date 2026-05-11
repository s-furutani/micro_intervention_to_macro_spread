"""グラフロードを 1 か所に集約するモジュール。

旧 `CTIC.py` / `intervention_analysis.py` /
`intervention_analysis_varying_eta.py` / `single_vs_combined.py` の
4 重複していた `load_graph_by_name()` をここに統合する。

- `graph.graph['graph_name']` を必ず付与（Nikolov 系の付与漏れを統一）。
- test グラフは有向化を行う（旧 intervention_analysis 系と同じ挙動）。
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np

from load_graph import (
    Nikolov_susceptibility_graph,
    randomized_nikolov_graph,
    uniform_nikolov_graph,
)

VALID_GRAPHS: tuple[str, ...] = (
    "test",
    "nikolov",
    "randomized_nikolov",
    "uniform_nikolov",
)


def _make_test_graph(seed: int = 42) -> nx.DiGraph:
    """erdos_renyi の test グラフ（有向化、suscep を一様乱数で付与）。

    旧 intervention_analysis 系の挙動を保つため、suscep の付与には
    `np.random` グローバル RNG を使用している。
    """
    G = nx.erdos_renyi_graph(200, 0.1, seed=seed)
    G = nx.to_directed(G)
    G.graph["graph_name"] = "test"
    for node in G.nodes():
        G.nodes[node]["suscep"] = np.random.uniform(0.0, 1.0)
    return G


def load_graph_by_name(graph_name: str, **kwargs) -> nx.DiGraph:
    """名前からグラフを返す。`graph.graph['graph_name']` を必ず付与する。"""
    if graph_name == "test":
        return _make_test_graph(**kwargs)
    if graph_name == "nikolov":
        G = Nikolov_susceptibility_graph()
        G.graph["graph_name"] = "nikolov"
        return G
    if graph_name == "randomized_nikolov":
        G = randomized_nikolov_graph()
        G.graph["graph_name"] = "randomized_nikolov"
        return G
    if graph_name == "uniform_nikolov":
        G = uniform_nikolov_graph()
        G.graph["graph_name"] = "uniform_nikolov"
        return G
    raise ValueError(
        f"Unknown graph name: {graph_name}. Available options: {VALID_GRAPHS}"
    )

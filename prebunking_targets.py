"""Prebunking のターゲットノード選定を 1 関数に集約するモジュール。

旧:
- `CTIC.py` の `run_continuous_time_independent_cascade` 内ブロック
- `quenched_mean_field.py` の `prebunking_targets`

の 2 実装を統合する。乱数を使う戦略（random / cocoon / ランダム埋め）は
渡された `random` モジュールに準拠したシャッフルを行うため、CTIC 側で
`random.seed()` してから呼び出す呼び出し方を維持できる。
"""

from __future__ import annotations

import random as _random_mod
from typing import Any, Iterable, Mapping

import networkx as nx

VALID_STRATEGIES: tuple[str, ...] = (
    "random",
    "high_degree",
    "high_susceptible",
    "cocoon",
)


def select_prebunking_targets(
    graph: nx.DiGraph,
    *,
    k: int,
    strategy: str,
    seed_nodes: Iterable[Any],
    susceptibility: Mapping[Any, float],
    inactive_nodes: set | None = None,
    rng: _random_mod.Random | None = None,
) -> list:
    """prebunking 対象ノードを最大 k 個返す。

    Args:
        graph: 有向グラフ（out_degree, successors を使う）。
        k: 選定したいノード数（1 以上）。
        strategy: 'random' / 'high_degree' / 'high_susceptible' / 'cocoon'。
        seed_nodes: 既に拡散している初期ノード集合。
        susceptibility: ノード -> susceptibility の辞書。
        inactive_nodes: 候補となる未活性ノード集合。None なら graph - seed_nodes。
        rng: シャッフル用の `random.Random` インスタンス。
            None なら `random` グローバル状態を使う（旧 CTIC.py と同じ挙動）。

    Returns:
        対象ノードのリスト（最大 k 個）。順序は戦略に依存。
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Use one of {VALID_STRATEGIES}"
        )
    if k < 1:
        return []
    if inactive_nodes is None:
        inactive_nodes = set(graph.nodes()) - set(seed_nodes)

    shuffle = rng.shuffle if rng is not None else _random_mod.shuffle

    if strategy == "high_degree":
        cand = sorted(
            inactive_nodes, key=lambda v: graph.out_degree(v), reverse=True
        )
    elif strategy == "high_susceptible":
        cand = sorted(inactive_nodes, key=lambda v: susceptibility[v], reverse=True)
    elif strategy == "cocoon":
        # シードノードの out_neighbor からランダム選択。足りない場合は 2-hop へ拡張。
        out_neighbors = set()
        for s in seed_nodes:
            out_neighbors.update(
                nbr for nbr in graph.successors(s) if nbr in inactive_nodes
            )
        nbr_list = list(out_neighbors)
        shuffle(nbr_list)
        cand = nbr_list[:k]
        if len(cand) < k:
            two_hop = set()
            for s in seed_nodes:
                for nbr in graph.successors(s):
                    two_hop.update(
                        nbr2
                        for nbr2 in graph.successors(nbr)
                        if nbr2 in inactive_nodes
                    )
            two_hop_list = list(two_hop)
            shuffle(two_hop_list)
            cand += two_hop_list[: k - len(cand)]
        if len(cand) < k:
            rest = list(inactive_nodes - set(cand))
            shuffle(rest)
            cand += rest[: k - len(cand)]
    else:  # 'random'
        cand = list(inactive_nodes)
        shuffle(cand)

    return cand[:k]

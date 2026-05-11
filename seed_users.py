"""初期拡散ノード（seed users）の選定ロジック。

旧 `util.py` から分離。`paths.py` 側で seed_mode の正規化／サフィックスを
管理し、ここではグラフから具体ノードを取り出す責務に集中する。

互換性のため、`util.SEED_MODES` / `util.normalize_seed_mode` /
`util.seed_mode_filename_suffix` / `util.get_seed_users` は `util.py` で
re-export する。
"""

from __future__ import annotations

import random
from typing import Any, List

import numpy as np

from params import (
    SEED_USER_SINGLE_LARGEST_DEGREE,
    SEED_USERS_MULTIPLE_MODERATE_DEGREE,
)

# paths.py 側に seed_mode の集合定義を集約しているため再エクスポート。
from paths import (
    SEED_MODE_ALIASES,
    SEED_MODES,
    normalize_seed_mode,
    seed_mode_filename_suffix,
)


def get_seed_users(
    graph,
    seed_mode: str = "single_largest_degree",
    rng_seed: int = 42,
) -> List[Any]:
    """初期拡散ノードを返す。

    - test グラフ: 最大 out_degree のノード 1 つ。
    - Nikolov 系（`graph_name != 'test'`）:
        - single_largest_degree: 固定ノード 1 つ
        - multiple_moderate_degree: 固定ノード 3 つ
        - random10: rng_seed で再現可能なランダム 10 ノード
    """
    seed_mode = normalize_seed_mode(seed_mode)
    if graph.graph["graph_name"] == "test":
        out_degrees = dict(graph.out_degree())
        max_node = max(out_degrees, key=out_degrees.get)
        return [max_node]

    if seed_mode == "single_largest_degree":
        seed_users = [SEED_USER_SINGLE_LARGEST_DEGREE]
    elif seed_mode == "multiple_moderate_degree":
        seed_users = list(SEED_USERS_MULTIPLE_MODERATE_DEGREE)
    elif seed_mode == "random10":
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        seed_users = random.sample(list(graph.nodes()), 10)
    else:
        raise ValueError(
            f"Unknown seed_mode {seed_mode!r}; expected one of {sorted(SEED_MODES)}"
        )

    for u in seed_users:
        print(
            f"seed user: {u}, suscep: {graph.nodes[u]['suscep']}, "
            f"out_degree: {graph.out_degree(u)}"
        )
    return seed_users

"""固定 η ヒートマップと vary_eta ヒートマップの共通エンジン。

旧 [intervention_analysis.py](intervention_analysis.py) と
[intervention_analysis_varying_eta.py](intervention_analysis_varying_eta.py) の
`create_heatmap` / `save_heatmap_data` を 1 か所に集約する。

CLI / ファイル出力の互換性は完全に維持し、各スクリプトはここに薄く委譲する。
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
from tqdm import tqdm

import util
from CTIC import num_spread_nodes_wo_intervention, run_ctic_simulations
from params import get_eta_lam
from paths import heatmap_path_fixed_eta, heatmap_path_vary_eta

# ---- xlabel/ylabel の対応辞書 ----
_INTERVENTION_LABELS = {
    "prebunking": {
        "fixed": {
            "xlabel": r"$\varepsilon_{\mathrm{pre}}$",
            "ylabel": r"$\delta_{\mathrm{pre}}$",
            "title": "prebunking",
        },
        "vary_eta": {
            "xlabel": r"$\varepsilon_{\mathrm{pre}}$",
            "ylabel": r"$\eta$",
            "title": "prebunking",
        },
    },
    "contextualization": {
        "fixed": {
            "xlabel": r"$\varepsilon_{\mathrm{ctx}}$",
            "ylabel": r"$\phi_{\mathrm{ctx}}$",
            "title": "contextualization",
        },
        "vary_eta": {
            "xlabel": r"$\varepsilon_{\mathrm{ctx}}$",
            "ylabel": r"$\eta$",
            "title": "contextualization",
        },
    },
    "nudging": {
        "vary_eta": {
            "xlabel": r"$\varepsilon_{\mathrm{nud}}$",
            "ylabel": r"$\eta$",
            "title": "nudging",
        },
    },
}


def get_labels(intervention_type: str, axis_y: str) -> dict:
    """ヒートマップの軸ラベル dict を返す。"""
    try:
        return _INTERVENTION_LABELS[intervention_type][axis_y]
    except KeyError as exc:
        raise ValueError(
            f"Unknown intervention/axis combination: ({intervention_type}, {axis_y})"
        ) from exc


# =====================================================================
# create_heatmap (固定 η と vary_eta の両対応)
# =====================================================================
def create_heatmap(
    graph,
    intervention_type: str,
    x_range: Sequence[float],
    y_range: Sequence[float],
    *,
    axis_y: str,  # 'fixed' or 'vary_eta'
    n_simulations: int = 100,
    seed_base: int = 42,
    target_selection: str = "random",
    seed_mode: str = "single_largest_degree",
    eta_scale: str | float = "1",
    fixed_delta_pre: float = 0.2,
    fixed_intervention_threshold: float = 0.8,
):
    """ヒートマップを作って (x_grid, y_grid, heatmap_data) を返す。

    Args:
        axis_y: 'fixed' なら y は介入パラメータ（δ_pre / φ_ctx）、
                'vary_eta' なら y は η そのもの。
        eta_scale: 'fixed' のときに使う η 倍率（vary_eta のときは無視）。
        fixed_delta_pre / fixed_intervention_threshold: vary_eta で固定する値。
    """
    print(f"intervention type: {intervention_type}")
    print(f"axis_y: {axis_y}")
    print(f"# of simulations: {n_simulations}")
    print(f"seed_mode: {seed_mode}")

    if axis_y not in ("fixed", "vary_eta"):
        raise ValueError(f"axis_y は 'fixed' または 'vary_eta': {axis_y!r}")

    graph_name = graph.graph["graph_name"]
    num_x = len(x_range)
    num_y = len(y_range)
    heatmap_data = np.zeros((num_x, num_y))

    eta_base, lam = get_eta_lam(graph_name)
    if axis_y == "fixed":
        eta = eta_base * float(eta_scale)

    seed_nodes = util.get_seed_users(graph, seed_mode=seed_mode, rng_seed=seed_base)

    if axis_y == "fixed":
        # max_spread は介入なしの「想定到達数」を 1 度だけ計算
        expected_max_spread = num_spread_nodes_wo_intervention(
            graph=graph, seed_nodes=seed_nodes, eta=eta, lam=lam
        )

    pbar = tqdm(total=num_x * num_y, desc=f"{intervention_type.title()} Progress")
    for i, y in enumerate(y_range):
        # vary_eta 時は y を η として使い、行ごとに max_spread を再計算
        if axis_y == "vary_eta":
            eta = float(y)
            expected_max_spread = num_spread_nodes_wo_intervention(
                graph=graph, seed_nodes=seed_nodes, eta=eta, lam=lam
            )

        for j, x in enumerate(x_range):
            # 介入パラメータの初期化
            epsilon_pre = epsilon_ctx = epsilon_nud = 0.0
            delta_pre = 0.0
            intervention_threshold = 1.0

            if axis_y == "fixed":
                if intervention_type == "prebunking":
                    epsilon_pre = x
                    delta_pre = y
                elif intervention_type == "contextualization":
                    epsilon_ctx = x
                    intervention_threshold = y
                else:
                    raise ValueError(
                        f"axis_y='fixed' は prebunking/contextualization のみ対応: "
                        f"{intervention_type!r}"
                    )
            else:  # vary_eta
                delta_pre = fixed_delta_pre
                intervention_threshold = fixed_intervention_threshold
                if intervention_type == "prebunking":
                    epsilon_pre = x
                elif intervention_type == "contextualization":
                    epsilon_ctx = x
                elif intervention_type == "nudging":
                    epsilon_nud = x
                else:
                    raise ValueError(
                        f"Unknown intervention type: {intervention_type}"
                    )

            prevalences = run_ctic_simulations(
                num_simulations=n_simulations,
                graph=graph,
                seed_nodes=seed_nodes,
                eta=eta,
                lam=lam,
                max_time=1000.0,
                max_spread=expected_max_spread,
                epsilon_pre=epsilon_pre,
                epsilon_ctx=epsilon_ctx,
                epsilon_nud=epsilon_nud,
                delta_pre=delta_pre,
                intervention_threshold=intervention_threshold,
                target_selection=target_selection,
                seed=seed_base,
            )
            heatmap_data[i, j] = np.mean(prevalences)
            pbar.update(1)
    pbar.close()

    x_grid, y_grid = np.meshgrid(x_range, y_range)
    return x_grid, y_grid, heatmap_data


# =====================================================================
# save_heatmap_data
# =====================================================================
def save_heatmap_data(
    data_dict: dict,
    graph_name: str,
    intervention_type: str,
    *,
    axis_y: str,
    target_selection: str = "random",
    seed_mode: str = "single_largest_degree",
    eta_scale: str | float = "1",
) -> str:
    """ヒートマップデータを保存し、保存パスを返す。"""
    if axis_y == "fixed":
        save_path = heatmap_path_fixed_eta(
            graph_name=graph_name,
            intervention_type=intervention_type,
            target_selection=target_selection,
            seed_mode=seed_mode,
            eta_scale=eta_scale,
        )
    elif axis_y == "vary_eta":
        save_path = heatmap_path_vary_eta(
            graph_name=graph_name,
            intervention_type=intervention_type,
            target_selection=target_selection,
            seed_mode=seed_mode,
        )
    else:
        raise ValueError(f"axis_y は 'fixed' または 'vary_eta': {axis_y!r}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, data_dict)
    print(f"heatmap saved: {save_path}")
    return save_path


# =====================================================================
# 高水準ラッパ: グラフロード→create_heatmap→save の一貫実行
# =====================================================================
def run_intervention_analysis(
    intervention_type: str,
    *,
    axis_y: str,
    graph_name: str = "test",
    n_simulations: int = 100,
    seed_base: int = 42,
    save_data: bool = False,
    target_selection: str = "random",
    seed_mode: str = "single_largest_degree",
    eta_scale: str | float = "1",
    x_range: Sequence[float] | None = None,
    y_range: Sequence[float] | None = None,
) -> dict:
    """1 本の介入解析を実行して data_dict を返す。

    x_range / y_range が None の場合は axis_y に応じた既定グリッドを使う。
    """
    print(f"=== {intervention_type.upper()} heatmap generation (CTIC) ===")
    print(f"graph: {graph_name}")

    from graphs import load_graph_by_name  # 遅延 import で循環回避

    G = load_graph_by_name(graph_name)

    if x_range is None:
        x_range = np.arange(0.0, 1.05, 0.05)
    if y_range is None:
        y_range = (
            np.arange(0.0, 1.05, 0.05)
            if axis_y == "fixed"
            else np.arange(0.0, 0.105, 0.005)
        )

    x_grid, y_grid, heatmap_data = create_heatmap(
        G,
        intervention_type,
        x_range,
        y_range,
        axis_y=axis_y,
        n_simulations=n_simulations,
        seed_base=seed_base,
        target_selection=target_selection,
        seed_mode=seed_mode,
        eta_scale=eta_scale,
    )

    labels = get_labels(intervention_type, axis_y)
    data_dict = {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "heatmap_data": heatmap_data,
        "labels": labels,
        "x_range": x_range,
        "y_range": y_range,
        "intervention_type": intervention_type,
        "graph_name": graph_name,
        "n_simulations": n_simulations,
        "seed_mode": seed_mode,
    }

    if save_data:
        save_heatmap_data(
            data_dict,
            graph_name,
            intervention_type,
            axis_y=axis_y,
            target_selection=target_selection,
            seed_mode=seed_mode,
            eta_scale=eta_scale,
        )
    return data_dict

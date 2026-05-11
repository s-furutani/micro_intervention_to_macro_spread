# %% パス組み立て（intervention_analysis / intervention_analysis_varying_eta の保存規則）
"""
CTIC / 介入解析の結果を可視化する。

構成:
  1. パス用の短いヘルパ（このセル）
  2. 全ヒートマップ共通の `plot_heatmap_grid`（次セル）
  3. 以降 `# %%` ごとに、用途別のプロット関数（各関数がパスを自分で組み立てる）。
     例: `plot_heatmaps_2x5_prevalence`、`plot_heatmaps_2x3_prevalence_vary_eta`、
     `plot_pre_ctx_relative_spread_lines_1x2`（固定 η: δ・φ vs relative spread、ε は色分け）

使い方:
  - 末尾は `# %%` セル（行頭）ごとに `plot_*` を直接記述。Spyder ではセル単位実行可。
  - `python plot_results.py` でも末尾セルが上から順に実行される。
  - 各セルは `if __name__ == "__main__":` でガードしているため、
    `import plot_results` しても副次的にプロット処理が走らない。
  - Spyder のセル実行時は IPython kernel の __name__ が "__main__" になるため、
    セル単位実行は引き続き可能。
"""
from __future__ import annotations

import importlib
import os
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

import plot_heatmap  # util が参照する実体; reload 順序用
import util


def _reload_util_and_plot_heatmap() -> None:
    """Spyder 等で `reload(util)` だけだと `plot_heatmap` が古いまま残るため、先に reload。"""
    importlib.reload(plot_heatmap)
    importlib.reload(util)


# prebunking random vs targeted 用（固定 η の差分図）
PREBUNKING_DIFF_TARGETS: tuple[str, ...] = ("high_degree", "high_susceptible", "cocoon")


def _graph_dir(graph_name: str) -> str:
    return graph_name.strip().rstrip("/") + "/"


def _seed_suffix(seed_mode: str) -> str:
    """`util.seed_mode_filename_suffix` のエイリアス（random10 → _seed10 等）。"""
    return util.seed_mode_filename_suffix(seed_mode)


def _eta_fixed_suffix(eta_scale: str) -> str:
    """固定 η グリッド（intervention_analysis）の `_half` / `_double` / 空。"""
    if eta_scale in ("1", "1.0", ""):
        return ""
    if eta_scale in ("0.5", "half"):
        return "_half"
    if eta_scale in ("2", "2.0", "double"):
        return "_double"
    raise ValueError(f"eta_scale は '1'|'0.5'|'2' 想定: {eta_scale!r}")


def path_vary_eta(graph_name: str, stem: str, seed_mode: str) -> str:
    """例: stem=`nudging` → `results/nikolov/nudging_vary_eta.npy`（seed ありなら `_seed10` 等）。"""
    d = _graph_dir(graph_name)
    sm = _seed_suffix(seed_mode)
    return f"results/{d}{stem}{sm}_vary_eta.npy"


def path_fixed_eta(graph_name: str, stem: str, seed_mode: str, eta_scale: str = "1") -> str:
    """固定 η の .npy。eta_scale で `_half` / `_double`（vary_eta には付けない）。"""
    d = _graph_dir(graph_name)
    sm = _seed_suffix(seed_mode)
    em = _eta_fixed_suffix(eta_scale)
    return f"results/{d}{stem}{sm}{em}.npy"


# ヒートマップ共通: util への唯一の入口
def plot_heatmap_grid(
    nrows: int,
    ncols: int,
    path_list: Sequence[Any],
    relative_flag_list: Sequence[bool],
    titles: Sequence[str | None] | None = None,
    *,
    plot_critical_curve: bool = False,
    plot_axline: bool = False,
    panel_subcaptions: bool = True,
    panel_subcaptions_bottom_row_only: bool | None = None,
) -> None:
    """`util.plot_multiple_heatmaps` の薄いラッパ。path は str または差分用 (path_a, path_b)（a−b）。"""
    n = nrows * ncols
    if len(path_list) != n:
        raise ValueError(f"path_list の長さ {len(path_list)} が nrows×ncols={n} と一致しません")
    if len(relative_flag_list) != n:
        raise ValueError("relative_flag_list の長さが path_list と一致しません")
    ttl = list(titles) if titles is not None else [None] * n
    util.plot_multiple_heatmaps(
        nrows,
        ncols,
        list(path_list),
        list(relative_flag_list),
        ttl,
        plot_axline=plot_axline,
        plot_critical_curve=plot_critical_curve,
        panel_subcaptions=panel_subcaptions,
        panel_subcaptions_bottom_row_only=panel_subcaptions_bottom_row_only,
    )


# ① 2×5: prevalence（上段）+ relative prevalence（下段）
def plot_heatmaps_2x5_prevalence(
    graph_name: str,
    *,
    seed_mode: str = "single_largest_degree",
    eta_scale: str = "1",
    stems_vary: tuple[str, str, str] = ("nudging", "prebunking_random", "contextualization"),
    stems_fixed: tuple[str, str] = ("prebunking_random", "contextualization"),
    plot_critical_curve: bool = True,
) -> None:
    """左から vary_eta×3 + 固定η×2。上段=絶対、下段=相対。"""
    sm = util.normalize_seed_mode(seed_mode)
    # print(f"graph={graph_name} seed_mode={sm} eta_scale(fixedのみ)={eta_scale}")
    paths5 = [
        path_vary_eta(graph_name, stems_vary[0], seed_mode),
        path_vary_eta(graph_name, stems_vary[1], seed_mode),
        path_vary_eta(graph_name, stems_vary[2], seed_mode),
        path_fixed_eta(graph_name, stems_fixed[0], seed_mode, eta_scale),
        path_fixed_eta(graph_name, stems_fixed[1], seed_mode, eta_scale),
    ]
    plot_heatmap_grid(
        2,
        5,
        paths5 * 2,
        [False] * 5 + [True] * 5,
        [None] * 10,
        plot_critical_curve=plot_critical_curve,
        panel_subcaptions_bottom_row_only=True,
    )


# ①′ 2×3: vary_eta のみ prevalence（上段）+ relative prevalence（下段）
def plot_heatmaps_2x3_prevalence_vary_eta(
    graph_name: str,
    *,
    seed_mode: str = "single_largest_degree",
    stems_vary: tuple[str, str, str] = (
        "nudging",
        "prebunking_random",
        "contextualization",
    ),
    plot_critical_curve: bool = True,
) -> None:
    """左から vary_eta×3 のみ。上段=絶対、下段=相対（intervention_analysis_varying_eta の .npy）。"""
    sm = util.normalize_seed_mode(seed_mode)
    paths3 = [
        path_vary_eta(graph_name, stems_vary[0], seed_mode),
        path_vary_eta(graph_name, stems_vary[1], seed_mode),
        path_vary_eta(graph_name, stems_vary[2], seed_mode),
    ]
    plot_heatmap_grid(
        2,
        3,
        paths3 * 2,
        [False] * 3 + [True] * 3,
        [None] * 6,
        plot_critical_curve=plot_critical_curve,
    )


# ② 2×2: 固定 η のみ prevalence + relative（intervention_analysis）
def plot_heatmaps_2x2_prevalence_fixed_eta(
    graph_name: str,
    *,
    seed_mode: str = "single_largest_degree",
    eta_scale: str = "1",
    stems_fixed: tuple[str, str] = ("prebunking_random", "contextualization"),
    plot_critical_curve: bool = True,
) -> None:
    """(ε,δ)/(ε,φ) の 2 列のみ。上段絶対・下段相対。"""
    sm = util.normalize_seed_mode(seed_mode)
    # print(f"2x2 fixed-eta | graph={graph_name} seed_mode={sm} eta_scale={eta_scale}")
    p2 = [
        path_fixed_eta(graph_name, stems_fixed[0], seed_mode, eta_scale),
        path_fixed_eta(graph_name, stems_fixed[1], seed_mode, eta_scale),
    ]
    plot_heatmap_grid(
        2,
        2,
        p2 * 2,
        [False] * 2 + [True] * 2,
        [None] * 4,
        plot_critical_curve=plot_critical_curve,
    )


# ③ 1×5: nikolov − randomized_nikolov（各パネル差分）
def plot_heatmaps_1x5_nikolov_minus_randomized(
    *,
    seed_mode: str = "single_largest_degree",
    eta_scale: str = "1",
    stems_vary: tuple[str, str, str] = ("nudging", "prebunking_random", "contextualization"),
    stems_fixed: tuple[str, str] = ("prebunking_random", "contextualization"),
    plot_critical_curve: bool = True,
) -> None:
    """vary 3 + fixed 2 の順で、(nikolov, randomized) の差を 1 行に並べる。"""
    sm = util.normalize_seed_mode(seed_mode)
    # print(f"1x5 nikolov-randomized | seed_mode={sm} eta_scale(fixedのみ)={eta_scale}")
    pairs: list[tuple[str, str]] = []
    for stem in stems_vary:
        pairs.append(
            (
                path_vary_eta("nikolov", stem, seed_mode),
                path_vary_eta("randomized_nikolov", stem, seed_mode),
            )
        )
    for stem in stems_fixed:
        pairs.append(
            (
                path_fixed_eta("nikolov", stem, seed_mode, eta_scale),
                path_fixed_eta("randomized_nikolov", stem, seed_mode, eta_scale),
            )
        )
    plot_heatmap_grid(
        1,
        5,
        pairs,
        [False] * 5,
        [None] * 5,
        plot_critical_curve=plot_critical_curve,
    )


# ④ 1×3: prebunking random − targeted
def plot_heatmaps_1x3_prebunking_random_vs_targeted(
    graph_name: str,
    *,
    seed_mode: str = "single_largest_degree",
    vary_eta: bool = False,
    targets: Sequence[str] = PREBUNKING_DIFF_TARGETS,
    plot_critical_curve: bool = True,
) -> None:
    """random と各ターゲットの差分（固定 η なら `vary_eta=False`）。"""
    sm = util.normalize_seed_mode(seed_mode)
    sfx = "_vary_eta" if vary_eta else ""
    d = _graph_dir(graph_name)
    seed_mid = _seed_suffix(seed_mode)
    # prebunking_random は target 込みで stem が決まる保存なので、random 側は prebunking_random + seed
    random_path = f"results/{d}prebunking_random{seed_mid}{sfx}.npy"
    pairs = [
        (random_path, f"results/{d}prebunking_{t}{seed_mid}{sfx}.npy") for t in targets
    ]
    # print(
    #     f"1x3 prebunking diff | graph={graph_name} seed_mode={sm} vary_eta={vary_eta}"
    # )
    plot_heatmap_grid(
        1,
        3,
        pairs,
        [False] * 3,
        [None] * 3,
        plot_critical_curve=plot_critical_curve,
        plot_axline=False,
    )


# ⑤ 1×4: 単一 vs 複合 + γ 列の箱ひげ（util.boxplot_jitter_multiple_plots）
def plot_boxplots_1x4_single_vs_combined(
    graph_name: str = "nikolov",
    *,
    target_selection: str = "random",
    seed_mode: str = "single_largest_degree",
    eta_scale: str = "1",
    extra_gammas: tuple[float, ...] = (2.0, 5.0),
    save_name_suffix: str = "_panels_with_gamma",
    nrows: int = 1,
    ncols: int = 4,
    xtick_label_map: dict[str, str] | None = None,
) -> None:
    """
    4 条件の箱ひげ+γ 列。

    現状の `compare_single_and_combined_intervention_*.npy` には seed_mode / eta_scale が
    ファイル名に含まれない想定。引数は将来の拡張とログ用に受け取る。
    """
    sm = util.normalize_seed_mode(seed_mode)
    # print(
    #     f"1x4 boxplots | graph={graph_name} target={target_selection} "
    #     f"seed_mode={sm} eta_scale={eta_scale}"
    # )
    suffixes = ("_base", "_improve_strength", "_improve_reach", "_improve_both")
    titles = ["Base", "Improve strength", "Improve reach", "Improve both"]
    prevalences_lists: list = []
    extra_per_panel: list[dict] = []
    for suf in suffixes:
        base_path = (
            f"results/{graph_name}/compare_single_and_combined_intervention_"
            f"{target_selection}{suf}.npy"
        )
        prevalences_lists.append(np.load(base_path, allow_pickle=True))
        extras: dict[str, np.ndarray] = {}
        for g in extra_gammas:
            gpath = base_path.replace(".npy", f"_gamma{float(g):g}.npy")
            gprev = np.load(gpath, allow_pickle=True)
            extras[fr"Comb($\gamma={g:g}$)"] = gprev[4]
        extra_per_panel.append(extras)
    xmap = xtick_label_map or {
        "Comb": r"$\gamma=1$",
        r"Comb($\gamma=2$)": r"$\gamma=2$",
        r"Comb($\gamma=5$)": r"$\gamma=5$",
    }
    util.boxplot_jitter_multiple_plots(
        prevalences_lists,
        graph_name,
        titles=titles,
        relative_suppression=True,
        save_name=f"compare_single_and_combined_intervention_{target_selection}{save_name_suffix}",
        extra_per_panel=extra_per_panel,
        nrows=nrows,
        ncols=ncols,
        xtick_label_map=xmap,
    )


def _nearest_grid_index(value: float, grid: np.ndarray) -> int:
    """1 次元グリッド上で value に最も近いインデックス。"""
    g = np.asarray(grid, dtype=float).ravel()
    return int(np.argmin(np.abs(g - value)))


def _relative_spread_along_y(
    heatmap_data: np.ndarray,
    y_range: np.ndarray,
    dphi_list: Sequence[float],
    j_eps: int,
    *,
    baseline_y_coord: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    固定 ε（列 j_eps）について、各 y 点の prevalence を
    同じ列で baseline_y_coord に対応する行の値で割った比を、dphi_list の y で返す。

    prebunking（δ）: baseline_y_coord=0（δ=0 で割る）。
    contextualization（φ）: baseline_y_coord=1（φ=1≈介入なしで割る）。
    """
    col = heatmap_data[:, j_eps].astype(float)
    i_base = _nearest_grid_index(baseline_y_coord, y_range)
    denom = float(col[i_base])
    xs = np.asarray(dphi_list, dtype=float)
    if abs(denom) < 1e-15:
        return xs, np.full(len(dphi_list), np.nan)
    rel = col / denom
    ys = np.array([rel[_nearest_grid_index(v, y_range)] for v in dphi_list], dtype=float)
    return xs, ys


def plot_pre_ctx_relative_spread_lines_1x2(
    graph_name: str = "nikolov",
    *,
    seed_mode: str = "single_largest_degree",
    eta_scale: str = "1",
    epsilon_list: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
    dphi_list: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    save_png: bool = False,
    cmap_name: str = "tab10",
) -> None:
    """
    固定 η の `prebunking_random` / `contextualization` の .npy から、
    1 行 2 列の図を描く。左: (δ, relative spread)、右: (φ, relative spread)。
    `epsilon_list` の各 ε を同じパネルに複数折れ線で重ね、色で区別する。

    比: pre は prevalence(δ)/prevalence(δ=0)、ctx は prevalence(φ)/prevalence(φ=1)
    （いずれも同じ ε の列）。
    """
    if len(epsilon_list) < 1:
        raise ValueError("epsilon_list は 1 要素以上必要です")
    sm = util.normalize_seed_mode(seed_mode)
    path_pre = path_fixed_eta(graph_name, "prebunking_random", seed_mode, eta_scale)
    path_ctx = path_fixed_eta(graph_name, "contextualization", seed_mode, eta_scale)
    d_pre = np.load(path_pre, allow_pickle=True).item()
    d_ctx = np.load(path_ctx, allow_pickle=True).item()
    x_pre = np.asarray(d_pre["x_range"], dtype=float).ravel()
    y_pre = np.asarray(d_pre["y_range"], dtype=float).ravel()
    x_ctx = np.asarray(d_ctx["x_range"], dtype=float).ravel()
    y_ctx = np.asarray(d_ctx["y_range"], dtype=float).ravel()
    h_pre = np.asarray(d_pre["heatmap_data"], dtype=float)
    h_ctx = np.asarray(d_ctx["heatmap_data"], dtype=float)

    fig, (ax_pre, ax_ctx) = plt.subplots(
        1, 2, figsize=(14, 5.5), dpi=300, constrained_layout=True
    )
    cmap = plt.get_cmap(cmap_name)
    n_eps = len(epsilon_list)
    for k, eps in enumerate(epsilon_list):
        color = cmap(k / max(n_eps - 1, 1)) if n_eps > 1 else cmap(0.4)
        j_pre = _nearest_grid_index(eps, x_pre)
        j_ctx = _nearest_grid_index(eps, x_ctx)
        xs, y_pre_rel = _relative_spread_along_y(
            h_pre, y_pre, dphi_list, j_pre, baseline_y_coord=0.0
        )
        _, y_ctx_rel = _relative_spread_along_y(
            h_ctx, y_ctx, dphi_list, j_ctx, baseline_y_coord=1.0
        )
        lab = rf"$\varepsilon={eps:g}$"
        ax_pre.plot(xs, y_pre_rel, marker="o", linewidth=2, color=color, label=lab)
        ax_ctx.plot(xs, y_ctx_rel, marker="o", linewidth=2, color=color, label=lab)

    ax_pre.set_title("Prebunking (random target)", fontsize=13)
    ax_pre.set_xlabel(r"$\delta_{\mathrm{pre}}$", fontsize=14)
    ax_pre.set_ylabel("Relative spread", fontsize=14)
    ax_pre.set_xticks(np.asarray(dphi_list, dtype=float))
    ax_pre.grid(True, alpha=0.35)
    ax_pre.legend(loc="best", fontsize=10, title=r"$\varepsilon_{\mathrm{pre}}$")

    ax_ctx.set_title("Contextualization", fontsize=13)
    ax_ctx.set_xlabel(r"$\phi_{\mathrm{ctx}}$", fontsize=14)
    ax_ctx.set_ylabel("Relative spread", fontsize=14)
    ax_ctx.set_xticks(np.asarray(dphi_list, dtype=float))
    ax_ctx.grid(True, alpha=0.35)
    ax_ctx.legend(loc="best", fontsize=10, title=r"$\varepsilon_{\mathrm{ctx}}$")

    fig.suptitle(
        f"{graph_name} | seed_mode={sm} | pre: prev./prev. at "
        r"$\delta_{\mathrm{pre}}=0$ | ctx: prev./prev. at $\phi_{\mathrm{ctx}}=1$",
        fontsize=11,
    )
    if save_png:
        out_dir = f"results/{_graph_dir(graph_name)}png"
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(f"{out_dir}/pre_ctx_relative_spread_lines_1x2_dpi300.png")
    plt.show()


# 旧名（2×4 版）からの互換エイリアス
plot_pre_ctx_relative_spread_slices_2x4 = plot_pre_ctx_relative_spread_lines_1x2


# 以下は実行用セル（# %%）。
# `python plot_results.py` か Spyder のセル実行で動作する。
# 一方で `import plot_results` による副次的な実行を防ぐため、
# 各セルを `if __name__ == "__main__":` でガードしている。
# Spyder では IPython kernel の __name__ が "__main__" になるため、セル実行は引き続き可能。


# 補助関数（NN プロット）
def compute_nearest_neighbor_mean(graph, x):
    x_nn = {}
    for v in graph.nodes():
        neighbor_signals = [x[u] for u in graph.predecessors(v)]
        x_nn[v] = np.mean(neighbor_signals) if neighbor_signals else 0.0
    return x_nn


def plot_nearest_neighbor_mean(node_list, suscep, suscep_nn, suscep_r, suscep_nn_r):
    import seaborn as sns

    x, y = [], []
    x_r, y_r = [], []
    for node in node_list:
        x.append(suscep[node])
        y.append(suscep_nn[node])
        x_r.append(suscep_r[node])
        y_r.append(suscep_nn_r[node])

    cmap = plt.get_cmap('YlGnBu_r')
    plt.figure(figsize=(10, 5.5), dpi=300)
    x_list = [x, x_r]
    y_list = [y, y_r]
    title_list = ['Original', 'Randomized']
    subcaption_text = ['(a)', '(b)']
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        sns.kdeplot(x=x_list[i], y=y_list[i], fill=True, cmap=cmap, thresh=0.000, levels=10)
        plt.xlabel(r'$s_v$', fontsize=24)
        plt.ylabel(r'$\langle s_v^{\mathrm{NN}} \rangle$', fontsize=24)
        plt.xlim(0, 0.5)
        plt.ylim(0, 0.5)
        plt.title(title_list[i], fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.text(0.25, -0.1, subcaption_text[i], fontsize=28, ha='center', va='top')
    plt.tight_layout()
    plt.show()


# %% nikolov 2x3 vary_eta and 2x2 fixed-eta
if __name__ == "__main__":
    _reload_util_and_plot_heatmap()
    print("nikolov 2x3 vary_eta")
    plot_heatmaps_2x3_prevalence_vary_eta(
        "nikolov",
        seed_mode="single_largest_degree",
    )

    print("nikolov 2x2 fixed-eta")
    plot_heatmaps_2x2_prevalence_fixed_eta(
        "nikolov",
        seed_mode="single_largest_degree",
        eta_scale="1",
    )

# %% 2×5 単一グラフ（nikolov & randomized_nikolov）
if __name__ == "__main__":
    _reload_util_and_plot_heatmap()
    print("nikolov")
    plot_heatmaps_2x5_prevalence(
        "nikolov",
        seed_mode="single_largest_degree",
        eta_scale="1",
    )
    print("randomized_nikolov")
    plot_heatmaps_2x5_prevalence(
        "randomized_nikolov",
        seed_mode="single_largest_degree",
        eta_scale="1",
    )


# %% nikolov − randomized（1×5）
if __name__ == "__main__":
    _reload_util_and_plot_heatmap()
    print("nikolov − randomized_nikolov")
    plot_heatmaps_1x5_nikolov_minus_randomized(
        seed_mode="single_largest_degree",
        eta_scale="1",
    )

# %% 固定 η のみ（2×2）
if __name__ == "__main__":
    _reload_util_and_plot_heatmap()
    print("nikolov (eta = 0.5 * eta_base)")
    plot_heatmaps_2x2_prevalence_fixed_eta(
        "nikolov",
        seed_mode="single_largest_degree",
        eta_scale="0.5",
    )

    print("nikolov (eta = 2 * eta_base)")
    plot_heatmaps_2x2_prevalence_fixed_eta(
        "nikolov",
        seed_mode="single_largest_degree",
        eta_scale="2",
    )

# %% different seed nodes
if __name__ == "__main__":
    _reload_util_and_plot_heatmap()
    print("nikolov with 10 randomly selected seed nodes")
    plot_heatmaps_2x5_prevalence(
        "nikolov",
        seed_mode="random10",
        eta_scale="1",
    )

    print("nikolov with 3 moderate-degree high-susceptible seed nodes")
    plot_heatmaps_2x5_prevalence(
        "nikolov",
        seed_mode="multiple_moderate_degree",
        eta_scale="1",
    )


# %% pre / ctx: 固定 η（δ・φ vs relative spread, 1×2・ε は色分け）
if __name__ == "__main__":
    _reload_util_and_plot_heatmap()
    print("pre / ctx relative spread lines 1x2")
    plot_pre_ctx_relative_spread_lines_1x2(
        "nikolov",
        seed_mode="single_largest_degree",
        eta_scale="1",
    )

# %% prebunking random vs targeted（固定 η）
if __name__ == "__main__":
    _reload_util_and_plot_heatmap()
    print("prebunking random vs targeted")
    plot_heatmaps_1x3_prebunking_random_vs_targeted(
        "nikolov",
        seed_mode="single_largest_degree",
        vary_eta=False,
    )

# %% 箱ひげ 1×4
if __name__ == "__main__":
    _reload_util_and_plot_heatmap()
    print("boxplots of single vs combined interventions")
    plot_boxplots_1x4_single_vs_combined(
        "nikolov",
        target_selection="random",
        seed_mode="single_largest_degree",
        eta_scale="1",
    )


# %% NN マップ
if __name__ == "__main__":
    from load_graph import Nikolov_susceptibility_graph, randomized_nikolov_graph

    nikolov_graph = Nikolov_susceptibility_graph()
    suscep = {node: nikolov_graph.nodes[node]['suscep'] for node in nikolov_graph.nodes}
    suscep_nn = compute_nearest_neighbor_mean(nikolov_graph, suscep)
    nikolov_graph_r = randomized_nikolov_graph()
    suscep_r = {node: nikolov_graph_r.nodes[node]['suscep'] for node in nikolov_graph_r.nodes}
    suscep_nn_r = compute_nearest_neighbor_mean(nikolov_graph_r, suscep_r)
    node_list = list(nikolov_graph.nodes)

    plot_nearest_neighbor_mean(node_list, suscep, suscep_nn, suscep_r, suscep_nn_r)


# %%

import heapq
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

import util
from graphs import load_graph_by_name
from prebunking_targets import select_prebunking_targets

INACTIVE, SPREAD = 0, 1

def run_continuous_time_independent_cascade(
    graph: nx.Graph,
    seed_nodes: List[Any],
    # model parameters
    eta: float,
    lam: float = 1.0,
    max_time: float = 1000.0,
    max_spread: float = 10000000.0,
    # intervention parameters
    epsilon_pre: float = 0.0,    # Prebunking効果強度 (0-1)
    epsilon_ctx: float = 0.0,     # Debunking効果強度 (0-1)
    epsilon_nud: float = 0.0,     # Nudging効果強度 (0-1)
    delta_pre: float = 0.0,      # Prebunking対象ノード割合 (0-1)
    intervention_threshold: float = 1.0,  # Debunking介入判断の閾値
    # intervention strategy
    target_selection: str = 'random',  # 'random', 'high_degree', 'high_susceptible', 'cocoon'
    # combination rule: gamma-saturated multiplicative reduction
    # eps_comb(gamma) = (1 - prod_i (1 - eps_i^gamma))^(1/gamma)
    # gamma=1: 独立乗法 (従来挙動), gamma->infty: max_i eps_i (完全飽和)
    gamma: float = 1.0,
    # random seed (for reproducibility, None for random)
    seed: int = None,
) -> Tuple[List[float], List[int], Optional[float]]:
    """
    連続時間独立カスケード（CTIC）をイベント駆動で実行。
    - エッジ到達イベント (t_arrive, u, v) を優先度キューで最時刻順処理
    - 到着成立（prob * throttling）後に、ノード内部で採用確率 p_adopt を一度だけ判定
    - 採用→SPREAD（以降）
    - 介入 (debunking) はSPREADの割合がintervention_thresholdを超えたら適用
    - Prebunking は開始前に対象ノードの susceptibility を下げる
    - Nudging が開始前に全ノードの susceptibility を下げる

    グラフ要件:
      - 各ノード: graph.nodes[v]['suscep']（なければ1.0）
      - 各エッジ: 
          - graph[u][v]['prob']  : 到着成功確率（なければ1.0）


    返り値はイベント時刻ごとの系列（記録は「何か状態が変わった時」にだけ更新）。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    nodes = list(graph.nodes())
    N = len(nodes)

    # --- 初期状態 ---
    state: Dict[Any, int] = {v: INACTIVE for v in nodes}
    spread_time: Dict[Any, float] = {}
    spread_nodes = set(seed_nodes)
    inactive_nodes = set(nodes) - spread_nodes

    # node susceptibility (raw, immutable). gamma-saturated 結合のため、介入による低減は
    # p_adopt 計算時に combine_factor() でまとめて適用する。
    susceptibility: Dict[Any, float] = {v: float(graph.nodes[v].get('suscep', 0.0)) for v in nodes}

    # --- Prebunking target selection (before start) ---
    # 注: 旧実装では susceptibility[v] *= (1 - epsilon_pre) を即時適用していたが、
    # gamma-saturated 結合のため適用は遅延させ、ここでは対象集合 pre_targets のみ確定する。
    pre_targets: set = set()
    if epsilon_pre > 0 and delta_pre > 0:
        k = max(1, int(N * delta_pre))
        # 注: rng=None で `random` グローバル状態を使い、上記 random.seed(seed) の影響を受ける
        pre_targets = set(
            select_prebunking_targets(
                graph,
                k=k,
                strategy=target_selection,
                seed_nodes=seed_nodes,
                susceptibility=susceptibility,
                inactive_nodes=inactive_nodes,
                rng=None,
            )
        )

    # 注: Nudging (epsilon_nud) は全ノードへ作用するため、ここでは何もしない。
    # ctx (epsilon_ctx) も intervention_done フラグが立った後で全ノードへ作用する。
    # いずれも下記 combine_factor() でまとめて p_adopt に反映する。

    def combine_factor(v: Any, ctx_active: bool) -> float:
        """ノード v に作用する介入群の有効因子 (1 - eps_comb) を返す。

        eps_comb(gamma) = (1 - prod_i (1 - eps_i^gamma))^(1/gamma)
        gamma == 1.0 で独立乗法 (= prod_i (1 - eps_i)) と同一。
        """
        eps_list = []
        if epsilon_pre > 0 and v in pre_targets:
            eps_list.append(epsilon_pre)
        if epsilon_nud > 0:
            eps_list.append(epsilon_nud)
        if ctx_active and epsilon_ctx > 0:
            eps_list.append(epsilon_ctx)
        if not eps_list:
            return 1.0
        if gamma == 1.0:
            f = 1.0
            for e in eps_list:
                f *= (1.0 - e)
            return f
        # gamma-saturated rule (大 gamma での桁落ちを避けるため log1p / expm1 で計算)
        # log_prod = sum_i log(1 - e_i^gamma) = sum_i log1p(-exp(gamma * log e_i))
        log_prod = 0.0
        for e in eps_list:
            if e <= 0.0:
                continue
            log_prod += math.log1p(-math.exp(gamma * math.log(e)))
        # 1 - prod = -expm1(log_prod)（log_prod ~ 0 でも精度が保たれる）
        one_minus_prod = -math.expm1(log_prod)
        if one_minus_prod <= 0.0:
            return 1.0
        eps_comb = math.exp(math.log(one_minus_prod) / gamma)
        return 1.0 - eps_comb

    # --- seed nodes to SPREAD ---
    for s in list(spread_nodes):
        state[s] = SPREAD
        spread_time[s] = 0.0
        if s in inactive_nodes:
            inactive_nodes.remove(s)

    # --- event queue (scheduled arrival) ---
    # elements: (t_arrive, u, v)
    Q: List[Tuple[float, Any, Any]] = []

    def schedule_from(u: Any, t0: float):
        """schedule the arrival event of u→* once when u becomes SPREAD at time t0."""
        for v in graph.neighbors(u):
            if state[v] != INACTIVE:
                continue
            delay = np.random.exponential(lam)
            heapq.heappush(Q, (t0 + delay, u, v))

    # schedule the spread from seed nodes
    for s in spread_nodes:
        schedule_from(s, 0.0)

    # record only when an event occurs
    times      = [0.0]
    in_counts  = [len(inactive_nodes)]
    sp_counts  = [len(spread_nodes)]

    # intervention (debunk/throttle) flag
    intervention_done = False
    intervention_time = None
    prop_rate = 1.0

    # --- main loop (jump to event time) ---
    while Q:
        t, u, v = heapq.heappop(Q)
        if t > max_time:
            break

        # intervention is applied when the proportion of SPREAD exceeds intervention_threshold
        # 注: 旧実装では adoption_rate *= (1 - epsilon_ctx) としていたが、
        # gamma-saturated 結合のためフラグだけ立て、適用は combine_factor() に集約する。
        if (not intervention_done) and (sp_counts[-1] / max_spread >= intervention_threshold):
            intervention_done = True
            intervention_time = t

        # if already active, ignore (only the first arrival triggers the decision)
        if state[v] != INACTIVE:
            continue

        # 1) success of edge arrival
        p_prop = eta * prop_rate
        if random.random() >= p_prop:
            continue

        # 2) decision inside the node (once after arrival)
        # 介入の有効因子を gamma-saturated ルールで結合してから素のsusceptibilityに乗じる
        p_adopt = susceptibility[v] * combine_factor(v, intervention_done)
        if random.random() < p_adopt:
            state[v] = SPREAD
            spread_nodes.add(v)
            inactive_nodes.discard(v)
            spread_time[v] = t
            schedule_from(v, t)

        # update the log (only when an event occurs)
        times.append(t)
        in_counts.append(len(inactive_nodes))
        sp_counts.append(len(spread_nodes))

    return times, sp_counts, intervention_time


def plot_results(
    times: List[int], 
    spread_counts: List[int],
    intervention_time: int = None,
    title: str = "Improved Independent Cascade Model Results"
):
    """
    plot the results of the improved independent cascade model
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    # show the intervention step with a vertical line
    if intervention_time:
        plt.axvline(x=intervention_time, color='k', linestyle='--')
        plt.text(intervention_time+1, max(spread_counts), f'$\\tau$={intervention_time:.2f}', color='k', fontsize=14)

    plt.plot(times, spread_counts, label='Spread', color='green', marker='^')
    
    plt.xlabel('Time')
    plt.ylabel('Number of Nodes')
    plt.xlim(0, max(times))
    # plt.title(title)
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    plt.show()

# load_graph_by_name は graphs.py に集約済み（冒頭で import）。
# 旧 API として CTIC.load_graph_by_name で参照されても動くよう、ここでは何もしない。


def get_high_susceptible_nodes(G):
    return sorted(G.nodes(), key=lambda v: G.nodes[v]['suscep'], reverse=True)

def run_ctic_simulations(num_simulations, graph, seed_nodes, eta, lam, max_time, max_spread, epsilon_pre, epsilon_ctx, epsilon_nud, delta_pre, intervention_threshold, target_selection, seed=42, gamma: float = 1.0):
    n = len(graph.nodes())
    prevalences = []
    for sim in range(num_simulations):
        times, spread_counts, intervention_time = run_continuous_time_independent_cascade(
            graph=graph,
            seed_nodes=seed_nodes,
            eta=eta,
            lam=lam,
            max_time=max_time,
            max_spread=max_spread,
            epsilon_pre=epsilon_pre,
            epsilon_ctx=epsilon_ctx,
            epsilon_nud=epsilon_nud,
            delta_pre=delta_pre,
            intervention_threshold=intervention_threshold,
            target_selection=target_selection,
            gamma=gamma,
            seed=seed + sim
        )
        prevalence = spread_counts[-1] / n
        prevalences.append(prevalence)
    return np.array(prevalences)

def num_spread_nodes_wo_intervention(graph, seed_nodes, eta, lam):
    realizations = []
    for k in (range(10)):
        _, spread_counts, _ = run_continuous_time_independent_cascade(graph=graph, seed_nodes=seed_nodes, eta=eta, lam=lam)
        realizations.append(spread_counts[-1])
    mean = np.mean(realizations)
    std = np.std(realizations)
    # print(f'mean: {mean:.1f}, std: {std:.1f}')
    return mean

def compare_nikolov_and_randomized_nikolov(eta, lam, epsilon_pre, epsilon_ctx, epsilon_nud, delta_pre, intervention_threshold, target_selection, seed=42):
    G = load_graph_by_name('nikolov')
    G_r = load_graph_by_name('randomized_nikolov')
    n = len(G.nodes())
    seed_nodes = util.get_seed_users(G, seed_mode='single_largest_degree')
    # k回分の曲線をそのままプロット（薄い緑/赤）、平均曲線を濃く
    k = 1  # シミュレーション回数

    def interpolate_spread(times, spread_counts, t_grid):
        return np.interp(t_grid, times, spread_counts, left=spread_counts[0], right=spread_counts[-1])

    all_times = []
    all_spreads = []
    all_times_r = []
    all_spreads_r = []
    intervention_times = []
    intervention_times_r = []
    max_time1 = []
    max_time2 = []

    for sim in tqdm(range(k)):
        t, s, tau = run_continuous_time_independent_cascade(
            graph=G, seed_nodes=seed_nodes, eta=eta, lam=lam,
            epsilon_pre=epsilon_pre, epsilon_ctx=epsilon_ctx, epsilon_nud=epsilon_nud,
            delta_pre=delta_pre, intervention_threshold=intervention_threshold,
            target_selection=target_selection, seed=42+sim
        )
        t_r, s_r, tau_r = run_continuous_time_independent_cascade(
            graph=G_r, seed_nodes=seed_nodes, eta=eta, lam=lam,
            epsilon_pre=epsilon_pre, epsilon_ctx=epsilon_ctx, epsilon_nud=epsilon_nud,
            delta_pre=delta_pre, intervention_threshold=intervention_threshold,
            target_selection=target_selection, seed=1042+sim
        )
        all_times.append(t)
        all_spreads.append(np.array(s)/n)
        all_times_r.append(t_r)
        all_spreads_r.append(np.array(s_r)/n)
        intervention_times.append(tau)
        intervention_times_r.append(tau_r)
        max_time1.append(t[-1])
        max_time2.append(t_r[-1])

    t_max = max(max(max_time1), max(max_time2))
    t_grid = np.linspace(0, t_max, 300)

    arr = np.array([interpolate_spread(tt, ss, t_grid) for tt, ss in zip(all_times, all_spreads)])
    arr_r = np.array([interpolate_spread(tt, ss, t_grid) for tt, ss in zip(all_times_r, all_spreads_r)])

    mean_curve = arr.mean(axis=0)
    mean_curve_r = arr_r.mean(axis=0)

    # 平均intervention_timeも参考に得る（Noneの場合に注意）
    def safe_mean(values):
        values = [v for v in values if v is not None]
        return np.mean(values) if values else None

    tau_mean = safe_mean(intervention_times)
    tau_mean_r = safe_mean(intervention_times_r)

    plt.figure(figsize=(8,5))
    # k回それぞれの曲線
    for i in range(k):
        plt.plot(t_grid, arr[i], color="green", alpha=0.15, linewidth=1)
        plt.plot(t_grid, arr_r[i], color="red", alpha=0.15, linewidth=1)
    # 平均曲線
    plt.plot(t_grid, mean_curve, color="green", linewidth=2.5, label='Nikolov (mean)')
    plt.plot(t_grid, mean_curve_r, color="red", linewidth=2.5, label='Randomized Nikolov (mean)')
    # 半減期をプロット
    h = mean_curve[-1] / 2
    t_h = t_grid[np.argmin(np.abs(mean_curve - h))]
    h_r = mean_curve_r[-1] / 2
    t_h_r = t_grid[np.argmin(np.abs(mean_curve_r - h_r))]
    plt.axvline(x=t_h, color='green', linestyle='--', alpha=0.7)
    plt.text(t_h+1, mean_curve.max()*0.95, r'$\tau_{G}=$'+f'${t_h:.2f}$', color='green', fontsize=12)
    plt.axvline(x=t_h_r, color='red', linestyle='--', alpha=0.7)
    plt.text(t_h_r+1, mean_curve_r.max()*0.85, r'$\tau_{\tilde{G}}=$'+f'${t_h_r:.2f}$', color='red', fontsize=12)

    # 介入タイミング平均値
    if tau_mean is not None:
        plt.axvline(x=tau_mean, color='green', linestyle='--', alpha=0.7)
        plt.text(tau_mean+1, mean_curve.max()*0.95, f'$\\tau_{{Nik}}$={tau_mean:.2f}', color='green', fontsize=12)
    if tau_mean_r is not None:
        plt.axvline(x=tau_mean_r, color='red', linestyle='--', alpha=0.7)
        plt.text(tau_mean_r+1, mean_curve_r.max()*0.85, f'$\\tau_{{Rand}}$={tau_mean_r:.2f}', color='red', fontsize=12)
    plt.xlabel('Time')
    plt.ylabel('Number of Nodes')
    # plt.xlim(0, 200)
    plt.ylim(0, 0.6)
    plt.title(f"Target Selection: {target_selection}  ({k} runs, raw curves and mean)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# example
if __name__ == "__main__":
    # compare_nikolov_and_randomized_nikolov(eta=0.079, lam=0.4, epsilon_pre=0.2, epsilon_ctx=0.0, epsilon_nud=0.0, delta_pre=0.2, intervention_threshold=1.0, target_selection='random')
    # compare_nikolov_and_randomized_nikolov(eta=0.079, lam=0.4, epsilon_pre=0.2, epsilon_ctx=0.0, epsilon_nud=0.0, delta_pre=0.2, intervention_threshold=1.0, target_selection='cocoon')
    # create a test graph
    dataset_name = 'nikolov'
    G = load_graph_by_name(dataset_name)

    best_eta = 0.026
    out_degree = [d for _, d in G.out_degree()]
    ave_degree = np.mean(out_degree)
    ave_suscep = np.mean(list(G.nodes[node]['suscep'] for node in G.nodes))
    print(f"ave_degree: {ave_degree:.1f}, ave_suscep: {ave_suscep:.3f}, best_eta: {best_eta:.3f}")
    R0 = best_eta * ave_degree * ave_suscep
    print(f"R0: {R0:.1f}")

    print(len(G.nodes()))
    seed_nodes = util.get_seed_users(G, seed_mode='single_largest_degree')
    # initial spread nodes
    eta = 0.026
    lam = 0.25
    max_time = 200
    epsilon_pre = 0.2
    epsilon_ctx = 0.0
    epsilon_nud = 0.0
    delta_pre = 0.3
    intervention_threshold = 1.0
    target_selection = 'cocoon'
    print(f"target_selection: {target_selection}")

    times, spread_counts, intervention_time = run_continuous_time_independent_cascade(G, seed_nodes=seed_nodes, eta=eta, lam=lam, max_time=max_time, epsilon_pre=epsilon_pre, epsilon_ctx=epsilon_ctx, epsilon_nud=epsilon_nud, delta_pre=delta_pre, intervention_threshold=intervention_threshold, target_selection=target_selection)
    print(f"intervention_time: {intervention_time}")
    plot_results(times, spread_counts, intervention_time, title="CTIC with Interventions")
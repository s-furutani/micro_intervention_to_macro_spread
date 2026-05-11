import argparse
import os
import time

import numpy as np

import util
from CTIC import num_spread_nodes_wo_intervention, run_ctic_simulations
from graphs import load_graph_by_name
from params import get_eta_lam
from paths import single_vs_combined_path


def str2bool(v: object) -> bool:
    """argparse 用: 'False' 文字列が真にならないように bool に変換する。"""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("yes", "true", "t", "y", "1"):
        return True
    if s in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"boolean value expected, got {v!r}")

def compare_single_and_combined_intervention(
    graph_name,
    n_simulations=100,
    target_selection='high_susceptible',
    eta_scale=1.0,
    improve_reach=False,
    improve_strength=False,
    gamma: float = 1.0,
    seed_mode: str = 'single_largest_degree',
):
    """単一介入と複合介入の比較"""
    graph = load_graph_by_name(graph_name)
    seed_base = 42
    intervention_type_list = ['none', 'prebunking', 'contextualization', 'nudging', 'combined']

    eta, lam = get_eta_lam(graph_name)
    eta = eta * float(eta_scale)

    # gamma != 1.0 では combined のみ結果が変わるため、それ以外は gamma=1.0 の結果ファイルから流用する。
    is_gamma_neq_1 = float(gamma) != 1.0
    if is_gamma_neq_1:
        base_path = single_vs_combined_path(
            graph_name,
            target_selection,
            improve_reach=improve_reach,
            improve_strength=improve_strength,
            seed_mode=seed_mode,
            eta_scale=eta_scale,
            gamma=1.0,
        )
        if not os.path.exists(base_path):
            raise FileNotFoundError(
                f"gamma={gamma} の実行には先に gamma=1.0 の結果が必要にゃん: {base_path}"
            )
        baseline_prev = np.load(base_path, allow_pickle=True)

    prevalences_list = []
    for idx, intervention_type in enumerate(intervention_type_list):
        # gamma != 1.0 のとき，combined 以外は gamma に依存しないため再シミュレーションは省略
        if is_gamma_neq_1 and intervention_type != 'combined':
            print(f"intervention type: {intervention_type} -> skip (reuse gamma=1.0 result)")
            prevalences_list.append(np.asarray(baseline_prev[idx]))
            continue
        print(f"intervention type: {intervention_type}")
        epsilon_nud = 0.0
        epsilon_pre = 0.0
        epsilon_ctx = 0.0
        delta_pre = 0.0
        phi_ctx = 1.0

        if intervention_type == 'none':
            pass
        elif intervention_type == 'prebunking':
            epsilon_pre = 0.2
            delta_pre = 0.2
            if improve_reach:
                delta_pre = delta_pre + 0.1
            if improve_strength:
                epsilon_pre = epsilon_pre + 0.1
        elif intervention_type == 'contextualization':
            epsilon_ctx = 0.4
            phi_ctx = 0.8
            if improve_reach:
                phi_ctx = phi_ctx - 0.1
            if improve_strength:
                epsilon_ctx = epsilon_ctx + 0.1
        elif intervention_type == 'nudging':
            epsilon_nud = 0.2
            if improve_strength:
                epsilon_nud = epsilon_nud + 0.1
        elif intervention_type == 'combined':
            epsilon_nud = 0.2
            epsilon_pre = 0.2
            epsilon_ctx = 0.4
            delta_pre = 0.2
            phi_ctx = 0.7
            if improve_reach:
                phi_ctx = phi_ctx - 0.1
            if improve_strength:
                epsilon_ctx = epsilon_ctx + 0.1
                epsilon_nud = epsilon_nud + 0.1
        
        seed_nodes = util.get_seed_users(graph, seed_mode=seed_mode, rng_seed=seed_base)
        if intervention_type in ['contextualization', 'combined']:
            expected_max_spread = num_spread_nodes_wo_intervention(graph=graph, seed_nodes=seed_nodes, eta=eta, lam=lam)
        else:
            expected_max_spread = 10000000.0
        prevalences = run_ctic_simulations(n_simulations, graph, seed_nodes, eta, lam, 1000.0, expected_max_spread, epsilon_pre, epsilon_ctx, epsilon_nud, delta_pre, phi_ctx, target_selection, seed_base, gamma=gamma)
        prevalences_list.append(prevalences)
    
    prevalences_list = np.array(prevalences_list)
    output_path = single_vs_combined_path(
        graph_name,
        target_selection,
        improve_reach=improve_reach,
        improve_strength=improve_strength,
        seed_mode=seed_mode,
        eta_scale=eta_scale,
        gamma=gamma,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, prevalences_list)
    return prevalences_list


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_name', default='nikolov')
    parser.add_argument('--n_simulations', type=int, default=10)
    parser.add_argument('--target_selection', default='random')
    # --improve_reach のみ → True; --improve_reach False → 偽（文字列 "False" の罠を避ける）
    parser.add_argument(
        '--improve_reach',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
    )
    parser.add_argument(
        '--improve_strength',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
    )
    # gamma-saturated 結合のサチュレーション度合い (gamma=1: 独立乗法)
    parser.add_argument('--gamma', type=float, default=1.0)
    _seed_choices = sorted(util.SEED_MODES | set(util.SEED_MODE_ALIASES.keys()))
    parser.add_argument(
        '--seed_mode',
        type=str,
        default='single_largest_degree',
        choices=_seed_choices,
        help='初期拡散ノード（util.get_seed_users の seed_mode）',
    )
    args = parser.parse_args()
    graph_name = args.graph_name
    n_simulations = args.n_simulations
    target_selection = args.target_selection
    improve_reach = args.improve_reach
    improve_strength = args.improve_strength
    gamma = args.gamma
    seed_mode = util.normalize_seed_mode(args.seed_mode)
    prevalences_list = compare_single_and_combined_intervention(
        graph_name,
        n_simulations=n_simulations,
        target_selection=target_selection,
        improve_reach=improve_reach,
        improve_strength=improve_strength,
        gamma=gamma,
        seed_mode=seed_mode,
    )
    # util.violin_plot(prevalences_list, graph_name)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

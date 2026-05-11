import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import random
import networkx as nx
from CTIC import run_continuous_time_independent_cascade
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from load_graph import Nikolov_susceptibility_graph
from collections import Counter
plt.rcParams['font.family'] = 'Arial'

@dataclass
class LossConfig:
    w_curve: float = 1.0
    w_tfrac: float = 0.5
    w_final: float = 0.01
    t_fracs: Tuple[float, float] = (0.10, 0.25)

def euclidean_distance(emp_curve, sim_curve, N=276):
    n = len(emp_curve)
    n = min(n, N)
    dist = 0.0
    for i in range(n):
        dist += np.abs(emp_curve[i] - sim_curve[i])
    return dist / n

def get_sim_curve(graph, e, l, R, base_seed=123):    
    sim_curves = []
    twitter_seed_nodes = get_seed_users_twitter(min_cascade_size=100)
    out_degree = [graph.out_degree(node) for node in twitter_seed_nodes]
    max_out_degree_node = twitter_seed_nodes[np.argmax(out_degree)]
    seed_nodes = [max_out_degree_node]
    # print(f"max_out_degree_node: {max_out_degree_node}, out_degree: {out_degree[np.argmax(out_degree)]}")
    for r in range(R):
        times, counts, _ = run_continuous_time_independent_cascade(graph, seed_nodes, e, l, seed=base_seed+r)
        sim_curves.append((times, counts))
    sim_curve = representive_curve(sim_curves, method='mean')
    return sim_curve

def loss_func(graph, e, l, emp_curve, R, early_phase_hours=None):
    """
    ロス関数（立ち上がり区間を重視可能）
    
    Args:
        graph: グラフ
        e: etaパラメータ
        l: lamパラメータ
        emp_curve: 観測曲線
        R: シミュレーション試行回数
        early_phase_hours: 立ち上がり区間の時間（時間単位）。指定した場合、この区間のみを評価
    
    Returns:
        ロス値
    """
    sim_curve = get_sim_curve(graph, e, l, R=R)
    
    # 立ち上がり区間のみを評価する場合、曲線を最初のk個に切り詰める
    if early_phase_hours is not None:
        # グリッドは1時間刻みを想定（representive_curveでgrid = np.arange(0.0, 100 + 1e-9, 1)）
        k = int(early_phase_hours)
        if k > 0 and k <= len(emp_curve):
            emp_curve = emp_curve[:k]
            sim_curve = sim_curve[:k]
        elif k > len(emp_curve):
            # kが曲線の長さを超える場合は、曲線全体を使用
            pass
    
    dist = euclidean_distance(emp_curve, sim_curve)
    return dist

def loss_func_wrapper(x, graph, emp_curve, R, early_phase_hours=None):
    eta, lam = x[0], x[1]
    return loss_func(graph, eta, lam, emp_curve, R=R, early_phase_hours=early_phase_hours)

def optimization_callback(xk):
    global step_count, optimization_history, graph, empirical_curve, R
    step_count += 1
    eta, lam = xk[0], xk[1]
    
    loss_value = loss_func_wrapper(xk, graph, empirical_curve, R)
    optimization_history.append({
        'step': step_count,
        'eta': eta,
        'lam': lam,
        'loss': loss_value
    })
    
    print(f"Step {step_count}: eta={eta:.6f}, lam={lam:.6f}, loss={loss_value:.6f}")

def resample_pc(times, y, grid):
    times = np.asarray(times, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.searchsorted(times, grid, side="right") - 1
    idx[idx < 0] = -1
    out = np.zeros_like(grid, dtype=float)
    m = len(times)
    for i, j in enumerate(idx):
        if j >= 0:
            out[i] = y[j]
        else:
            out[i] = 0.0
    return out

def representive_curve(observed, method='median'):
    grid = np.arange(0.0, 100 + 1e-9, 1)
    resampled_curves = []
    for (times, counts) in observed:
        resampled_curves.append(resample_pc(times, counts, grid))
        
    if method == 'median':
        return np.median(np.stack(resampled_curves, axis=0), axis=0)
    elif method == 'mean':
        return np.mean(np.stack(resampled_curves, axis=0), axis=0)
    else:
        raise ValueError(f"Unknown method: {method}. Available options: 'median', 'mean'")

def grid_search(graph, eta_grid, lam_grid, empirical_curve, R, early_phase_hours=None, show_progress=True):
    """
    グリッドサーチ（立ち上がり区間を重視可能）
    
    Args:
        graph: グラフ
        eta_grid: etaの探索グリッド
        lam_grid: lamの探索グリッド
        empirical_curve: 観測曲線
        R: シミュレーション試行回数
        early_phase_hours: 立ち上がり区間の時間（時間単位）
        show_progress: False で tqdm を無効化（ブートストラップ内ループ用）
    
    Returns:
        (best_eta, best_lam, best_loss)
    """
    best_loss = float('inf')
    best_eta = 0.0
    best_lam = 0.0
    eta_iter = tqdm(eta_grid, desc="Grid search", leave=False) if show_progress else eta_grid
    for eta in eta_iter:
        for lam in lam_grid:
            loss = loss_func(graph, eta, lam, empirical_curve, R=R, early_phase_hours=early_phase_hours)
            if loss < best_loss:
                best_loss = loss
                best_eta = eta
                best_lam = lam
    return best_eta, best_lam, best_loss

def fit_parameters(graph, empirical_curve, seed, R, eta_grid, lam_grid, early_phase_hours=None, verbose=True):
    """
    パラメータフィッティング（立ち上がり区間を重視可能）
    
    Args:
        graph: グラフ
        empirical_curve: 観測曲線
        seed: 乱数シード
        R: シミュレーション試行回数
        eta_grid: etaの探索グリッド
        lam_grid: lamの探索グリッド
        early_phase_hours: 立ち上がり区間の時間（時間単位）。指定した場合、この区間のみを評価
        verbose: False のときログとグリッド tqdm を抑制
    
    Returns:
        最適パラメータの辞書
    """
    random.seed(seed)
    np.random.seed(seed)
    show_progress = verbose

    if verbose:
        print("1st Grid search starts...")
        print(f"Range: eta={eta_grid}, lam={lam_grid}")
        if early_phase_hours is not None:
            print(f"立ち上がり区間のみ評価: {early_phase_hours}時間")
    eta_init, lam_init, best_loss = grid_search(
        graph, eta_grid, lam_grid, empirical_curve, R,
        early_phase_hours=early_phase_hours, show_progress=show_progress,
    )
    
    if verbose:
        print(f"1st Grid search result: eta={eta_init:.6f}, lam={lam_init:.6f}, loss={best_loss:.6f}")
    
    d_eta = 0.001
    d_lam = 0.01
    eta_grid = np.arange(eta_init - d_eta * 5, eta_init + d_eta * 5 + 1e-9, d_eta)
    lam_grid = np.arange(lam_init - d_lam * 5, lam_init + d_lam * 5 + 1e-9, d_lam)

    if verbose:
        print("2nd Grid search starts...")
        print(f"Range: eta={eta_grid}, lam={lam_grid}")
    best_eta, best_lam, best_loss = grid_search(
        graph, eta_grid, lam_grid, empirical_curve, R,
        early_phase_hours=early_phase_hours, show_progress=show_progress,
    )
    
    if verbose:
        print(f"2nd Grid search result: eta={best_eta:.6f}, lam={best_lam:.6f}, loss={best_loss:.6f}")

    return {    
        "eta": float(best_eta),
        "lam": float(best_lam),
        "loss": float(best_loss),
    }


def bootstrap_eta_lam(
    graph,
    observed: List[Tuple[Any, Any]],
    seed: int,
    R: int,
    eta_grid: np.ndarray,
    lam_grid: np.ndarray,
    early_phase_hours: Optional[float] = None,
    B: int = 0,
    alpha: float = 0.05,
    rng_seed: Optional[int] = None,
    store_samples: bool = True,
) -> Dict[str, Any]:
    """
    Nonparametric bootstrap over cascades: resample cascades with replacement,
    build the bootstrap mean curve, refit (eta, lam) each time.

    Point estimates (eta, lam, loss) are from fitting the full-data mean curve.
    When B > 0, eta_ci_* / lam_ci_* are percentile intervals over bootstrap refits.
    """
    emp_curve_mean = representive_curve(observed, method='mean')
    best = fit_parameters(
        graph, emp_curve_mean, seed, R, eta_grid, lam_grid,
        early_phase_hours=early_phase_hours, verbose=True,
    )
    out: Dict[str, Any] = dict(best)
    out["n_bootstrap"] = int(B)
    out["alpha"] = float(alpha)

    if B <= 0:
        out["eta_ci_low"] = out["eta_ci_high"] = best["eta"]
        out["lam_ci_low"] = out["lam_ci_high"] = best["lam"]
        if store_samples:
            out["eta_bootstrap"] = np.array([])
            out["lam_bootstrap"] = np.array([])
        return out

    n = len(observed)
    if n < 30:
        print(f"Warning: only {n} cascades; bootstrap intervals may be unstable.")

    rng = np.random.default_rng(rng_seed if rng_seed is not None else seed + 9999)
    eta_samples: List[float] = []
    lam_samples: List[float] = []

    for b in tqdm(range(B), desc="Bootstrap refits"):
        idx = rng.integers(0, n, size=n)
        boot_obs = [observed[int(i)] for i in idx]
        emp_b = representive_curve(boot_obs, method='mean')
        boot_seed = seed + 1000 * (b + 1)
        fit_b = fit_parameters(
            graph, emp_b, boot_seed, R, eta_grid, lam_grid,
            early_phase_hours=early_phase_hours, verbose=False,
        )
        eta_samples.append(fit_b["eta"])
        lam_samples.append(fit_b["lam"])

    pct_lo = 100.0 * (alpha / 2.0)
    pct_hi = 100.0 * (1.0 - alpha / 2.0)
    out["eta_ci_low"] = float(np.percentile(eta_samples, pct_lo))
    out["eta_ci_high"] = float(np.percentile(eta_samples, pct_hi))
    out["lam_ci_low"] = float(np.percentile(lam_samples, pct_lo))
    out["lam_ci_high"] = float(np.percentile(lam_samples, pct_hi))
    if store_samples:
        out["eta_bootstrap"] = np.asarray(eta_samples, dtype=float)
        out["lam_bootstrap"] = np.asarray(lam_samples, dtype=float)

    print(
        f"Bootstrap (B={B}, alpha={alpha}): eta in [{out['eta_ci_low']:.5f}, {out['eta_ci_high']:.5f}], "
        f"lam in [{out['lam_ci_low']:.4f}, {out['lam_ci_high']:.4f}]"
    )
    return out


def get_seed_users_twitter(min_cascade_size=100):
    cascade_file = 'data/twitter_diffusion_dataset/cascade_all.txt'
    seed_users = []
    with open(cascade_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            user_times = line.split()
            if len(user_times) == 0:
                continue
            first_user = user_times[0].split(",")[0]
            if len(user_times) < min_cascade_size:
                continue
            seed_users.append(first_user)
    seed_user_counts = Counter(seed_users)
    seed_user_counts = sorted(seed_user_counts.items(), key=lambda x: (-x[1], x[0]))
    filtered_seed_users = [user for user, _ in seed_user_counts]
    
    return filtered_seed_users

def load_graph(graph_name, **kwargs):
    nikolov_graph = Nikolov_susceptibility_graph()
    suscep = [nikolov_graph.nodes[node]['suscep'] for node in nikolov_graph.nodes]
    np.random.seed(42)
    suscep_permuted = np.random.permutation(suscep)

    if graph_name == 'test':
        G = nx.erdos_renyi_graph(1000, 0.1, seed=42)
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['suscep'] = suscep_permuted[i]
        return G
    elif graph_name == 'twitter':
        path = f'data/twitter_diffusion_dataset/edges.txt'
        G = nx.DiGraph()
        with open(path, 'r') as f:
            for line in f:
                source, target = line.strip().split(',')
                G.add_edge(source, target)
        G.graph['graph_name'] = 'twitter'
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['suscep'] = suscep_permuted[i]
        return G
    else:
        raise ValueError(f"Unknown graph name: {graph_name}. Available options: 'test'")

def get_cascade_cumulative_counts(min_cascade_size=100, max_hours=744):
    cascade_file = 'data/twitter_diffusion_dataset/cascade_all.txt'
    
    if not os.path.exists(cascade_file):
        raise FileNotFoundError(f"Cascade file not found: {cascade_file}")
    
    observed = []
    
    print(f"Twitter diffusion dataset cascade data loading: {cascade_file}")
    
    with open(cascade_file, 'r') as f:
        for line_num, line in enumerate(tqdm(f, desc="Cascade processing")):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if not parts:
                continue
                
            user_times = []
            for part in parts:
                if ',' in part:
                    uid, retweet_time = part.split(',', 1)
                    try:
                        retweet_time = float(retweet_time)
                        user_times.append((uid, retweet_time))
                    except ValueError:
                        continue
            
            if len(user_times) < min_cascade_size:
                continue
                
            user_times.sort(key=lambda x: x[1])
            
            first_time = user_times[0][1]
            elapsed_times_millis = [t - first_time for _, t in user_times]
            elapsed_times_hours = [t / (1000 * 60 * 60) for t in elapsed_times_millis]  # millis to hours
            
            filtered_times = [t for t in elapsed_times_hours if t <= max_hours] # <= 744 hours (1 month)
            filtered_counts = list(range(1, len(filtered_times) + 1))
            
            if len(filtered_counts) >= min_cascade_size:
                observed.append((filtered_times, filtered_counts))
    
    print(f"Processing completed: {len(observed)} cascades obtained")
    return observed

def get_best_parameters():
    return {'eta': 0.026, 'lam': 0.25}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit eta, lam to Twitter cascades; optional bootstrap CIs.")
    parser.add_argument("--fit", action="store_true", help="Run grid search on full-data mean curve (required for bootstrap)")
    parser.add_argument("--bootstrap", type=int, default=0, metavar="B", help="Number of bootstrap refits (0 = point estimate only)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for fitting / bootstrap resampling")
    parser.add_argument("--rng-seed", type=int, default=None, help="Separate RNG seed for resampling cascades (default: seed+9999)")
    parser.add_argument("--R", type=int, default=100, help="CTIC replications per loss evaluation")
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided CI level for bootstrap percentiles")
    parser.add_argument("--early-phase", type=float, default=None, dest="early_phase", help="If set, loss uses first N hours only")
    parser.add_argument("--no-store-samples", action="store_true", help="Do not store eta_bootstrap / lam_bootstrap arrays")
    args = parser.parse_args()

    G = load_graph('twitter')
    print(G)
    print(len(G.nodes()), len(G.edges()))

    observed = get_cascade_cumulative_counts(min_cascade_size=100, max_hours=100)
    grid = np.arange(0.0, 100 + 1e-9, 1)
    emp_curve_mean = representive_curve(observed, method='mean')

    eta_grid = np.array([0.02, 0.025, 0.03])
    lam_grid = np.array([0.2, 0.3, 0.4, 0.5])

    if args.fit:
        best = bootstrap_eta_lam(
            G,
            observed,
            seed=args.seed,
            R=args.R,
            eta_grid=eta_grid,
            lam_grid=lam_grid,
            early_phase_hours=args.early_phase,
            B=args.bootstrap,
            alpha=args.alpha,
            rng_seed=args.rng_seed,
            store_samples=not args.no_store_samples,
        )
    else:
        if args.bootstrap > 0:
            print("Note: --bootstrap ignored without --fit; using get_best_parameters().")
        best = dict(get_best_parameters())
        best["n_bootstrap"] = 0
        best["alpha"] = args.alpha
        best["eta_ci_low"] = best["eta_ci_high"] = best["eta"]
        best["lam_ci_low"] = best["lam_ci_high"] = best["lam"]

    print(best)

    sim_curve = get_sim_curve(G, best["eta"], best["lam"], R=args.R, base_seed=123)

    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 6), dpi=300)
    for (times, counts) in observed:
        resampled_curve = resample_pc(times, counts, grid)
        plt.plot(grid, resampled_curve, '-', color='gray', linewidth=1, alpha=0.5)
    plt.plot(grid, emp_curve_mean, 'k-', linewidth=3, label='Mean')
    if best.get("n_bootstrap", 0) > 0:
        leg = (
            rf"Estimated ($\eta={best['eta']:.3f}$ [{best['eta_ci_low']:.3f}, {best['eta_ci_high']:.3f}], "
            rf"$\lambda={best['lam']:.2f}$ [{best['lam_ci_low']:.2f}, {best['lam_ci_high']:.2f}]$)"
        )
    else:
        leg = rf"Estimated ($\eta={best['eta']:.3f}$, $\lambda={best['lam']:.2f}$)"
    plt.plot(grid, sim_curve, 'r--', linewidth=3, label=leg)
    plt.axvline(x=48, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Time (hours)', fontsize=20)
    plt.ylabel('Cumulative Retweet Count', fontsize=20)
    plt.xlim(0, 100)
    plt.ylim(1, 3000)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(frameon=False, fontsize=14, loc='upper left')
    plt.yscale('log')
    plt.savefig('results/estimated_eta_lam.png')
    plt.show()


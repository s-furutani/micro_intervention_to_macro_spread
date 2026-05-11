import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import tqdm
from scipy.sparse.linalg import LinearOperator

import util
from load_graph import Nikolov_susceptibility_graph, randomized_nikolov_graph
from prebunking_targets import select_prebunking_targets


def prebunking_targets(graph, delta_pre, susceptibility, target_selection='random'):
    """旧 API 互換ラッパ。

    乱数挙動も旧実装と同じく random.seed(42) / np.random.seed(42) を行ってから、
    共通実装 `prebunking_targets.select_prebunking_targets` に委譲する。
    """
    N = len(graph.nodes())
    k = max(1, int(N * delta_pre))
    seed_nodes = util.get_seed_users(graph, seed_mode='single_largest_degree')
    inactive_nodes = set(graph.nodes()) - set(seed_nodes)

    random.seed(42)
    np.random.seed(42)
    return select_prebunking_targets(
        graph,
        k=k,
        strategy=target_selection,
        seed_nodes=seed_nodes,
        susceptibility=susceptibility,
        inactive_nodes=inactive_nodes,
        rng=None,
    )



def qmf_threshold_eta_linearop(A_csr: sp.csr_matrix, s: np.ndarray,
                               tol=1e-6, maxiter=5000):
    n = A_csr.shape[0]

    def matvec(x):
        # (A @ diag(s)) x = A @ (s * x)
        return A_csr @ (s * x)

    M_op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)

    vals, _ = spla.eigs(M_op, k=1, which='LM', tol=tol, maxiter=maxiter)
    lambda_max = float(np.real(vals[0]))
    eta_c = 1.0 / lambda_max
    return eta_c, lambda_max

def qmf_lambda_max(A_csr, s_vec, tol=1e-6, maxiter=5000):
    n = A_csr.shape[0]
    def matvec(x):
        return A_csr @ (s_vec * x)  # A @ diag(s) @ x
    M_op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    vals, _ = spla.eigs(M_op, k=1, which='LM', tol=tol, maxiter=maxiter)
    return float(np.real(vals[0]))

def s_after_targeted_prebunking(graph, s_dict, targets, eps):
    s_new = s_dict.copy()
    for v in targets:
        s_new[v] = (1.0 - eps) * s_new[v]
    s_new = np.array([s_new[v] for v in graph.nodes()])
    return s_new

def critical_eps_for_delta(A_csr, s, s_dict, graph, eta, delta, strategy,
                           tol_eps=1e-3, tol_eigs=1e-6, maxiter=5000):
    # 1) δ固定でターゲット固定
    targets = prebunking_targets(graph, delta, s_dict, target_selection=strategy)
    targets = set(targets)

    # 2) f(eps) = lambda_max - 1/eta
    inv_eta = 1.0 / eta

    def f(eps):
        s_new = s_after_targeted_prebunking(graph, s_dict, targets, eps)
        lam = qmf_lambda_max(A_csr, s_new, tol=tol_eigs, maxiter=maxiter)
        return lam - inv_eta, lam

    f0, lam0 = f(0.0)
    f1, lam1 = f(1.0)

    if f0 <= 0:
        # 介入なしで既に臨界以下
        return 0.0, lam0, "already_subcritical"
    if f1 > 0:
        # 最大介入でも臨界に届かない
        return None, lam1, "cannot_reach"

    # 3) 二分探索
    lo, hi = 0.0, 1.0
    lam_mid = None
    while (hi - lo) > tol_eps:
        mid = 0.5 * (lo + hi)
        fm, lam_mid = f(mid)
        if fm > 0:
            lo = mid
        else:
            hi = mid

    # hiが「臨界に入る最小のeps」に近い
    return hi, lam_mid, "ok"


def main():
    G = Nikolov_susceptibility_graph()
    G.graph['graph_name'] = 'nikolov'
    A = sp.csr_matrix(nx.adjacency_matrix(G))
    s = np.array([G.nodes[node]['suscep'] for node in G.nodes()])
    s_dict = {node: G.nodes[node]['suscep'] for node in G.nodes()}
    start_time = time.time()
    eta_c, lambda_max = qmf_threshold_eta_linearop(A, s)
    end_time = time.time()
    print(f"eta_c: {eta_c:.3f}, lambda_max: {lambda_max:.3f}")
    print(f"Time taken: {end_time - start_time} seconds")

    print('Critical eps for delta')
    delta_range = np.arange(0.1, 1.005, 0.01)
    target_strategies = ['high_degree', 'high_susceptible', 'cocoon', 'random']
    for target_strategy in target_strategies:
        print(target_strategy)
        eta_pred = 0.026
        eps_c_list = []
        start_time = time.time()
        for delta_pre in tqdm.tqdm(delta_range):
            hi, lam_mid, status = critical_eps_for_delta(
                A, s, s_dict, G, eta_pred, delta_pre, strategy=target_strategy
            )
            eps_c_list.append(hi)
        print(eps_c_list)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()


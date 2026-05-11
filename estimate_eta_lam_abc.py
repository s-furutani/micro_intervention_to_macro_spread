"""
ABC-SMC による eta, lam の事後分布推定
========================================

近似ベイズ計算 (Approximate Bayesian Computation) の Sequential Monte Carlo
版を、外部依存を増やさず軽量に自前実装する。

参考:
  - Sisson, Fan & Tanaka (2007) Sequential Monte Carlo without likelihoods.
    PNAS, 104(6).
  - Toni, Welch, Strelkowa, Ipsen & Stumpf (2009) ABC scheme for parameter
    inference and model selection in dynamical systems. J. R. Soc. Interface.
  - Beaumont, Cornuet, Marin & Robert (2009) Adaptive ABC. Biometrika.

設計の要点:
  - 事前分布: log-normal（log 空間での Gaussian）。デフォルトでは現在の
    グリッド最適点 (0.026, 0.25) を中心に sigma_log=1.0 の弱情報事前。
  - シミュレータ: 既存の get_sim_curve（CTIC を R 回実行して平均曲線）。
  - 距離: 既存の loss_func（観測平均曲線との Euclidean 距離）。
  - 摂動カーネル: 多変量ガウス（重み付き経験共分散の 2 倍）。
  - tolerance: ε_t は前世代距離の所定分位点で適応的に下げる。

使い方:
    # まず軽め
    python estimate_eta_lam_abc.py --R 30 --n-particles 50 --n-pops 3

    # 本番（caffeinate 推奨）
    caffeinate -i python estimate_eta_lam_abc.py \
        --R 100 --n-particles 200 --n-pops 5 --quantile 0.5
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 既存モジュールの関数を再利用（破壊的変更なし）
from estimate_eta_lam import (
    get_best_parameters,
    get_cascade_cumulative_counts,
    get_sim_curve,
    load_graph,
    loss_func,
    representive_curve,
    resample_pc,
)

plt.rcParams["font.family"] = "Arial"


# =====================================================================
# 事前分布: log-normal (= Gaussian in log-space)
# =====================================================================

def prior_log_pdf(
    log_theta: np.ndarray, mu_log: np.ndarray, sigma_log: np.ndarray
) -> np.ndarray:
    """log 空間で Gaussian 事前の対数密度を返す。

    Args:
        log_theta: shape (d,) または (N, d)
        mu_log:    shape (d,) （log 空間での平均）
        sigma_log: shape (d,) （log 空間での標準偏差、対角共分散を仮定）

    Returns:
        scalar (1点) または shape (N,) の log 密度
    """
    arr = np.atleast_2d(log_theta)
    diffs = (arr - mu_log) / sigma_log
    log_pdf = (
        -0.5 * np.sum(diffs ** 2, axis=1)
        - np.sum(np.log(sigma_log))
        - 0.5 * arr.shape[1] * np.log(2.0 * np.pi)
    )
    return log_pdf if arr.shape[0] > 1 else float(log_pdf[0])


# =====================================================================
# 距離関数 (シミュレータ + 距離をまとめたラッパ)
# =====================================================================

def distance(
    graph,
    log_theta: np.ndarray,
    emp_curve: np.ndarray,
    R: int,
    early_phase_hours: Optional[float],
) -> float:
    """log_theta を実空間に戻して loss_func を呼ぶ。"""
    eta = float(np.exp(log_theta[0]))
    lam = float(np.exp(log_theta[1]))
    return float(
        loss_func(graph, eta, lam, emp_curve, R=R, early_phase_hours=early_phase_hours)
    )


# =====================================================================
# 重み付き分位点（事後信用区間用）
# =====================================================================

def weighted_quantile(
    values: np.ndarray, quantiles, weights: np.ndarray
) -> np.ndarray:
    """重み付き分位点を線形補間で計算する。"""
    values = np.asarray(values)
    weights = np.asarray(weights)
    quantiles = np.atleast_1d(quantiles)

    sorter = np.argsort(values)
    values_s = values[sorter]
    weights_s = weights[sorter]
    cum = np.cumsum(weights_s) - 0.5 * weights_s
    cum /= cum[-1]
    return np.interp(quantiles, cum, values_s)


# =====================================================================
# ABC-SMC 本体
# =====================================================================

def abc_smc(
    graph,
    emp_curve: np.ndarray,
    R: int,
    n_particles: int = 100,
    n_pops: int = 4,
    quantile: float = 0.5,
    eps_min: Optional[float] = None,
    early_phase_hours: Optional[float] = None,
    mu_log: Optional[np.ndarray] = None,
    sigma_log: Optional[np.ndarray] = None,
    eta_bounds: Tuple[float, float] = (1e-5, 5.0),
    lam_bounds: Tuple[float, float] = (1e-5, 50.0),
    seed: int = 12345,
    max_attempts: int = 2000,
) -> List[Dict[str, Any]]:
    """ABC-SMC サンプラ。

    返り値: 世代ごとの粒子・重み・距離・εなどを格納した dict のリスト。
    """
    if mu_log is None or sigma_log is None:
        raise ValueError("mu_log と sigma_log は必須")

    rng = np.random.default_rng(seed)
    log_lower = np.array([np.log(eta_bounds[0]), np.log(lam_bounds[0])])
    log_upper = np.array([np.log(eta_bounds[1]), np.log(lam_bounds[1])])
    d_dim = len(mu_log)

    populations: List[Dict[str, Any]] = []

    # ---------- Pop 0: 事前分布から N 粒子サンプル + シミュレーション ----------
    print(f"[Pop 0] Sampling {n_particles} particles from prior + simulating...")
    particles_log: List[np.ndarray] = []
    distances: List[float] = []
    n_sims_pop0 = 0
    pbar = tqdm(total=n_particles, desc="Pop 0")
    while len(particles_log) < n_particles:
        cand = rng.normal(mu_log, sigma_log)
        n_sims_pop0 += 1
        # ハード境界の外に出たものは却下（事前分布のテイル切り）
        if np.any(cand < log_lower) or np.any(cand > log_upper):
            continue
        d = distance(graph, cand, emp_curve, R, early_phase_hours)
        particles_log.append(cand)
        distances.append(d)
        pbar.update(1)
    pbar.close()

    particles_arr = np.array(particles_log)
    dist_arr = np.array(distances)
    weights = np.ones(n_particles) / n_particles

    populations.append(
        {
            "particles_log": particles_arr.copy(),
            "weights": weights.copy(),
            "distances": dist_arr.copy(),
            "epsilon": float(np.max(dist_arr)),
            "n_sims": n_sims_pop0,
        }
    )
    print(
        f"  Pop 0 done: dist range = [{dist_arr.min():.4f}, {dist_arr.max():.4f}], "
        f"sims = {n_sims_pop0}"
    )

    # ---------- Pop 1.. : 摂動カーネルで提案 + 受理 ----------
    for t in range(1, n_pops):
        eps_t = float(np.quantile(dist_arr, quantile))
        if eps_min is not None and eps_t < eps_min:
            print(f"[Pop {t}] eps_t={eps_t:.4f} < eps_min={eps_min}; stopping.")
            break
        print(f"[Pop {t}] eps_t = {eps_t:.4f}")

        # 摂動カーネル: 重み付き共分散の 2 倍 (Beaumont 2009 の OLCM 簡易版)
        weighted_mean = np.sum(weights[:, None] * particles_arr, axis=0)
        diffs = particles_arr - weighted_mean
        cov = (weights[:, None] * diffs).T @ diffs
        cov_kernel = 2.0 * cov + 1e-9 * np.eye(d_dim)
        inv_cov = np.linalg.inv(cov_kernel)
        _, logdet = np.linalg.slogdet(2.0 * np.pi * cov_kernel)

        new_particles = np.zeros_like(particles_arr)
        new_distances = np.zeros(n_particles)
        n_sims = 0
        last_cand = None
        last_d = None

        pbar = tqdm(total=n_particles, desc=f"Pop {t}")
        for i in range(n_particles):
            accepted = False
            for _ in range(max_attempts):
                # 前世代から重み付きで一粒選び摂動
                idx = rng.choice(n_particles, p=weights)
                cand = rng.multivariate_normal(particles_arr[idx], cov_kernel)
                n_sims += 1
                if np.any(cand < log_lower) or np.any(cand > log_upper):
                    continue
                d = distance(graph, cand, emp_curve, R, early_phase_hours)
                last_cand, last_d = cand, d
                if d < eps_t:
                    new_particles[i] = cand
                    new_distances[i] = d
                    accepted = True
                    break
            if not accepted:
                # 受理失敗。粒子退化を避けるためフォールバックで最後の候補を採用
                print(
                    f"  Warning: particle {i} not accepted in {max_attempts} attempts "
                    f"(last d={last_d:.4f}, eps_t={eps_t:.4f})"
                )
                new_particles[i] = last_cand
                new_distances[i] = last_d
            pbar.update(1)
        pbar.close()

        # 重み更新: w_i ∝ π(θ'_i) / Σ_j w_j K(θ'_i | θ_j^(t-1))
        log_prior_new = prior_log_pdf(new_particles, mu_log, sigma_log)
        new_weights = np.zeros(n_particles)
        for i in range(n_particles):
            diffs_i = particles_arr - new_particles[i][None, :]
            quad = np.einsum("ij,jk,ik->i", diffs_i, inv_cov, diffs_i)
            log_kernel = -0.5 * quad - 0.5 * logdet
            log_w_plus_k = np.log(weights + 1e-300) + log_kernel
            max_lwk = float(np.max(log_w_plus_k))
            log_denom = max_lwk + np.log(np.sum(np.exp(log_w_plus_k - max_lwk)))
            new_weights[i] = float(np.exp(log_prior_new[i] - log_denom))
        new_weights /= new_weights.sum()

        # 状態更新
        particles_arr = new_particles
        dist_arr = new_distances
        weights = new_weights

        ess = float(1.0 / np.sum(weights ** 2))
        accept_rate = n_particles / max(n_sims, 1)
        print(
            f"  Pop {t} done: ESS = {ess:.1f} / {n_particles}, "
            f"acc.rate = {accept_rate:.3f}, sims = {n_sims}"
        )

        populations.append(
            {
                "particles_log": particles_arr.copy(),
                "weights": weights.copy(),
                "distances": dist_arr.copy(),
                "epsilon": eps_t,
                "n_sims": n_sims,
                "ess": ess,
                "acc_rate": accept_rate,
            }
        )

    return populations


# =====================================================================
# 事後分布の要約
# =====================================================================

def posterior_summary(
    populations: List[Dict[str, Any]], alpha: float = 0.05
) -> Dict[str, Any]:
    """最終世代の重み付き粒子から事後平均・中央値・95%信用区間を返す。

    点推定は事後平均を採用（多峰でない場合に安定）。
    """
    last = populations[-1]
    pl = last["particles_log"]
    w = last["weights"]

    eta = np.exp(pl[:, 0])
    lam = np.exp(pl[:, 1])

    eta_q = weighted_quantile(eta, [alpha / 2, 0.5, 1 - alpha / 2], w)
    lam_q = weighted_quantile(lam, [alpha / 2, 0.5, 1 - alpha / 2], w)

    return {
        "eta": float(np.sum(w * eta)),  # 事後平均
        "lam": float(np.sum(w * lam)),
        "eta_median": float(eta_q[1]),
        "lam_median": float(lam_q[1]),
        "eta_ci_low": float(eta_q[0]),
        "eta_ci_high": float(eta_q[2]),
        "lam_ci_low": float(lam_q[0]),
        "lam_ci_high": float(lam_q[2]),
        "alpha": float(alpha),
        "n_particles": int(len(w)),
        "n_populations": int(len(populations)),
        "epsilon_final": float(last["epsilon"]),
        "n_bootstrap": 0,  # 既存プロット側との互換のためダミー
    }


# =====================================================================
# main
# =====================================================================

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ABC-SMC による eta, lam の事後分布推定"
    )
    # シミュレーション・粒子設定
    p.add_argument("--R", type=int, default=50, help="距離評価あたりの CTIC 回数")
    p.add_argument("--n-particles", type=int, default=100)
    p.add_argument("--n-pops", type=int, default=4, help="世代数（pop 0 含む）")
    p.add_argument(
        "--quantile", type=float, default=0.5,
        help="ε_t を前世代距離の何分位点に置くか（小さいほど強く絞る）",
    )
    p.add_argument("--eps-min", type=float, default=None, help="ε がこの値を下回ったら停止")
    p.add_argument("--alpha", type=float, default=0.05, help="信用区間の両側有意水準")
    p.add_argument("--early-phase", type=float, default=None, dest="early_phase")
    # 事前分布（log-normal）
    p.add_argument("--prior-mu-eta", type=float, default=0.026, help="事前中心 (実空間)")
    p.add_argument("--prior-mu-lam", type=float, default=0.25)
    p.add_argument(
        "--prior-sigma-log-eta", type=float, default=1.0,
        help="log 空間での事前標準偏差（1.0 で 1σ ≒ 倍率 e）",
    )
    p.add_argument("--prior-sigma-log-lam", type=float, default=1.0)
    # ハード境界（安全弁）
    p.add_argument("--eta-min", type=float, default=1e-5)
    p.add_argument("--eta-max", type=float, default=5.0)
    p.add_argument("--lam-min", type=float, default=1e-5)
    p.add_argument("--lam-max", type=float, default=50.0)
    # その他
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-attempts", type=int, default=2000)
    p.add_argument("--out-suffix", type=str, default="abc")
    return p


def main() -> None:
    args = _make_parser().parse_args()

    # ------------- データロード -------------
    G = load_graph("twitter")
    print(G)
    print(len(G.nodes()), len(G.edges()))

    observed = get_cascade_cumulative_counts(min_cascade_size=100, max_hours=100)
    grid = np.arange(0.0, 100 + 1e-9, 1)
    emp_curve_mean = representive_curve(observed, method="mean")

    # ------------- 事前分布パラメタ -------------
    mu_log = np.array(
        [np.log(args.prior_mu_eta), np.log(args.prior_mu_lam)], dtype=float
    )
    sigma_log = np.array(
        [args.prior_sigma_log_eta, args.prior_sigma_log_lam], dtype=float
    )
    print(f"Prior log-normal: mu_log = {mu_log}, sigma_log = {sigma_log}")
    print(
        f"  -> 95% prior intervals: "
        f"eta in [{np.exp(mu_log[0] - 1.96 * sigma_log[0]):.4f}, "
        f"{np.exp(mu_log[0] + 1.96 * sigma_log[0]):.4f}], "
        f"lam in [{np.exp(mu_log[1] - 1.96 * sigma_log[1]):.3f}, "
        f"{np.exp(mu_log[1] + 1.96 * sigma_log[1]):.3f}]"
    )

    # ------------- ABC-SMC 実行 -------------
    populations = abc_smc(
        G,
        emp_curve_mean,
        R=args.R,
        n_particles=args.n_particles,
        n_pops=args.n_pops,
        quantile=args.quantile,
        eps_min=args.eps_min,
        early_phase_hours=args.early_phase,
        mu_log=mu_log,
        sigma_log=sigma_log,
        eta_bounds=(args.eta_min, args.eta_max),
        lam_bounds=(args.lam_min, args.lam_max),
        seed=args.seed,
        max_attempts=args.max_attempts,
    )

    # ------------- 事後要約 -------------
    summary = posterior_summary(populations, alpha=args.alpha)
    print("=== Posterior summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # ------------- 事後平均でフィット曲線描画 -------------
    sim_curve = get_sim_curve(
        G, summary["eta"], summary["lam"], R=args.R, base_seed=123
    )

    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 6), dpi=300)
    for (times, counts) in observed:
        rc = resample_pc(times, counts, grid)
        plt.plot(grid, rc, "-", color="gray", linewidth=1, alpha=0.5)
    plt.plot(grid, emp_curve_mean, "k-", linewidth=3, label="Mean")
    leg = (
        rf"ABC ($\eta={summary['eta']:.4f}$ "
        rf"[{summary['eta_ci_low']:.4f}, {summary['eta_ci_high']:.4f}], "
        rf"$\lambda={summary['lam']:.3f}$ "
        rf"[{summary['lam_ci_low']:.3f}, {summary['lam_ci_high']:.3f}])"
    )
    plt.plot(grid, sim_curve, "r--", linewidth=3, label=leg)
    plt.axvline(x=48, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Time (hours)", fontsize=20)
    plt.ylabel("Cumulative Retweet Count", fontsize=20)
    plt.xlim(0, 100)
    plt.ylim(1, 3000)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.legend(frameon=False, fontsize=14, loc="upper left")
    plt.yscale("log")
    out_path = f"results/estimated_eta_lam_{args.out_suffix}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    # ------------- 周辺事後ヒストグラム -------------
    last = populations[-1]
    eta_arr = np.exp(last["particles_log"][:, 0])
    lam_arr = np.exp(last["particles_log"][:, 1])
    w = last["weights"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    axes[0].hist(
        eta_arr, bins=20, weights=w, color="steelblue", edgecolor="white"
    )
    axes[0].axvline(
        summary["eta"], color="red", linestyle="--",
        label=f"mean={summary['eta']:.4f}",
    )
    axes[0].axvline(summary["eta_ci_low"], color="red", linestyle=":", alpha=0.5)
    axes[0].axvline(summary["eta_ci_high"], color="red", linestyle=":", alpha=0.5)
    axes[0].set_xlabel(r"$\eta$")
    axes[0].set_ylabel("posterior density (weighted)")
    axes[0].legend(frameon=False)

    axes[1].hist(
        lam_arr, bins=20, weights=w, color="darkorange", edgecolor="white"
    )
    axes[1].axvline(
        summary["lam"], color="red", linestyle="--",
        label=f"mean={summary['lam']:.3f}",
    )
    axes[1].axvline(summary["lam_ci_low"], color="red", linestyle=":", alpha=0.5)
    axes[1].axvline(summary["lam_ci_high"], color="red", linestyle=":", alpha=0.5)
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    posterior_path = f"results/posterior_eta_lam_{args.out_suffix}.png"
    fig.savefig(posterior_path)
    plt.close(fig)
    print(f"Saved: {posterior_path}")


if __name__ == "__main__":
    main()

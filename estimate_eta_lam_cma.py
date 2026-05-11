"""
CMA-ES + ブートストラップによる eta, lam の連続最適化と不確実性評価
============================================================

estimate_eta_lam.py のグリッド探索を CMA-ES に置き換える。
- eta, lam は正値なので log 空間で最適化（境界つき）
- 損失は既存 loss_func を流用（内部で CTIC を R 回回して平均曲線を比較）
- Common Random Numbers (CRN) は既存どおり base_seed=123 固定なので、
  同じ (eta, lam) を評価した場合は同一の損失が返る。これにより
  「カスケード再サンプリングの不確実性」のみをブートストラップが捕える。
- 信頼区間は percentile / basic (pivotal) から選択可能。

使い方の例:
    # 点推定のみ（CMA-ES で連続フィット）
    python estimate_eta_lam_cma.py --fit --R 100 --cma-maxiter 40

    # ブートストラップ CI つき（B 回再フィット）
    python estimate_eta_lam_cma.py --fit --R 100 --bootstrap 50 --ci-method percentile
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 既存モジュールの関数を再利用（破壊的変更を避ける）
from estimate_eta_lam import (
    get_best_parameters,
    get_cascade_cumulative_counts,
    get_sim_curve,
    load_graph,
    loss_func,
    representive_curve,
    resample_pc,
)

try:
    import cma
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "cma パッケージが必要です。以下でインストールしてください。\n"
        "  pip install cma"
    ) from e

plt.rcParams["font.family"] = "Arial"


# =====================================================================
# CMA-ES での連続最適化
# =====================================================================

def _obj_log(
    log_params: np.ndarray,
    graph,
    emp_curve: np.ndarray,
    R: int,
    early_phase_hours: Optional[float],
) -> float:
    """log 空間の実数ベクトルから損失を評価するラッパー。

    CMA-ES は無制約実数空間で動くため、eta, lam の正値性を保つために
    パラメータを log 変換して受け取る。
    """
    eta = float(np.exp(log_params[0]))
    lam = float(np.exp(log_params[1]))
    return float(
        loss_func(
            graph, eta, lam, emp_curve, R=R, early_phase_hours=early_phase_hours
        )
    )


def fit_cma(
    graph,
    emp_curve: np.ndarray,
    R: int,
    early_phase_hours: Optional[float] = None,
    x0_eta: float = 0.03,
    x0_lam: float = 0.3,
    sigma0: float = 0.5,
    maxiter: int = 50,
    popsize: Optional[int] = None,
    eta_bounds: Tuple[float, float] = (1e-4, 1.0),
    lam_bounds: Tuple[float, float] = (1e-3, 5.0),
    seed: int = 123,
    verbose: bool = True,
) -> Dict[str, Any]:
    """CMA-ES による (eta, lam) の連続最適化。

    Args:
        graph: 拡散先ネットワーク（networkx）
        emp_curve: 観測平均曲線（resampled, shape=(T,)）
        R: 損失評価あたりの CTIC シミュレーション回数
        early_phase_hours: 立ち上がり区間のみで損失評価するときの時間長
        x0_eta, x0_lam: 探索開始点（実空間の値。内部で log 化して渡す）
        sigma0: log 空間での初期探索幅（0.5 なら ~1.65 倍の範囲）
        maxiter: CMA-ES の最大世代数
        popsize: 1 世代あたりの個体数（None で自動）
        eta_bounds, lam_bounds: 実空間での探索範囲（log 化して境界に）
        seed: CMA-ES 内部の乱数シード
        verbose: True で CMA の進捗ログを出す

    Returns:
        dict: eta, lam（実空間の最良解）, loss, n_eval
    """
    # -> log 空間に変換して初期点と境界を与える
    log_x0 = [float(np.log(x0_eta)), float(np.log(x0_lam))]
    lower = [float(np.log(eta_bounds[0])), float(np.log(lam_bounds[0]))]
    upper = [float(np.log(eta_bounds[1])), float(np.log(lam_bounds[1]))]

    opts: Dict[str, Any] = {
        "maxiter": int(maxiter),
        "bounds": [lower, upper],
        "seed": int(seed),
        # verbose=-9 はログを極力抑える設定（cma のマニュアル参照）
        "verbose": 1 if verbose else -9,
        "verb_disp": 10 if verbose else 0,
        "verb_log": 0,
        "tolx": 1e-4,
        "tolfun": 1e-4,
    }
    if popsize is not None:
        opts["popsize"] = int(popsize)

    es = cma.CMAEvolutionStrategy(log_x0, sigma0, opts)

    # ask/tell ループで明示的に評価（関数引数を閉じ込めるため）
    while not es.stop():
        solutions = es.ask()
        losses = [
            _obj_log(np.asarray(s), graph, emp_curve, R, early_phase_hours)
            for s in solutions
        ]
        es.tell(solutions, losses)
        if verbose:
            es.disp()

    res = es.result
    log_eta, log_lam = res.xbest
    return {
        "eta": float(np.exp(log_eta)),
        "lam": float(np.exp(log_lam)),
        "loss": float(res.fbest),
        "n_eval": int(res.evaluations),
    }


# =====================================================================
# 信頼区間
# =====================================================================

def percentile_ci(samples: np.ndarray, alpha: float) -> Tuple[float, float]:
    """両側 (1-alpha) パーセンタイル区間"""
    lo = float(np.percentile(samples, 100.0 * alpha / 2.0))
    hi = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def basic_ci(
    samples: np.ndarray, point_est: float, alpha: float
) -> Tuple[float, float]:
    """Basic (pivotal) ブートストラップ区間

    [2*theta_hat - q_{1-a/2},  2*theta_hat - q_{a/2}]
    """
    p_lo, p_hi = percentile_ci(samples, alpha)
    return 2.0 * point_est - p_hi, 2.0 * point_est - p_lo


# =====================================================================
# ブートストラップ
# =====================================================================

def bootstrap_cma(
    graph,
    observed: List[Tuple[Any, Any]],
    R: int,
    B: int,
    alpha: float = 0.05,
    early_phase_hours: Optional[float] = None,
    ci_method: str = "percentile",
    rng_seed: int = 12345,
    cma_seed: int = 123,
    **cma_kwargs,
) -> Dict[str, Any]:
    """点推定 + カスケード単位の非パラメトリック・ブートストラップ。

    - 各反復で観測カスケードを「元と同じ本数 n」だけ復元抽出
    - その再サンプルから平均曲線を作り直し、CMA-ES で再フィット
    - B 個の (eta, lam) から信頼区間を構成

    Returns:
        dict: 点推定 + ci 下限/上限 + 標本配列
    """
    emp_curve = representive_curve(observed, method="mean")

    # -> 点推定
    point = fit_cma(
        graph,
        emp_curve,
        R=R,
        early_phase_hours=early_phase_hours,
        seed=cma_seed,
        verbose=True,
        **cma_kwargs,
    )

    out: Dict[str, Any] = dict(point)
    out["n_bootstrap"] = int(B)
    out["alpha"] = float(alpha)
    out["ci_method"] = ci_method

    if B <= 0:
        out["eta_ci_low"] = out["eta_ci_high"] = point["eta"]
        out["lam_ci_low"] = out["lam_ci_high"] = point["lam"]
        out["eta_bootstrap"] = np.array([])
        out["lam_bootstrap"] = np.array([])
        return out

    n = len(observed)
    if n < 30:
        print(f"Warning: only {n} cascades; bootstrap intervals may be unstable.")

    rng = np.random.default_rng(rng_seed)
    eta_samples: List[float] = []
    lam_samples: List[float] = []
    loss_samples: List[float] = []

    for b in tqdm(range(B), desc="CMA-ES bootstrap refits"):
        # -> カスケード単位で復元抽出
        idx = rng.integers(0, n, size=n)
        boot_obs = [observed[int(i)] for i in idx]
        emp_b = representive_curve(boot_obs, method="mean")
        # verbose=False で CMA の逐次ログを抑制
        fit_b = fit_cma(
            graph,
            emp_b,
            R=R,
            early_phase_hours=early_phase_hours,
            seed=cma_seed + b + 1,
            verbose=False,
            **cma_kwargs,
        )
        eta_samples.append(fit_b["eta"])
        lam_samples.append(fit_b["lam"])
        loss_samples.append(fit_b["loss"])

    eta_arr = np.asarray(eta_samples)
    lam_arr = np.asarray(lam_samples)

    if ci_method == "percentile":
        out["eta_ci_low"], out["eta_ci_high"] = percentile_ci(eta_arr, alpha)
        out["lam_ci_low"], out["lam_ci_high"] = percentile_ci(lam_arr, alpha)
    elif ci_method == "basic":
        out["eta_ci_low"], out["eta_ci_high"] = basic_ci(eta_arr, point["eta"], alpha)
        out["lam_ci_low"], out["lam_ci_high"] = basic_ci(lam_arr, point["lam"], alpha)
    else:
        raise ValueError(f"Unknown ci_method: {ci_method}")

    out["eta_bootstrap"] = eta_arr
    out["lam_bootstrap"] = lam_arr
    out["loss_bootstrap"] = np.asarray(loss_samples)

    print(
        f"CMA-ES Bootstrap (B={B}, alpha={alpha}, {ci_method}): "
        f"eta in [{out['eta_ci_low']:.5f}, {out['eta_ci_high']:.5f}], "
        f"lam in [{out['lam_ci_low']:.4f}, {out['lam_ci_high']:.4f}]"
    )
    return out


# =====================================================================
# main
# =====================================================================

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CMA-ES による eta, lam の連続最適化とブートストラップ信頼区間"
    )
    p.add_argument("--fit", action="store_true", help="CMA-ES で当てはめる（指定無しなら get_best_parameters() を流用）")
    p.add_argument("--bootstrap", type=int, default=0, metavar="B", help="ブートストラップ再フィット回数")
    p.add_argument("--R", type=int, default=100, help="損失評価あたりの CTIC シミュレーション回数")
    p.add_argument("--alpha", type=float, default=0.05, help="信頼区間の両側有意水準")
    p.add_argument("--ci-method", type=str, default="percentile", choices=["percentile", "basic"])
    p.add_argument("--early-phase", type=float, default=None, dest="early_phase")
    # CMA-ES ハイパラ
    p.add_argument("--cma-maxiter", type=int, default=40)
    p.add_argument("--cma-sigma", type=float, default=0.5, help="log 空間での初期探索幅")
    p.add_argument("--popsize", type=int, default=None, help="世代あたりの個体数（None で自動）")
    p.add_argument("--cma-seed", type=int, default=123)
    p.add_argument("--rng-seed", type=int, default=12345, help="ブートストラップ再サンプリング用 RNG")
    p.add_argument("--x0-eta", type=float, default=0.03)
    p.add_argument("--x0-lam", type=float, default=0.3)
    p.add_argument("--eta-min", type=float, default=1e-4)
    p.add_argument("--eta-max", type=float, default=1.0)
    p.add_argument("--lam-min", type=float, default=1e-3)
    p.add_argument("--lam-max", type=float, default=5.0)
    p.add_argument("--out-suffix", type=str, default="cma", help="出力図のファイル名接尾辞")
    return p


def main() -> None:
    args = _make_parser().parse_args()

    # -> グラフとカスケードをロード
    G = load_graph("twitter")
    print(G)
    print(len(G.nodes()), len(G.edges()))

    observed = get_cascade_cumulative_counts(min_cascade_size=100, max_hours=100)
    grid = np.arange(0.0, 100 + 1e-9, 1)
    emp_curve_mean = representive_curve(observed, method="mean")

    # -> 点推定（＋ CI）もしくはデフォルトパラメタ流用
    if args.fit:
        best = bootstrap_cma(
            G,
            observed,
            R=args.R,
            B=args.bootstrap,
            alpha=args.alpha,
            early_phase_hours=args.early_phase,
            ci_method=args.ci_method,
            rng_seed=args.rng_seed,
            cma_seed=args.cma_seed,
            x0_eta=args.x0_eta,
            x0_lam=args.x0_lam,
            sigma0=args.cma_sigma,
            maxiter=args.cma_maxiter,
            popsize=args.popsize,
            eta_bounds=(args.eta_min, args.eta_max),
            lam_bounds=(args.lam_min, args.lam_max),
        )
    else:
        if args.bootstrap > 0:
            print("Note: --fit 無しだと --bootstrap は無視されます。")
        best = dict(get_best_parameters())
        best["n_bootstrap"] = 0
        best["alpha"] = args.alpha
        best["ci_method"] = args.ci_method
        best["eta_ci_low"] = best["eta_ci_high"] = best["eta"]
        best["lam_ci_low"] = best["lam_ci_high"] = best["lam"]

    # 配列はプリントすると長いので別表示
    summary = {k: v for k, v in best.items() if not isinstance(v, np.ndarray)}
    print(summary)

    # -> 推定パラメタで再シミュレーションして観測と重ね書き
    sim_curve = get_sim_curve(G, best["eta"], best["lam"], R=args.R, base_seed=123)

    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 6), dpi=300)
    for (times, counts) in observed:
        resampled_curve = resample_pc(times, counts, grid)
        plt.plot(grid, resampled_curve, "-", color="gray", linewidth=1, alpha=0.5)
    plt.plot(grid, emp_curve_mean, "k-", linewidth=3, label="Mean")
    if best.get("n_bootstrap", 0) > 0:
        leg = (
            rf"CMA ($\eta={best['eta']:.4f}$ "
            rf"[{best['eta_ci_low']:.4f}, {best['eta_ci_high']:.4f}], "
            rf"$\lambda={best['lam']:.3f}$ "
            rf"[{best['lam_ci_low']:.3f}, {best['lam_ci_high']:.3f}])"
        )
    else:
        leg = rf"CMA ($\eta={best['eta']:.4f}$, $\lambda={best['lam']:.3f}$)"
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

    # -> ブートストラップ標本の簡易可視化
    if best.get("n_bootstrap", 0) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
        axes[0].hist(best["eta_bootstrap"], bins=20, color="steelblue", edgecolor="white")
        axes[0].axvline(best["eta"], color="red", linestyle="--", label="point")
        axes[0].set_xlabel(r"$\eta$")
        axes[0].set_ylabel("count")
        axes[0].legend(frameon=False)

        axes[1].hist(best["lam_bootstrap"], bins=20, color="darkorange", edgecolor="white")
        axes[1].axvline(best["lam"], color="red", linestyle="--", label="point")
        axes[1].set_xlabel(r"$\lambda$")
        axes[1].legend(frameon=False)

        axes[2].scatter(
            best["eta_bootstrap"], best["lam_bootstrap"],
            s=20, alpha=0.6, color="gray", edgecolor="k",
        )
        axes[2].scatter([best["eta"]], [best["lam"]], s=60, color="red", label="point", zorder=3)
        axes[2].set_xlabel(r"$\eta$")
        axes[2].set_ylabel(r"$\lambda$")
        axes[2].legend(frameon=False)
        fig.tight_layout()
        boot_path = f"results/bootstrap_eta_lam_{args.out_suffix}.png"
        fig.savefig(boot_path)
        plt.close(fig)
        print(f"Saved: {boot_path}")


if __name__ == "__main__":
    main()

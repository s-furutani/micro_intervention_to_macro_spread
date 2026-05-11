"""実験用の共通パラメータ・定数を集約するモジュール。

- 旧 `intervention_analysis.py` / `intervention_analysis_varying_eta.py` /
  `single_vs_combined.py` の `get_eta_lam()` を 1 か所に集約。
- `util.get_seed_users` 内にハードコードされていたシードユーザ ID 群も
  ここで定数として宣言する。
- `estimate_eta_lam.get_best_parameters()` の値（最良 η, λ）も同期させる。
"""

from __future__ import annotations

# Twitter cascade fitting で得た最良点（estimate_eta_lam で fit した値）。
BEST_ETA_NIKOLOV: float = 0.026
BEST_LAM_NIKOLOV: float = 0.25

# test グラフ（erdos_renyi）用のデフォルト値。
DEFAULT_ETA_TEST: float = 0.2
DEFAULT_LAM_TEST: float = 1.0

# Nikolov 系グラフ名（fit 済み eta, lam を共有する）。
NIKOLOV_GRAPHS: tuple[str, ...] = (
    "nikolov",
    "randomized_nikolov",
    "uniform_nikolov",
)

# get_seed_users で使う Nikolov 上の固定シード集合。
# - 単一最大次数: suscep=1.0, out_degree=2269 のノード
# - 中程度次数 3 ノード: 互いに離れた suscep=1.0 のノード
SEED_USER_SINGLE_LARGEST_DEGREE: int = 131989
SEED_USERS_MULTIPLE_MODERATE_DEGREE: list[int] = [36566, 7394, 108009]


def get_eta_lam(graph_name: str) -> tuple[float, float]:
    """グラフ名から既定の (eta, lam) を返す。

    Nikolov 系（randomized / uniform 含む）はフィット結果を共有する設計。
    test ほか未知グラフは人工的な既定値を返す。
    """
    if graph_name in NIKOLOV_GRAPHS:
        return BEST_ETA_NIKOLOV, BEST_LAM_NIKOLOV
    return DEFAULT_ETA_TEST, DEFAULT_LAM_TEST


def get_best_parameters() -> dict:
    """estimate_eta_lam 互換の dict を返す。"""
    return {"eta": BEST_ETA_NIKOLOV, "lam": BEST_LAM_NIKOLOV}

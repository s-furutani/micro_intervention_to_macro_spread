"""η を y 軸に取った介入ヒートマップ生成（CLI）。

(ε, η) の格子で prevalence を計算する。Prebunking は δ_pre=0.2、
Contextualization は φ_ctx=0.8 を固定値として使う。
本体は [intervention_analysis_common.py](intervention_analysis_common.py) に集約済み。
"""

from __future__ import annotations

import argparse

import util
from intervention_analysis_common import run_intervention_analysis


def main():
    parser = argparse.ArgumentParser(description="Create heatmap under intervention")
    parser.add_argument(
        "--intervention",
        "-i",
        type=str,
        required=True,
        choices=["prebunking", "contextualization", "nudging"],
        help="介入タイプを指定",
    )
    parser.add_argument(
        "--graph",
        "-g",
        type=str,
        default="test",
        choices=["test", "nikolov", "randomized_nikolov", "uniform_nikolov"],
        help="使用するグラフを指定 (デフォルト: test)",
    )
    parser.add_argument(
        "--simulations",
        "-s",
        type=int,
        default=100,
        help="シミュレーション回数 (デフォルト: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="乱数シード (デフォルト: 42)"
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        help="ヒートマップデータをnpyファイルとして保存",
    )
    parser.add_argument(
        "--target_selection",
        type=str,
        default="random",
        choices=["random", "high_degree", "high_susceptible", "cocoon"],
        help="介入ターゲット選択戦略 (デフォルト: random)",
    )
    _seed_choices = sorted(util.SEED_MODES | set(util.SEED_MODE_ALIASES.keys()))
    parser.add_argument(
        "--seed_mode",
        type=str,
        default="single_largest_degree",
        choices=_seed_choices,
        help=(
            "初期拡散ノード: single_largest_degree / multiple_moderate_degree / "
            "random10（single は前者の別名）"
        ),
    )
    args = parser.parse_args()
    seed_mode = util.normalize_seed_mode(args.seed_mode)

    run_intervention_analysis(
        intervention_type=args.intervention,
        axis_y="vary_eta",
        graph_name=args.graph,
        n_simulations=args.simulations,
        seed_base=args.seed,
        save_data=args.save_data,
        target_selection=args.target_selection,
        seed_mode=seed_mode,
    )


if __name__ == "__main__":
    main()

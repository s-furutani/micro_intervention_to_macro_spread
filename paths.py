"""results/ 配下の .npy ファイル名構築を 1 か所に集約するモジュール。

これまで [intervention_analysis.py] / [intervention_analysis_varying_eta.py] /
[single_vs_combined.py] / [plot_results.py] で個別に書かれていた
パス組み立て規則をここでまとめる。

ファイル名規則（不変）:
- 固定 η ヒートマップ:
    `results/{graph}/{stem}{seed_sfx}{eta_sfx}.npy`
- vary_eta ヒートマップ:
    `results/{graph}/{stem}{seed_sfx}_vary_eta.npy`
- single vs combined:
    `results/{graph}/compare_single_and_combined_intervention_{target}_{cmp_suf}{seed_sfx}{eta_sfx}{gamma_sfx}.npy`

ここで `stem` は介入名（prebunking_random / contextualization / nudging 等）。
"""

from __future__ import annotations

import os

# 既存の seed_mode 名 -> ファイル名サフィックス（先頭に _ あり）
_SEED_MODE_SUFFIX: dict[str, str] = {
    "single_largest_degree": "",
    "random10": "_seed10",
    "multiple_moderate_degree": "_seed_mod3",
}

# 旧 CLI などとの互換用略称
SEED_MODE_ALIASES: dict[str, str] = {"single": "single_largest_degree"}

# 有効な seed_mode 集合
SEED_MODES: frozenset[str] = frozenset(_SEED_MODE_SUFFIX.keys())


def normalize_seed_mode(mode: str | None) -> str:
    """略称を正規化し、内部 seed_mode 名を返す。"""
    if mode is None:
        return "single_largest_degree"
    m = str(mode).strip()
    m = SEED_MODE_ALIASES.get(m, m)
    if m not in SEED_MODES:
        raise ValueError(
            f"Unknown seed_mode {mode!r}. Use one of {sorted(SEED_MODES)} "
            f"or alias {sorted(SEED_MODE_ALIASES)}"
        )
    return m


def seed_mode_filename_suffix(seed_mode: str) -> str:
    """seed_mode のファイル名サフィックス（先頭に _ あり、デフォルトは空）。"""
    m = normalize_seed_mode(seed_mode)
    return _SEED_MODE_SUFFIX.get(m, "")


def eta_scale_suffix(eta_scale: str | float) -> str:
    """eta_scale のファイル名サフィックス（'_half' / '_double' / 空）。"""
    s = str(eta_scale).strip()
    if s in ("1", "1.0", ""):
        return ""
    if s in ("0.5", "half"):
        return "_half"
    if s in ("2", "2.0", "double"):
        return "_double"
    raise ValueError(f"eta_scale は '1' / '0.5' / '2' を想定: {eta_scale!r}")


def heatmap_stem(intervention_type: str, target_selection: str = "random") -> str:
    """ヒートマップ .npy のファイル先頭部分（拡張子・サフィックス前）を組み立てる。

    prebunking のみ target_selection を含む（旧スクリプトと同じ規則）。
    """
    if intervention_type == "prebunking":
        return f"prebunking_{target_selection}"
    return intervention_type


def heatmap_path_fixed_eta(
    graph_name: str,
    intervention_type: str,
    target_selection: str = "random",
    seed_mode: str = "single_largest_degree",
    eta_scale: str | float = "1",
) -> str:
    """固定 η ヒートマップ（intervention_analysis.py 系）の .npy パス。"""
    stem = heatmap_stem(intervention_type, target_selection)
    seed_sfx = seed_mode_filename_suffix(seed_mode)
    eta_sfx = eta_scale_suffix(eta_scale)
    fname = f"{stem}{eta_sfx}{seed_sfx}.npy"
    return os.path.join("results", graph_name, fname)


def heatmap_path_vary_eta(
    graph_name: str,
    intervention_type: str,
    target_selection: str = "random",
    seed_mode: str = "single_largest_degree",
) -> str:
    """vary_eta ヒートマップ（intervention_analysis_varying_eta.py 系）の .npy パス。"""
    stem = heatmap_stem(intervention_type, target_selection)
    seed_sfx = seed_mode_filename_suffix(seed_mode)
    fname = f"{stem}{seed_sfx}_vary_eta.npy"
    return os.path.join("results", graph_name, fname)


def single_vs_combined_path(
    graph_name: str,
    target_selection: str,
    *,
    improve_reach: bool = False,
    improve_strength: bool = False,
    seed_mode: str = "single_largest_degree",
    eta_scale: str | float = "1",
    gamma: float = 1.0,
) -> str:
    """single_vs_combined.py の .npy パス。

    改善モード suffix は base / improve_reach / improve_strength / improve_both。
    """
    if improve_reach and improve_strength:
        cmp_suf = "improve_both"
    elif improve_reach:
        cmp_suf = "improve_reach"
    elif improve_strength:
        cmp_suf = "improve_strength"
    else:
        cmp_suf = "base"

    seed_sfx = seed_mode_filename_suffix(seed_mode)
    eta_sfx = eta_scale_suffix(eta_scale)
    gamma_sfx = "" if float(gamma) == 1.0 else f"_gamma{float(gamma):g}"

    fname = (
        f"compare_single_and_combined_intervention_{target_selection}_{cmp_suf}"
        f"{seed_sfx}{eta_sfx}{gamma_sfx}.npy"
    )
    return os.path.join("results", graph_name, fname)

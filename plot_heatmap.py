"""ヒートマップ系プロット（介入解析の (eps, eta) / (eps, delta) 等）。

旧 `util.py` の `relative_suppression` と `plot_multiple_heatmaps` を分離。
互換性のため `util.py` 側で re-export している。
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm

import plot_style  # noqa: F401  rcParams を適用する副作用


def relative_suppression(heatmap_data):
    """各行を最左列で割った相対値を返す（介入なしを基準に正規化）。"""
    data_rel = np.array(
        [heatmap_data[i, :] / heatmap_data[i, 0] for i in range(heatmap_data.shape[0])]
    )
    return data_rel


# =====================================================================
# 内部ヘルパ
# =====================================================================
def _panel_subcaption_text(letter_index: int) -> str:
    """パネル番号 0→'(a)', 1→'(b)' …"""
    return f"({chr(ord('a') + letter_index)})"


# プレバンキング戦略 × δ の数値臨界曲線（事前に計算済みの値）。
# vary_eta ではない固定 η ヒートマップで、対応するパスに合わせて引く。
_CRITICAL_DELTA = np.arange(0.1, 1.005, 0.01)
_CRITICAL_EPS_RANDOM = [None] * 79 + [
    0.9892578125, 0.9794921875, 0.9716796875, 0.9580078125, 0.9541015625,
    0.94140625, 0.9326171875, 0.9228515625, 0.9150390625, 0.90625,
    0.89453125, 0.8828125,
]
_CRITICAL_EPS_DEGREE = [None, None] + [
    0.99609375, 0.98828125, 0.98046875, 0.974609375, 0.96875, 0.962890625,
    0.95703125, 0.9521484375, 0.947265625, 0.943359375, 0.939453125,
    0.9345703125, 0.9306640625, 0.927734375, 0.923828125, 0.919921875,
    0.9169921875, 0.9130859375, 0.9111328125, 0.9091796875, 0.90625,
    0.904296875, 0.90234375, 0.8994140625, 0.8984375, 0.896484375,
    0.8955078125, 0.89453125, 0.892578125, 0.8916015625, 0.8916015625,
    0.890625, 0.8896484375, 0.888671875, 0.888671875, 0.8876953125,
    0.8876953125, 0.88671875, 0.88671875, 0.88671875, 0.8857421875,
    0.8857421875, 0.8857421875, 0.884765625, 0.884765625, 0.884765625,
    0.884765625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625,
    0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625,
    0.8837890625, 0.8837890625, 0.8828125, 0.8828125, 0.8828125, 0.8828125,
    0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125,
    0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125,
    0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125,
    0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125,
    0.8828125, 0.8828125,
]
_CRITICAL_EPS_SUSCEP = [None] * 33 + [
    0.9716796875, 0.958984375, 0.9296875, 0.916015625, 0.9140625,
    0.9130859375, 0.91015625, 0.90625, 0.904296875, 0.90234375, 0.900390625,
    0.8984375, 0.896484375, 0.89453125, 0.892578125, 0.8916015625, 0.890625,
    0.8896484375, 0.888671875, 0.888671875, 0.8876953125, 0.88671875,
    0.8857421875, 0.8857421875, 0.884765625, 0.884765625, 0.884765625,
    0.884765625, 0.884765625, 0.8837890625, 0.8837890625, 0.8837890625,
    0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625,
    0.8837890625, 0.8837890625, 0.8828125, 0.8828125, 0.8828125, 0.8828125,
    0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125,
    0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125,
    0.8828125, 0.8828125, 0.8828125, 0.8828125,
]
_CRITICAL_EPS_COCOON = [None] * 68 + [
    0.994140625, 0.9912109375, 0.9833984375, 0.9794921875, 0.9765625,
    0.9697265625, 0.96484375, 0.9599609375, 0.94921875, 0.9423828125,
    0.9345703125, 0.931640625, 0.92578125, 0.921875, 0.91796875, 0.912109375,
    0.908203125, 0.900390625, 0.896484375, 0.890625, 0.8828125, 0.8828125,
    0.8828125,
]


def _critical_curve_xy(critical_eps_list: list) -> tuple[list, np.ndarray]:
    """事前計算の ε 列と δ グリッドを同じ長さに揃える（リスト長の取り違い対策）。"""
    n = min(len(critical_eps_list), len(_CRITICAL_DELTA))
    return critical_eps_list[:n], _CRITICAL_DELTA[:n]


def _vary_eta_critical_curve(intervention_type, lambda_max, delta_pre):
    """vary_eta の場合の (epsilon, eta_c) を返す。返り値は (eps_grid, eta_c)。"""
    epsilon_range = np.linspace(0.01, 1.0, 100)
    if intervention_type == "nudging":
        den = (1 - epsilon_range) * lambda_max
        eta_critical = np.divide(
            1.0,
            den,
            out=np.full_like(den, np.nan, dtype=float),
            where=np.abs(den) > 1e-15,
        )
        return epsilon_range, eta_critical
    if intervention_type == "prebunking":
        den = (1 - delta_pre * epsilon_range) * lambda_max
        eta_critical = np.divide(
            1.0,
            den,
            out=np.full_like(den, np.nan, dtype=float),
            where=np.abs(den) > 1e-15,
        )
        return epsilon_range, eta_critical
    return None, None


# =====================================================================
# メイン関数
# =====================================================================
def plot_multiple_heatmaps(
    row,
    col,
    path_list,
    relative_flag_list,
    title_list,
    diff_cmap="bwr",
    diff_vlim=0.3,
    plot_axline=False,
    plot_critical_curve=False,
    lambda_max=324.0259937461831,
    delta_pre=0.2,
    *,
    panel_subcaptions: bool = True,
    panel_subcaptions_bottom_row_only: Optional[bool] = None,
):
    base_width_per_col = 4
    if row == 2:
        base_height_per_row = 4
    else:
        base_height_per_row = 5
    fontsize = 20 * ((col + 3) / (2 + 3))
    figsize = (base_width_per_col * col, base_height_per_row * row)
    plt.figure(figsize=figsize, dpi=300)
    auto_bottom_row_only = (row, col, len(path_list)) in (
        (2, 5, 10),
        (2, 2, 4),
        (2, 3, 6),
    )
    if panel_subcaptions_bottom_row_only is None:
        eff_bottom_row_only = auto_bottom_row_only
    else:
        eff_bottom_row_only = panel_subcaptions_bottom_row_only
    for i, path in enumerate(path_list):
        plt.subplot(row, col, i + 1)
        relative = relative_flag_list[i]
        title = title_list[i]
        if len(path) == 2:
            path1, path2 = path
            if path1.endswith("vary_eta.npy"):
                extent = [0, 1, 0, 0.1]
                yticks = [0.0, 0.05, 0.1]
                plt.yticks(yticks, [f"{y:.2f}" for y in yticks])
            else:
                extent = [0, 1, 0, 1]
            data_dict = np.load(path1, allow_pickle=True).item()
            data_dict2 = np.load(path2, allow_pickle=True).item()
            data = data_dict["heatmap_data"] - data_dict2["heatmap_data"]
            im = plt.imshow(
                data,
                cmap="PuOr",
                aspect="auto",
                extent=extent,
                origin="lower",
                vmin=-diff_vlim,
                vmax=diff_vlim,
            )
        else:
            if path.endswith("vary_eta.npy"):
                extent = [0, 1, 0, 0.1]
                yticks = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
                plt.yticks(yticks, [f"{y:.2f}" for y in yticks])
            else:
                extent = [0, 1, 0, 1]
            data_dict = np.load(path, allow_pickle=True).item()
            if relative:
                data = relative_suppression(data_dict["heatmap_data"])
                levels = np.array([0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0])
                norm = BoundaryNorm(boundaries=levels, ncolors=256)
                im = plt.imshow(
                    data,
                    cmap="Spectral_r",
                    norm=norm,
                    aspect="auto",
                    extent=extent,
                    origin="lower",
                )
                im.set_alpha(0.8)
            else:
                plt.imshow(
                    data_dict["heatmap_data"],
                    cmap="YlGnBu_r",
                    aspect="auto",
                    extent=extent,
                    origin="lower",
                    vmin=0,
                )
        main_ax = plt.gca()
        plt.colorbar()
        x = {
            "prebunking": 0.204,
            "contextualization": 0.342,
            "nudging": 0.143,
            "debunking": 0.342,
        }
        if plot_axline:
            plt.axvline(
                x=x[data_dict["intervention_type"]],
                color="w",
                linestyle="--",
                linewidth=2,
            )
            if data_dict["labels"]["ylabel"] == r"$\eta$":
                plt.axhline(y=0.026, color="w", linestyle="--", linewidth=2)
            if data_dict["labels"]["ylabel"] == r"$\delta_{\mathrm{pre}}$":
                plt.axhline(y=0.2, color="w", linestyle="--", linewidth=2)
            if data_dict["labels"]["ylabel"] == r"$\phi_{\mathrm{ctx}}$":
                plt.axhline(y=0.8, color="w", linestyle="--", linewidth=2)

        if plot_critical_curve:
            path1 = path[1] if len(path) == 2 else path
            if path1.endswith("vary_eta.npy"):
                eps_grid, eta_critical = _vary_eta_critical_curve(
                    data_dict["intervention_type"], lambda_max, delta_pre
                )
                if eta_critical is not None:
                    mask = eta_critical <= 0.1
                    plt.plot(
                        eps_grid[mask],
                        eta_critical[mask],
                        "r:",
                        linewidth=3,
                        label=r"$\eta_c$",
                    )
            elif path1.endswith("random.npy"):
                xe, ye = _critical_curve_xy(_CRITICAL_EPS_RANDOM)
                plt.plot(xe, ye, "r:", linewidth=3)
            elif path1.endswith("high_degree.npy"):
                xe, ye = _critical_curve_xy(_CRITICAL_EPS_DEGREE)
                plt.plot(xe, ye, "r:", linewidth=3)
            elif path1.endswith("high_susceptible.npy"):
                xe, ye = _critical_curve_xy(_CRITICAL_EPS_SUSCEP)
                plt.plot(xe, ye, "r:", linewidth=3)
            elif path1.endswith("cocoon.npy"):
                xe, ye = _critical_curve_xy(_CRITICAL_EPS_COCOON)
                plt.plot(xe, ye, "r:", linewidth=3)

        plt.xlabel(data_dict["labels"]["xlabel"], fontsize=fontsize)
        plt.ylabel(data_dict["labels"]["ylabel"], fontsize=fontsize)
        if title is not None:
            plt.title(title, fontsize=fontsize)
        if panel_subcaptions:
            sub_fs = max(18, int(fontsize * 0.9))
            if col == 5:
                y_pos = -0.4
            elif col == 2:
                y_pos = -0.25
            else:
                y_pos = -0.3
            if eff_bottom_row_only:
                row_idx = i // col
                if row_idx == row - 1:
                    main_ax.text(
                        0.5,
                        y_pos,
                        _panel_subcaption_text(i % col),
                        transform=main_ax.transAxes,
                        fontsize=28,
                        va="top",
                        ha="center",
                    )
            else:
                if col == 5:
                    y_pos = -0.4
                elif col == 3:
                    y_pos = -0.3
                else:
                    y_pos = -0.4
                main_ax.text(
                    0.5,
                    y_pos,
                    _panel_subcaption_text(i),
                    transform=main_ax.transAxes,
                    fontsize=sub_fs,
                    va="top",
                    ha="center",
                )
    if panel_subcaptions:
        plt.tight_layout(rect=[0, 0.07, 1, 1])
    else:
        plt.tight_layout()
    plt.show()

"""単一/複合介入比較の violin / boxplot 系プロット。

旧 `util.py` の以下の関数を分離:
- `violin_plot`
- `boxplot_jitter_plot`
- `boxplot_jitter_multiple_plots`

互換性のため `util.py` 側で re-export している。
"""

from __future__ import annotations

import math
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import plot_style  # noqa: F401  rcParams を適用する副作用


def violin_plot(prevalences_list, graph_name, relative_suppression=False):
    plt.figure(figsize=(6, 4), dpi=300)
    intervention_type_list = ['None', 'Pre', 'Ctx', 'Nud', 'All']
    prevalences_dict = {
        'None': prevalences_list[0],
        'Pre': prevalences_list[1],
        'Ctx': prevalences_list[2],
        'Nud': prevalences_list[3],
        'All': prevalences_list[4],
    }
    intervention_type_list_sort = ['None', 'Nud', 'Pre', 'Ctx', 'All']

    prevalences_list = [prevalences_dict[it] for it in intervention_type_list_sort]
    palette = {
        'None': 'darkgray',
        'Pre': 'steelblue',
        'Ctx': 'darkorange',
        'Nud': 'darkseagreen',
        'All': 'indianred',
    }
    if relative_suppression:
        rho0 = np.mean(prevalences_dict['None'])
        relative_supp = (prevalences_list) / rho0
        df = pd.DataFrame(relative_supp.T, columns=intervention_type_list_sort)
        ylabel = 'Relative Prevalence'
        plt.ylim((0.62, 1.1))
        plt.yticks([0.7, 0.8, 0.9, 1.0])
    else:
        df = pd.DataFrame(prevalences_list.T, columns=intervention_type_list_sort)
        ylabel = 'Prevalence'
    means = df.mean()
    print(means)
    for category, val in means.items():
        i = intervention_type_list_sort.index(category)
        plt.text(i + 0.15, val + 0.03, f"{val:.2f}", ha="center", va="bottom",
                 fontsize=20, fontweight="bold", color="black")
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=20)
    sns.violinplot(data=df, palette=palette, alpha=0.9)
    plt.ylabel(ylabel, fontsize=24)
    os.makedirs(f'results/{graph_name}/png', exist_ok=True)
    plt.savefig(f'results/{graph_name}/png/compare_single_and_combined_intervention_dpi300.png')
    plt.show()


def boxplot_jitter_plot(prevalences_list, graph_name, relative_suppression=False):
    """単一 vs 複合介入の普及率を箱ひげ図 + ジッター（stripplot）で描画する。

    violin_plot と同じ列順・配色・平均ラベル。点は視認性のため灰色に統一。
    """
    plt.figure(figsize=(6, 4), dpi=300)
    intervention_type_list_sort = ["None", "Nud", "Pre", "Ctx", "Comb"]
    prevalences_dict = {
        "None": prevalences_list[0],
        "Pre": prevalences_list[1],
        "Ctx": prevalences_list[2],
        "Nud": prevalences_list[3],
        "Comb": prevalences_list[4],
    }
    prevalences_ordered = [
        prevalences_dict[it] for it in intervention_type_list_sort
    ]
    palette = {
        "None": "darkgray",
        "Pre": "steelblue",
        "Ctx": "darkorange",
        "Nud": "darkseagreen",
        "Comb": "indianred",
    }
    arr = np.asarray(prevalences_ordered, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if relative_suppression:
        rho0 = float(np.mean(prevalences_dict["None"]))
        relative_supp = arr / rho0
        df = pd.DataFrame(relative_supp.T, columns=intervention_type_list_sort)
        ylabel = "Relative Prevalence"
        plt.ylim((0.62, 1.1))
        plt.yticks([0.7, 0.8, 0.9, 1.0])
    else:
        df = pd.DataFrame(arr.T, columns=intervention_type_list_sort)
        ylabel = "Prevalence"

    means = df.mean()
    print(means)
    for category, val in means.items():
        i = intervention_type_list_sort.index(category)
        plt.text(
            i + 0.15,
            val + 0.03,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight="bold",
            color="black",
        )
    plt.yticks(fontsize=20)

    melted = df.melt(var_name="intervention", value_name="value")
    order = intervention_type_list_sort
    ax = sns.boxplot(
        data=melted,
        x="intervention",
        y="value",
        order=order,
        hue="intervention",
        palette={k: palette[k] for k in order},
        legend=False,
        width=0.55,
        linewidth=1.2,
        fliersize=0,
    )
    ax.tick_params(axis="x", labelsize=24, pad=6.0)
    plt.ylabel(ylabel, fontsize=24)
    plt.xlabel("")
    os.makedirs(f"results/{graph_name}/png", exist_ok=True)
    plt.savefig(
        f"results/{graph_name}/png/compare_single_and_combined_intervention_box_jitter_dpi300.png"
    )
    plt.show()


def _panel_subcaption_text(letter_index: int) -> str:
    """パネル番号 0→'(a)', 1→'(b)' …"""
    return f"({chr(ord('a') + letter_index)})"


def boxplot_jitter_multiple_plots(
    prevalences_lists,
    graph_name,
    titles=None,
    relative_suppression=False,
    show_jitter=False,
    save_name="compare_single_and_combined_intervention_box_jitter_multi",
    extra_per_panel=None,
    extra_palette=None,
    nrows=1,
    ncols=None,
    panel_width=5.0,
    panel_height=5.0,
    xtick_label_map=None,
    x_tick_pad=6.0,
    x_tick_labelsize=18,
):
    """boxplot_jitter_plot を格子状サブプロットで描画する。

    （引数の説明は元の docstring を参照。詳しくは旧 util.py を参照。）
    """
    n_panels = len(prevalences_lists)
    if titles is None:
        titles = [None] * n_panels
    assert len(titles) == n_panels, "titles と prevalences_lists の長さが一致しません"

    if ncols is None:
        ncols = math.ceil(n_panels / nrows)
    assert nrows * ncols >= n_panels, (
        f"nrows*ncols={nrows*ncols} は n_panels={n_panels} より小さいです"
    )

    if extra_per_panel is None:
        extra_per_panel = [None] * n_panels
    assert len(extra_per_panel) == n_panels, (
        "extra_per_panel と prevalences_lists の長さが一致しません"
    )

    intervention_order = ["None", "Nud", "Pre", "Ctx", "Comb"]
    palette = {
        "None": "darkgray",
        "Pre": "steelblue",
        "Ctx": "darkorange",
        "Nud": "darkseagreen",
        "Comb": "indianred",
    }

    extra_labels: list = []
    for ext in extra_per_panel:
        if not ext:
            continue
        for k in ext.keys():
            if k not in extra_labels and k not in intervention_order:
                extra_labels.append(k)
    full_order = intervention_order + extra_labels

    extra_palette = dict(extra_palette) if extra_palette else {}
    if extra_labels:
        n_missing = sum(1 for k in extra_labels if k not in extra_palette)
        if n_missing > 0:
            reds_cmap = sns.color_palette("Reds", n_colors=len(extra_labels) + 2)
            j = 0
            for k in extra_labels:
                if k not in extra_palette:
                    extra_palette[k] = reds_cmap[j + 2]
                    j += 1
    full_palette = {**palette, **extra_palette}

    if relative_suppression:
        ylim = (0.62, 1.1)
        yticks = [0.7, 0.8, 0.9, 1.0]
    else:
        all_vals = []
        for prev in prevalences_lists:
            for arr in prev:
                all_vals.extend(np.asarray(arr).ravel().tolist())
        for ext in extra_per_panel:
            if not ext:
                continue
            for arr in ext.values():
                all_vals.extend(np.asarray(arr).ravel().tolist())
        vmax = max(all_vals) if all_vals else 1.0
        ylim = (0.0, vmax * 1.15)
        yticks = None

    fig, axes_2d = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        dpi=300,
        sharey=True,
    )
    axes_arr = np.atleast_2d(axes_2d) if nrows == 1 or ncols == 1 else axes_2d
    axes_flat = list(np.array(axes_arr).reshape(-1))

    for i, (prev_list, title) in enumerate(zip(prevalences_lists, titles)):
        ax = axes_flat[i]
        prevalences_dict = {
            "None": prev_list[0],
            "Pre": prev_list[1],
            "Ctx": prev_list[2],
            "Nud": prev_list[3],
            "Comb": prev_list[4],
        }
        ext = extra_per_panel[i] or {}
        for k, v in ext.items():
            if k in prevalences_dict:
                continue
            prevalences_dict[k] = v

        cols_arr = {
            k: np.asarray(prevalences_dict[k], dtype=float).ravel()
            for k in full_order
            if k in prevalences_dict
        }
        max_len = max((len(a) for a in cols_arr.values()), default=0)
        df_data = {}
        for k in full_order:
            if k not in cols_arr:
                continue
            a = cols_arr[k]
            if len(a) < max_len:
                a = np.concatenate([a, np.full(max_len - len(a), np.nan)])
            df_data[k] = a
        df = pd.DataFrame(df_data)

        if relative_suppression:
            rho0 = float(np.mean(prevalences_dict["None"]))
            df = df / rho0

        means = df.mean()

        present_order = [k for k in full_order if k in df.columns]
        melted = df.melt(var_name="intervention", value_name="value")
        sns.boxplot(
            data=melted,
            x="intervention",
            y="value",
            order=present_order,
            hue="intervention",
            palette={k: full_palette[k] for k in present_order},
            legend=False,
            width=0.55,
            linewidth=1.2,
            fliersize=0,
            ax=ax,
        )
        if show_jitter:
            sns.stripplot(
                data=melted,
                x="intervention",
                y="value",
                order=present_order,
                color="#4a4a4a",
                edgecolor="#2a2a2a",
                alpha=0.75,
                size=4,
                jitter=0.22,
                ax=ax,
            )

        for category, val in means.items():
            xi = present_order.index(category)
            ax.text(
                xi + 0.15,
                val + 0.03,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color="black",
            )

        ax.set_ylim(ylim)
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_xlabel("")
        if xtick_label_map:
            new_xticklabels = [xtick_label_map.get(k, k) for k in present_order]
            ax.set_xticks(range(len(present_order)))
            ax.set_xticklabels(new_xticklabels)
        col_idx = i % ncols
        if col_idx == 0:
            ax.set_ylabel(
                "Relative Prevalence" if relative_suppression else "Prevalence",
                fontsize=20,
            )
        else:
            ax.set_ylabel("")
        ax.set_xlabel("")
        if title is not None:
            ax.set_title(title, fontsize=20)
        ax.tick_params(axis="x", labelsize=x_tick_labelsize, pad=x_tick_pad)
        ax.text(
            0.5,
            -0.2,
            _panel_subcaption_text(i),
            transform=ax.transAxes,
            fontsize=28,
            va="top",
            ha="center",
        )

    for j in range(n_panels, nrows * ncols):
        axes_flat[j].axis("off")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs(f"results/{graph_name}/png", exist_ok=True)
    plt.savefig(f"results/{graph_name}/png/{save_name}_dpi300.png")
    plt.show()

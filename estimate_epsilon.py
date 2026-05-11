import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams['font.family'] = 'Arial'

def compute_epsilon(df, item_col="item_id", cond_col="Condition", share_col="share01",
                    control_value=None, treat_value=None, s0_min=0.1):
    """
    df: tidy data frame
    item_col: item (a)
    cond_col: condition (0/1 or string)
    control_value: control 群の値
    treat_value: intervention 群の値

    returns: df indexed by item_id with s0, s1, epsilon
    """
    # Control と介入だけに絞る
    df2 = df[df[cond_col].isin([control_value, treat_value])].copy()

    # item × condition で平均 s(i,a) を計算
    tab = (
        df2.groupby([item_col, cond_col])[share_col]
           .mean()
           .unstack()
           .rename(columns={control_value: "s0", treat_value: "s1"})
    )
    # ε(a) = 1 − s1 / s0, s0 >= s0_min
    tab = tab[tab["s0"] >= s0_min] # floor effect removal
    tab["epsilon"] = 1 - tab["s1"] / tab["s0"]

    return tab

def estimate_epsilon_nudge(accuracy_prompt=True):

    if accuracy_prompt:
        # nudge_dir = "data/intervention_dataset/accuracy_prompts_Pennycook2020/"
        nudge_dir = "data/intervention_dataset/accuracy_prompts_Pennycook2021/"
        item_col = "fake_index"
        control_value = 1
        treat_value = 2
    else:
        nudge_dir = "data/intervention_dataset/friction_fazio2020/"
        item_col = "item_id"
        control_value = 0
        treat_value = 1
    nudge_path = nudge_dir + "/preprocessed_data.csv"
    df_nudge = pd.read_csv(nudge_path)
    # Condition: 1=control, 2=intervention
    eps_nudge = compute_epsilon(
        df_nudge,
        item_col=item_col,
        cond_col="Condition",
        share_col="share01",
        control_value=control_value,
        treat_value=treat_value
    )
    return eps_nudge


def estimate_epsilon_accuracy_prompt_pennycook2020():
    """
    Pennycook et al. 2020 - accuracy prompts（Study 2）。
    preprocess: fake_index, Condition 1=control, 2=intervention
    """
    nudge_dir = "data/intervention_dataset/accuracy_prompts_Pennycook2020/"
    nudge_path = nudge_dir + "preprocessed_data.csv"
    df = pd.read_csv(nudge_path)
    return compute_epsilon(
        df,
        item_col="fake_index",
        cond_col="Condition",
        share_col="share01",
        control_value=1,
        treat_value=2,
    )


def estimate_epsilon_prebunk(is_goviral=True):
    if is_goviral:
        treat_val = "GoViral"
    else:
        treat_val = "Infographics"
    prebunk_dir = "data/intervention_dataset/inoculation_Basol2021"
    prebunk_path = prebunk_dir + "/preprocessed_data.csv"
    df_prebunk = pd.read_csv(prebunk_path)
    df_prebunk = df_prebunk[df_prebunk["phase"] == "post"] # post のみ
    eps_prebunk = compute_epsilon(
        df_prebunk,
        item_col="item_id",
        cond_col="Condition",
        share_col="share01",
        control_value="Control",
        treat_value=treat_val
    )
    return eps_prebunk

def estimate_epsilon_contextualization():
    contextualization_dir = "data/intervention_dataset/community_notes_Drolsbach2024"
    contextualization_path = contextualization_dir + "/preprocessed_data.csv"
    df_contextualization = pd.read_csv(contextualization_path)
    # Condition: 0 = No Flag, 1 = Community Note
    eps_contextualization = compute_epsilon(
        df_contextualization,
        item_col="item_id",
        cond_col="Condition",
        share_col="share01",
        control_value=0,
        treat_value=1
    )
    return eps_contextualization

def estimate_epsilon_warning_label():
    """
    Pennycook et al. 2020 - warning labels (Study 2)。
    direct effectのみ考えるため、Control vs W (FALSE警告のみ) で推定する。
    Condition: 1=Control, 2=W (warning only)
    """
    warning_dir = "data/intervention_dataset/warning_labels_pennycook2020"
    warning_path = warning_dir + "/preprocessed_data.csv"
    df_warning = pd.read_csv(warning_path)
    eps_warning = compute_epsilon(
        df_warning,
        item_col="item_id",
        cond_col="Condition",
        share_col="share01",
        control_value=1,
        treat_value=2,
    )
    return eps_warning

def estimate_epsilon_source_credibility(provider="FactChecker"):
    """
    Celadin et al. 2023 - source-credibility labels。
    Baseline vs (FactChecker | Laypeople) で推定する。
    provider: "FactChecker" または "Laypeople"
    """
    if provider not in ("FactChecker", "Laypeople"):
        raise ValueError(f"provider must be 'FactChecker' or 'Laypeople', got {provider}")
    sc_dir = "data/intervention_dataset/source-credibility_labels_Celadin2023"
    sc_path = sc_dir + "/preprocessed_data.csv"
    df_sc = pd.read_csv(sc_path)
    eps_sc = compute_epsilon(
        df_sc,
        item_col="item_id",
        cond_col="Condition",
        share_col="share01",
        control_value="Baseline",
        treat_value=provider,
    )
    return eps_sc


def estimate_epsilon_harmony_square_roozenbeek2020():
    """
    Roozenbeek et al. 2020 - Harmony Square。
    Condition2: 1=Control, 2=Harmony Square intervention
    """
    hs_dir = "data/intervention_dataset/inoculation_Roozenbeek2020"
    hs_path = hs_dir + "/preprocessed_data.csv"
    df_hs = pd.read_csv(hs_path)
    eps_hs = compute_epsilon(
        df_hs,
        item_col="item_id",
        cond_col="Condition",
        share_col="share01",
        control_value=1,
        treat_value=2,
    )
    return eps_hs


def estimate_epsilon_video_inoculation_roozenbeek2022_study1():
    """
    Roozenbeek et al. 2022 - 動画イノキュレーション Study 1（Emotional language）のみ。
    Condition2: 1=Control, 2=Inoculation
    """
    vid_dir = "data/intervention_dataset/inoculation_video_Roozenbeek2022"
    vid_path = vid_dir + "/preprocessed_data.csv"
    df_vid = pd.read_csv(vid_path)
    eps_vid = compute_epsilon(
        df_vid,
        item_col="item_id",
        cond_col="Condition",
        share_col="share01",
        control_value=1,
        treat_value=2,
    )
    return eps_vid


def plot_epsilon_distribution(eps_df, label, bins=15):
    plt.figure(figsize=(7,5))
    sns.histplot(eps_df["epsilon"], bins=bins, kde=True, color="skyblue")
    plt.axvline(eps_df["epsilon"].mean(), color="red", linestyle="--", label=f"mean={eps_df['epsilon'].mean():.3f}")
    plt.title(f"Epsilon Distribution: {label}")
    plt.xlabel("epsilon (relative reduction)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 表示順: nudge -> inoculation -> contextualization / labeling / community notes
INTERVENTION_ORDER = [
    # nudge
    "Accuracy Prompt (2020)",
    "Accuracy Prompt (2021)",
    "Friction Nudge",
    # inoculation
    "Infographics",
    "GoViral",
    "Harmony Square",
    "Video Inoc. (S1)",
    # contextualization & labeling
    "Community Notes",
    "Warning Label",
    "Source Cred. (FC)",
    "Source Cred. (LP)",
]

# カテゴリ内で明度・彩度を連続的に変える（ナッジ=緑、プレバンキング=青、文脈化=黄系）
PALETTE = {
    "Accuracy Prompt (2020)": "#c8e6c9",   # 薄い緑
    "Accuracy Prompt (2021)": "#66bb6a",  # 中間の緑
    "Friction Nudge": "#2e7d32",          # 濃い緑
    "Infographics": "#bbdefb",            # 薄い青（白背景でも判別しやすく）
    "GoViral": "#64b5f6",                 # 明るい青
    "Harmony Square": "#1976d2",          # 中間の青
    "Video Inoc. (S1)": "#0d47a1",       # 濃い青
    "Community Notes": "#fff9c4",       # 薄い黄
    "Warning Label": "#ffee58",          # 黄
    "Source Cred. (FC)": "#ffca28",      # アンバー寄り
    "Source Cred. (LP)": "#f9a825",      # やや濃い琥珀
}


def main():
    """全データセットの ε を推定し、分布を可視化して PNG に保存する。

    インタプリタから `import estimate_epsilon` した場合に巨大処理が走らないよう、
    トップレベルのスクリプト実行を `__main__` ガードに移動した。
    """
    eps_accuracy_pc2020 = estimate_epsilon_accuracy_prompt_pennycook2020()
    eps_nudge = estimate_epsilon_nudge(accuracy_prompt=True)
    eps_nudge_friction = estimate_epsilon_nudge(accuracy_prompt=False)
    eps_prebunk = estimate_epsilon_prebunk(is_goviral=True)
    eps_prebunk_infographics = estimate_epsilon_prebunk(is_goviral=False)
    eps_contextualization = estimate_epsilon_contextualization()
    eps_warning = estimate_epsilon_warning_label()
    eps_sc_fc = estimate_epsilon_source_credibility(provider="FactChecker")
    eps_sc_lp = estimate_epsilon_source_credibility(provider="Laypeople")
    eps_harmony = estimate_epsilon_harmony_square_roozenbeek2020()
    eps_video_s1 = estimate_epsilon_video_inoculation_roozenbeek2022_study1()

    summaries = [
        ("Accuracy Prompt (Pennycook 2020)", eps_accuracy_pc2020),
        ("Accuracy Prompt (Pennycook 2021)", eps_nudge),
        ("Friction Nudge", eps_nudge_friction),
        ("Prebunking (Infographics)", eps_prebunk_infographics),
        ("Prebunking (GoViral)", eps_prebunk),
        ("Contextualization (Community Notes)", eps_contextualization),
        ("Warning Label (W)", eps_warning),
        ("Source Credibility (FactChecker)", eps_sc_fc),
        ("Source Credibility (Laypeople)", eps_sc_lp),
        ("Harmony Square (Roozenbeek 2020)", eps_harmony),
        ("Video Inoculation Study 1 (Roozenbeek 2022)", eps_video_s1),
    ]
    for label, df in summaries:
        print(f"{label} ε(a):")
        print(df["epsilon"].mean())
        print("--------------------------------")

    df_eps_all = pd.concat([
        eps_accuracy_pc2020.reset_index().assign(intervention="Accuracy Prompt (2020)"),
        eps_nudge.reset_index().assign(intervention="Accuracy Prompt (2021)"),
        eps_nudge_friction.reset_index().assign(intervention="Friction Nudge"),
        eps_prebunk_infographics.reset_index().assign(intervention="Infographics"),
        eps_prebunk.reset_index().assign(intervention="GoViral"),
        eps_contextualization.reset_index().assign(intervention="Community Notes"),
        eps_warning.reset_index().assign(intervention="Warning Label"),
        eps_sc_fc.reset_index().assign(intervention="Source Cred. (FC)"),
        eps_sc_lp.reset_index().assign(intervention="Source Cred. (LP)"),
        eps_harmony.reset_index().assign(intervention="Harmony Square"),
        eps_video_s1.reset_index().assign(intervention="Video Inoc. (S1)"),
    ], ignore_index=True)

    x_labels = [chr(65 + i) for i in range(len(INTERVENTION_ORDER))]

    df_plot = df_eps_all.copy()
    df_plot["x_label"] = df_plot["intervention"].map(
        {name: lab for name, lab in zip(INTERVENTION_ORDER, x_labels)}
    )

    print("--- x-axis labels (intervention) ---")
    for lab, name in zip(x_labels, INTERVENTION_ORDER):
        print(f"  {lab}: {name}")
    print("------------------------------------")

    # --- Violin（従来）
    plt.figure(figsize=(17, 6), dpi=300)
    plt.hlines(0, -1, len(INTERVENTION_ORDER), colors="k", linestyles="dashed", linewidth=1)
    sns.violinplot(
        data=df_plot,
        x="x_label",
        y="epsilon",
        hue="intervention",
        order=x_labels,
        hue_order=INTERVENTION_ORDER,
        palette=PALETTE,
        legend=False,
        alpha=0.9,
    )
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14)
    plt.xlabel("Intervention", fontsize=18)
    plt.ylabel("Suppression Rate", fontsize=18)
    plt.tight_layout()
    plt.savefig("results/estimated_epsilon.png")
    plt.close()

    # --- 箱ひげ（点のジッターは現状コメントアウト）
    plt.figure(figsize=(6, 3), dpi=300)
    plt.hlines(0, -1, len(INTERVENTION_ORDER), colors="k", linestyles="dashed", linewidth=1)
    sns.boxplot(
        data=df_plot,
        x="x_label",
        y="epsilon",
        hue="intervention",
        order=x_labels,
        hue_order=INTERVENTION_ORDER,
        palette=PALETTE,
        legend=False,
        width=0.5,
        showfliers=False,
        medianprops={"color": "0.2", "linewidth": 1.5},
    )
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14)
    plt.xlabel("Intervention", fontsize=18)
    plt.ylabel("Relative Reduction", fontsize=18)
    plt.tight_layout()
    plt.savefig("results/estimated_epsilon_boxjitter.png")
    if matplotlib.get_backend().lower() == "agg":
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
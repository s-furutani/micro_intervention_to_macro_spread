import pandas as pd
import os
import re
import numpy as np

def preprocess_accuracy_prompts_Pennycook2020():
    directory = "data/intervention_dataset/accuracy_prompts_Pennycook2020"
    path = os.path.join(directory, "raw/Data/Pennycook_et_al__Study_2.csv")
    df_raw = pd.read_csv(path, encoding='mac_roman')
    # print(df_raw.columns.tolist()[:20])
    df_raw = df_raw.rename(columns={"Ô..Condition": "Condition"})
    df = df_raw[df_raw["Finished"] == 1].copy()

    # Fake1_1~Fake1_15 (except RT columns Fake1_RT_...)
    fake_cols = [
        c for c in df.columns
        if c.startswith("Fake1_") and not c.startswith("Fake1_RT_")
    ]

    subset = df[["rid", "Condition"] + fake_cols].copy()

    # wide -> long
    fake_long = subset.melt(
        id_vars=["rid", "Condition"],
        value_vars=fake_cols,
        var_name="item",
        value_name="share"
    )

    fake_long["fake_index"] = (
        fake_long["item"].str.replace("Fake1_", "", regex=False).astype(int)
    )
    fake_long["share01"] = (fake_long["share"] - 1) / 5

    fake_long = fake_long[["rid", "Condition", "fake_index", "share", "share01"]]
    output_path = os.path.join(directory, "preprocessed_data.csv")
    fake_long.to_csv(output_path, index=False)
    print(fake_long.head())


def preprocess_accuracy_prompts_Pennycook2021():
    directory = "data/intervention_dataset/accuracy_prompts_Pennycook2021"
    path = os.path.join(directory, "raw/Data and Code/Study_3_data.csv")

    df_raw = pd.read_csv(path)
    df = df_raw.copy()

    df = df[df["Condition"].isin([1, 2])].copy()
    share_cols = [c for c in df.columns if re.fullmatch(r"Fake\d+_3", c)]

    df["id"] = df["confirmCode"].astype(str)

    records = []
    for col in share_cols:
        fake_index = int(col.replace("Fake", "").replace("_3", ""))
        for _, row in df.iterrows():
            value = row[col]
            if pd.isna(value):
                continue
            records.append(
                {
                    "id": row["id"],
                    "Condition": row["Condition"],
                    "fake_index": fake_index,
                    "share": value,
                }
            )

    df_long = pd.DataFrame(records)
    if df_long.empty:
        raise RuntimeError("df_long is empty – share columns may be wrong, check patterns.")

    df_long["share01"] = (df_long["share"] - 1) / 5.0

    output_path = os.path.join(directory, "preprocessed_data.csv")
    df_long.to_csv(output_path, index=False)
    print(df_long.head())


def preprocess_friction_Fazio2020():
    directory = "data/intervention_dataset/friction_fazio2020"
    path = os.path.join(directory, "raw/share_data_osf.csv")
    df = pd.read_csv(path)
    needed = ["Subject", "Explain"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")

    df = df[df["Explain"].isin([0, 1])].copy()

    false_cols = [f"S{i}" for i in range(13, 25)]
    for col in false_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found (expected false headline column).")

    share_df = df[["Subject", "Explain"] + false_cols]
    df_long = share_df.melt(
        id_vars=["Subject", "Explain"],
        value_vars=false_cols,
        var_name="item_raw",
        value_name="share"
    ).dropna(subset=["share"])

    df_long["item_id"] = df_long["item_raw"].str.extract(r"S(\d+)").astype(int)
    df_long["item_id"] = df_long["item_id"] - 12  # 13→1, 24→12

    df_long["Condition"] = df_long["Explain"].astype(int)

    df_long = df_long.rename(columns={"Subject": "id"})
    df_long["share01"] = (df_long["share"] - 1) / 5.0

    df_long = df_long[["id", "Condition", "item_id", "share", "share01"]]
    output_path = os.path.join(directory, "preprocessed_data.csv")
    df_long.to_csv(output_path, index=False)
    print(df_long.head())

def preprocess_inoculation_Basol2021():
    directory = "data/intervention_dataset/inoculation_Basol2021"
    path = os.path.join(directory, "raw/Data/Study_2_-_final.xlsx")
    df_raw = pd.read_excel(path, sheet_name="Study 2 - final")

    # 2) 有効なケースだけに絞る（必要に応じて調整）
    df = df_raw.copy()
    df = df[df["Finished"] == True]
    df = df[df["Informed consent"].astype(str).str.startswith("Yes")]

    id_col = "Prolific ID"   # 参加者ID
    cond_col = "Condition"

    def is_fake_sharing_col(col, phase_suffix):
        """
        phase_suffix: "-Pre-Sharing" or "-post-Sharing"
        fake候補: Emotion / Expert / Conspir で始まるもの
        """
        if not col.endswith(phase_suffix):
            return False
        if col.startswith("Emotion-") or col.startswith("Expert-") or col.startswith("Conspir-"):
            return True
        return False

    # Pre と Post それぞれで列名を集める
    pre_suffix = "-Pre-Sharing"
    post_suffix = "-post-Sharing"  # 列名そのままに合わせて小文字/大文字注意

    fake_pre_cols = [c for c in df.columns if is_fake_sharing_col(c, pre_suffix)]
    fake_post_cols = [c for c in df.columns if is_fake_sharing_col(c, post_suffix)]

    # print("Pre fake cols:", fake_pre_cols)
    # print("Post fake cols:", fake_post_cols)

    # 4) wide -> long：pre
    pre_long = df[[id_col, cond_col] + fake_pre_cols].melt(
        id_vars=[id_col, cond_col],
        value_vars=fake_pre_cols,
        var_name="item_raw",
        value_name="share"
    )
    pre_long["phase"] = "pre"

    # item_id（例: Emotion-2-tripl-Pre-Sharing -> Emotion-2-tripl）
    pre_long["item_id"] = pre_long["item_raw"].str.replace(pre_suffix, "", regex=False)

    # 5) wide -> long：post
    post_long = df[[id_col, cond_col] + fake_post_cols].melt(
        id_vars=[id_col, cond_col],
        value_vars=fake_post_cols,
        var_name="item_raw",
        value_name="share"
    )
    post_long["phase"] = "post"
    post_long["item_id"] = post_long["item_raw"].str.replace(post_suffix, "", regex=False)

    # 6) pre/post を結合
    long_all = pd.concat([pre_long, post_long], ignore_index=True)

    # 7) Likert (1〜7 を想定) → [0,1] にスケール
    #    欠損やテキストが紛れていてもいいように数値化してから処理
    long_all["share"] = pd.to_numeric(long_all["share"], errors="coerce")
    long_all = long_all.dropna(subset=["share"])

    # [0,1] スケール: (x-1)/6
    long_all["share01"] = (long_all["share"] - 1) / 6

    # 8) 欲しい列だけに整理
    preprocessed = long_all[[id_col, cond_col, "phase", "item_id", "share", "share01"]].rename(
        columns={
            id_col: "id",
            cond_col: "Condition"
        }
    )

    # 9) CSV に保存
    output_path = os.path.join(directory, "preprocessed_data.csv")
    preprocessed.to_csv(output_path, index=False)

    print(preprocessed.head())

def preprocess_community_notes_Drolsbach2024():
    directory = "data/intervention_dataset/community_notes_Drolsbach2024"
    path = os.path.join(directory, "raw/Data/df_main.csv")
    df_raw = pd.read_csv(path, low_memory=False)
    df = df_raw.copy()
    
    df = df[df["S_FactCheck"].isin(["No Fact-Check", "Community Note"])]
    df = df[df["T_Misleading"] == "Misleading"]    
    df["share01"] = df["S_WillReshare"]
    df["Condition"] = (df["S_FactCheck"] == "Community Note").astype(int)

    preprocessed = df[["P_Id", "T_Id", "Condition", "share01"]].rename(
        columns={
            "P_Id": "id",
            "T_Id": "item_id"
        }
    )

    output_path = os.path.join(directory, "preprocessed_data.csv")
    preprocessed.to_csv(output_path, index=False)
    print(preprocessed.head())
    # print(preprocessed["Condition"].value_counts()) # 0: No Fact-Check, 1: Community Note

def preprocess_source_credibility_labels_Celadin2023():
    """
    Celadin et al. 2023 - source-credibility (trustworthiness) labels介入。
    fake見出しのみに絞り [id, Condition, item_id, share, share01] を出力する。

    raw/Analysis_trustworthiness_rating.do の主分析に合わせ、treatment列ではなく
    集約された trt 列 (3値) を使用する:
        trt == 1 -> "Baseline"
        trt == 2 -> "Laypeople"      (LP1 + LP2 を集約)
        trt == 3 -> "FactChecker"    (FC1 + FC2 を集約)
        legend(order(2 "Baseline" 4 "Laypeople" 6 "Fact Checkers"))

    share intention の本体は `rating_headlines` 列 (1-6 Likert)。
    ※ `sharing` 列は multi-select文字列 (共有先SNS選択) なので使わない。
        do file: gen sharing_will = (rating_headlines - 1) / 5

    item_id は `news_headlines` (1〜12 = fake, 13〜24 = real) を使用。
    """
    directory = "data/intervention_dataset/source-credibility_labels_Celadin2023"
    path = os.path.join(directory, "raw/trustworthiness_rating.dta")
    df = pd.read_stata(path, convert_categoricals=False)

    # fake見出しのみに絞る (news_headlines 1〜12)
    df["dummy_fake"] = pd.to_numeric(df["dummy_fake"], errors="coerce")
    df = df[df["dummy_fake"] == 1].copy()

    # trt -> Condition (主分析と同じ集約)
    trt_map = {1: "Baseline", 2: "Laypeople", 3: "FactChecker"}
    df["trt"] = pd.to_numeric(df["trt"], errors="coerce")
    df = df[df["trt"].isin(trt_map.keys())].copy()
    df["Condition"] = df["trt"].map(trt_map)

    out = df[["ID", "Condition", "news_headlines", "rating_headlines"]].rename(
        columns={
            "ID": "id",
            "news_headlines": "item_id",
            "rating_headlines": "share",
        }
    )
    out["share"] = pd.to_numeric(out["share"], errors="coerce")
    out = out.dropna(subset=["id", "item_id", "share"])

    # 1-6 Likert -> [0, 1]
    out["share01"] = (out["share"] - 1) / 5.0

    out = out[["id", "Condition", "item_id", "share", "share01"]]
    output_path = os.path.join(directory, "preprocessed_data.csv")
    out.to_csv(output_path, index=False)
    print(out.head())
    print("Condition counts:", out["Condition"].value_counts().to_dict())


def preprocess_warning_labels_Pennycook2020():
    """
    Pennycook et al. 2020 - warning labels介入 (Study 2)。
    raw/S2.do より:
        gen real = item_num>36                            # fake = 1〜36, real = 37〜64
        gen Warned_W  = (imaget_==1 & real==0) & false==1
        gen Warned_WV = (imaget_==1 & real==0) & falsetrue==1
    すなわち警告ラベルは「fake見出し」かつ「ImageT_i==1」のときに付与される。

    Condition: 1=Control, 2=Warning(W), 3=Warning+Verified(WV)
    抽出方針:
        - fake見出しのみ (item_id <= 36)
        - Control群: 表示された全fake見出し
        - Treatment群 (Condition=2,3): 警告ラベル付きの見出しのみ (ImageT_i==1)
    share: 0/1 binary  ->  share01 = share
    """
    directory = "data/intervention_dataset/warning_labels_pennycook2020"
    path = os.path.join(directory, "raw/Study2_data.csv")

    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    # 完了した参加者のみ (V10==1: response終了フラグ)
    if "V10" in df.columns:
        df = df[df["V10"] == 1].copy()

    # 64 sharingアイテムを wide -> long
    records = []
    for i in range(1, 65):
        sh_col = f"Sharing_{i}"
        if sh_col not in df.columns:
            continue
        it_col = f"ImageT_{i}"

        sub_cols = ["confirmCode", "Condition", sh_col]
        if it_col in df.columns:
            sub_cols.append(it_col)
        tmp = df[sub_cols].copy().rename(
            columns={"confirmCode": "id", sh_col: "share"}
        )
        tmp["item_id"] = i
        tmp["share"] = pd.to_numeric(tmp["share"], errors="coerce")
        tmp = tmp.dropna(subset=["share"])

        # ImageT_i==1: 警告ラベル付きで表示
        if it_col in tmp.columns:
            tmp["image_t"] = pd.to_numeric(tmp[it_col], errors="coerce").fillna(0)
        else:
            tmp["image_t"] = 0

        records.append(tmp[["id", "Condition", "item_id", "share", "image_t"]])

    long_df = pd.concat(records, ignore_index=True)

    # fake見出しのみ (item_id <= 36)
    long_df = long_df[long_df["item_id"] <= 36]

    # Control群は全fake、Treatment群は警告ラベル付きfakeのみを残す
    is_control = long_df["Condition"] == 1
    is_warned = long_df["Condition"].isin([2, 3]) & (long_df["image_t"] == 1)
    long_df = long_df[is_control | is_warned].copy()

    # 0/1 sharingはそのまま [0,1]
    long_df["share01"] = long_df["share"]

    out = long_df[["id", "Condition", "item_id", "share", "share01"]]
    output_path = os.path.join(directory, "preprocessed_data.csv")
    out.to_csv(output_path, index=False)
    print(out.head())


def preprocess_inoculation_Roozenbeek2020():
    """
    Roozenbeek et al. 2020 - Harmony Square（ゲーム型イノキュレーション）。
    raw/Scripts の Stata では共有意向を pre3_/post3_（16アイテム）で扱うが、
    ここではフェイク見出しの事後評価のみを対象に、列名で識別できる
    Fake-*-1/2-Post-Sharing（8アイテム）を用いる（集約列 Fake-*-Post-Sharing は除外）。

    Condition2: 1=Control, 2=Harmony Square（Stata: sharinginoculation=condition2==2）
    share: 1-7 Likert -> share01 = (share - 1) / 6
    """
    directory = "data/intervention_dataset/inoculation_Roozenbeek2020"
    path = os.path.join(directory, "raw/Dataset/Harmony Square RCT - final.csv")
    df = pd.read_csv(path)

    if "ProlificID" not in df.columns or "Condition2" not in df.columns:
        raise ValueError("Expected columns ProlificID, Condition2 in Harmony Square CSV.")

    fake_share_cols = sorted(
        c
        for c in df.columns
        if re.fullmatch(r"Fake-.+-\d+-Post-Sharing", str(c))
    )
    if len(fake_share_cols) == 0:
        raise ValueError("No Fake-*-[12]-Post-Sharing columns found.")

    df = df[["ProlificID", "Condition2"] + fake_share_cols].copy()
    df = df.rename(columns={"ProlificID": "id"})
    df["Condition"] = pd.to_numeric(df["Condition2"], errors="coerce").astype("Int64")
    df = df[df["Condition"].isin([1, 2])].copy()
    df["Condition"] = df["Condition"].astype(int)

    records = []
    for j, col in enumerate(fake_share_cols, start=1):
        tmp = df[["id", "Condition", col]].rename(columns={col: "share"})
        tmp["item_id"] = j
        tmp["share"] = pd.to_numeric(tmp["share"], errors="coerce")
        tmp = tmp.dropna(subset=["share"])
        records.append(tmp)

    long_df = pd.concat(records, ignore_index=True)
    long_df["share01"] = (long_df["share"] - 1) / 6.0
    out = long_df[["id", "Condition", "item_id", "share", "share01"]]

    output_path = os.path.join(directory, "preprocessed_data.csv")
    out.to_csv(output_path, index=False)
    print(out.head())
    print("Condition counts:", out["Condition"].value_counts().to_dict())


def preprocess_inoculation_video_Roozenbeek2022_study1():
    """
    Roozenbeek et al. 2022 - 動画イノキュレーション。**Study 1（Emotional language）のみ**。

    raw/Codebook.txt: ValidResponse=="Yes" で解析サンプル。
    操作刺激ごとの共有意向は「{刺激名}-Sharing」（-C- が付かない列 = 操作的投稿）。
    Condition / Condition2: Control=1, Inoculation=2（補足分析 .do の Condition2==2）

    参加者ID列が無いため、フィルタ後の行番号を id とする（ε推定は item×条件の平均のみ使用）。
    share: 1-7 Likert -> share01 = (share - 1) / 6
    """
    directory = "data/intervention_dataset/inoculation_video_Roozenbeek2022"
    path = os.path.join(directory, "raw/Data/Studies 1-5 - datasets.xlsx")
    sheet = "Emotional language"

    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    if "ValidResponse" not in df.columns:
        raise ValueError("ValidResponse column missing (see Codebook.txt).")
    df = df[df["ValidResponse"].astype(str).str.strip().eq("Yes")].copy()

    if "Condition2" not in df.columns:
        raise ValueError("Condition2 column missing.")
    df["Condition"] = pd.to_numeric(df["Condition2"], errors="coerce").astype("Int64")
    df = df[df["Condition"].isin([1, 2])].copy()
    df["Condition"] = df["Condition"].astype(int)

    share_cols = [
        c
        for c in df.columns
        if isinstance(c, str)
        and c.endswith("-Sharing")
        and "-C-" not in c
        and c not in ("Fake-Sharing", "Control-Sharing", "Diff-Sharing", "SharingBehaviour")
    ]
    share_cols = sorted(share_cols)
    if len(share_cols) == 0:
        raise ValueError("No per-item manipulative Sharing columns found.")

    df = df.reset_index(drop=True)
    df["id"] = df.index.astype(int)

    records = []
    for col in share_cols:
        slug = col.replace("-Sharing", "")
        tmp = df[["id", "Condition", col]].rename(columns={col: "share"})
        tmp["item_id"] = slug
        tmp["share"] = pd.to_numeric(tmp["share"], errors="coerce")
        tmp = tmp.dropna(subset=["share"])
        records.append(tmp)

    long_df = pd.concat(records, ignore_index=True)
    long_df["share01"] = (long_df["share"] - 1) / 6.0
    out = long_df[["id", "Condition", "item_id", "share", "share01"]]

    output_path = os.path.join(directory, "preprocessed_data.csv")
    out.to_csv(output_path, index=False)
    print(out.head())
    print("Condition counts:", out["Condition"].value_counts().to_dict())


def main():
    """各データセットの前処理を実行する。

    実際にはこれまでトップレベルで呼ばれていた関数群をここで列挙する。
    必要に応じてコメントを外して使うにゃん。
    """
    preprocess_source_credibility_labels_Celadin2023()
    preprocess_warning_labels_Pennycook2020()
    preprocess_inoculation_Roozenbeek2020()
    preprocess_inoculation_video_Roozenbeek2022_study1()
    # preprocess_accuracy_prompts_Pennycook2020()
    # preprocess_inoculation_Basol2021()
    # preprocess_community_notes_Drolsbach2024()
    # preprocess_accuracy_prompts_Pennycook2021()
    # preprocess_friction_Fazio2020()


if __name__ == "__main__":
    main()

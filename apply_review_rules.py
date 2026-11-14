# -*- coding: utf-8 -*-
r"""MealRadar 第二阶段：应用人工审核，并保留逐条审核记录。

运行方式（先在 Anaconda/Miniconda Prompt 中激活 mealradar 环境）：
    python "D:\MealRadar\apply_review_rules.py"

本脚本读取原始 CSV，在内存中应用审核决定并排除不确定组。
不会修改原始 CSV，不会保存结果，不执行去重，也不会训练模型。
"""

from pathlib import Path
import pandas as pd
import unicodedata


# 第一部分：记录已确认的人工审核决定
# 字典的键是 review_key，值是人工审核后的标签：0=负面，1=正面。
label_rules = {
    "不太好吃": 0,
    "不怎么好吃": 0,
    "不错很好吃": 1,
    "味道还不错": 1,
    "味道还可以": 1,
    "太慢了": 0,
    "就是有点少": 0,
    "就是有点慢": 0,
    "很好很实惠": 1,
    "等了两个半小时": 0,
    "等了两个小时": 0,
    "还是不错的": 1,
    "送的挺快的": 1,
    "送餐太慢了": 0,
    "送餐速度快": 1,
    "速度有点慢": 0,
}

# 这一组会在第四部分从候选数据中整组排除。
exclude_keys = {
    "太慢了味道还不错",
}

assert len(label_rules) == 16
assert len(exclude_keys) == 1
assert set(label_rules).isdisjoint(exclude_keys)
assert set(label_rules.values()) <= {0, 1}

print("审核规则检查通过：16 组指定标签，1 组整组排除。")


# 第二部分：读取数据，保留原始信息，生成去重键
raw_path = Path(r"D:\MealRadar\data\raw\waimai_10k.csv")
df = pd.read_csv(raw_path)

# 原始索引从 0 开始，表示数据记录顺序，不是 CSV 的物理行号。
df["source_row_index"] = df.index
df["original_label"] = df["label"].copy()


def make_review_key(text):
    # 仅去除空白字符和 Unicode 标点，沿用此前实际使用的规则。
    return "".join(
        char
        for char in text
        if not char.isspace()
        and not unicodedata.category(char).startswith("P")
    )


# 新增一列用于比较；原始 review 列保持不变。
df["review_key"] = df["review"].map(make_review_key)

print("\n前 5 条数据：")
print(
    df[
        ["source_row_index", "original_label", "label", "review", "review_key"]
    ].head().to_string(index=False)
)


# 第三部分：核对实际冲突组与人工审核规则是否完全对应
label_counts = df.groupby("review_key")["original_label"].nunique()
actual_conflict_keys = set(label_counts[label_counts > 1].index)
reviewed_keys = set(label_rules) | exclude_keys

assert actual_conflict_keys == reviewed_keys, (
    f"有冲突但未审核的组：{actual_conflict_keys - reviewed_keys}；"
    f"审核表中不属于当前冲突的组：{reviewed_keys - actual_conflict_keys}"
)

# isin 逐行判断是否命中规则，得到 True/False；求和就是命中记录数。
# 此处只是生成筛选条件，尚未改标、排除或去重。
relabel_mask = df["review_key"].isin(label_rules)
exclude_mask = df["review_key"].isin(exclude_keys)

print("\n审核覆盖统计：")
print("实际冲突组数：", len(actual_conflict_keys))
print("指定标签规则覆盖记录数：", int(relabel_mask.sum()))
print("整组排除记录数：", int(exclude_mask.sum()))
print("审核覆盖记录总数：", int((relabel_mask | exclude_mask).sum()))



# 第四部分：应用审核规则，建立覆盖全部原始记录的阶段审核表
# reviewed_label 是审核后的标签；保留 original_label 和原始 review。
df["reviewed_label"] = df["original_label"].astype("Int64")
df.loc[relabel_mask, "reviewed_label"] = (
    df.loc[relabel_mask, "review_key"].map(label_rules).astype("Int64")
)

# 不确定组没有审核后的有效标签，用缺失值表示，而不是赋成 0。
df.loc[exclude_mask, "reviewed_label"] = pd.NA

# 只把确定标签且新旧标签不同的记录计为“实际改标”。
changed_mask = (
    relabel_mask
    & df["reviewed_label"].ne(df["original_label"]).fillna(False)
)
confirmed_mask = relabel_mask & ~changed_mask

df["review_action"] = "保留原标签（不属于本次冲突审核）"
df["review_reason"] = "未命中本次人工审核规则"
df.loc[relabel_mask, "review_reason"] = "依据已确认的人工审核表指定标签"
df.loc[confirmed_mask, "review_action"] = "人工审核确认，标签未变"
df.loc[changed_mask, "review_action"] = "人工审核改标"
df.loc[exclude_mask, "review_action"] = "整组排除"
df.loc[exclude_mask, "review_reason"] = "人工审核决定暂时排除，未指定统一标签"

# 审核表保留全部原始记录，包括稍后排除的记录。
# 最终去重时，还将补充每条记录的最终去向和保留代表的来源索引。
audit = df[[
    "source_row_index", "review", "review_key", "original_label",
    "reviewed_label", "review_action", "review_reason"
]].copy()

# ~ 表示取反：选择不属于排除组的记录，生成独立的候选数据表。
candidate_df = df.loc[~exclude_mask].copy()
candidate_df["label"] = candidate_df["reviewed_label"].astype("int64")

# 验收：记录数守恒、审核信息完整、剩余相同键不存在标签冲突。
assert len(audit) == len(df)
assert audit["source_row_index"].is_unique
assert int(changed_mask.sum()) + int(confirmed_mask.sum()) == int(relabel_mask.sum())
assert len(candidate_df) + int(exclude_mask.sum()) == len(df)
assert not candidate_df["review_key"].isin(exclude_keys).any()
assert candidate_df["label"].isin([0, 1]).all()
assert candidate_df.groupby("review_key")["label"].nunique().le(1).all()
assert candidate_df["review"].equals(df.loc[candidate_df.index, "review"])

print("\n第二阶段结果（尚未去重）：")
print("原始记录数：", len(df))
print("实际改标记录数：", int(changed_mask.sum()))
print("人工审核确认但标签未变记录数：", int(confirmed_mask.sum()))
print("排除记录数：", int(exclude_mask.sum()))
print("排除后候选记录数：", len(candidate_df))
print("阶段审核表记录数：", len(audit))
print("\n人工审核覆盖的全部记录：")
print(audit.loc[relabel_mask | exclude_mask].to_string(index=False))
print("\n第二阶段验收通过。结果仅在内存中，未修改原始 CSV，未保存文件，尚未去重。")

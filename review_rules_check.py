# -*- coding: utf-8 -*-
r"""MealRadar 第一阶段：准备数据并核对人工审核规则。

运行方式（先在 Anaconda/Miniconda Prompt 中激活 mealradar 环境）：
    python "D:\MealRadar\review_rules_check.py"

本脚本只读取原始 CSV，并打印检查结果。
不会修改原始 CSV，不会删除记录，不会保存清洗结果，也不会训练模型。
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

# 这一组后续需要整组排除；本脚本只核对数量，暂不执行排除。
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

print("\n本阶段检查结束。未修改原始 CSV，未保存清洗结果。")

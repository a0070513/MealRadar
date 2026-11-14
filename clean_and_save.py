# -*- coding: utf-8 -*-
r"""MealRadar 第三阶段：按原始顺序去重，验收并保存清洗成果。

运行方式（先在 Anaconda/Miniconda Prompt 中激活 mealradar 环境）：
    python "D:\MealRadar\clean_and_save.py"

本脚本独立执行已确认的清洗步骤，保存结果和审计记录。
原始 CSV 保持不变。输出至 data/processed 的独立运行文件夹，
并追加记录到 docs/experiment_log.md。不划分数据集，不训练模型。
"""

from pathlib import Path
import pandas as pd
import unicodedata
import hashlib
import json
import sys
from datetime import datetime, timezone, timedelta


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
project_root = Path(r"D:\MealRadar")
raw_path = project_root / "data" / "raw" / "waimai_10k.csv"
raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
df = pd.read_csv(raw_path)

# 这是可复现脚本的输入验收，确认仍是此前检查过的同一版本数据。
assert list(df.columns) == ["label", "review"], "原始列名或顺序与预期不一致"
assert not df.isna().any().any(), "原始数据存在缺失值"
assert df["review"].map(lambda value: isinstance(value, str)).all()
assert not df["review"].str.strip().eq("").any(), "存在空白评论"
assert df["label"].isin([0, 1]).all(), "标签必须是 0 或 1"
raw_checks = {
    "rows": len(df),
    "negative": int(df["label"].eq(0).sum()),
    "positive": int(df["label"].eq(1).sum()),
    "missing_cells": int(df.isna().sum().sum()),
    "blank_reviews": int(df["review"].str.strip().eq("").sum()),
    "exact_duplicate_rows_extra": int(df.duplicated().sum()),
    "label_descending": bool(df["label"].is_monotonic_decreasing),
}
assert raw_checks == {
    "rows": 11987, "negative": 7987, "positive": 4000,
    "missing_cells": 0, "blank_reviews": 0,
    "exact_duplicate_rows_extra": 0, "label_descending": True,
}, f"输入数据与此前已确认的检查结果不一致：{raw_checks}"


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



# 第五部分：按原始顺序，每个 review_key 只保留第一条代表记录
candidate_df = candidate_df.sort_values("source_row_index")
duplicate_mask = candidate_df.duplicated(subset=["review_key"], keep="first")
clean_df = candidate_df.loc[~duplicate_mask, [
    "source_row_index", "label", "review", "review_key"
]].copy().reset_index(drop=True)

# 每组最终保留的来源索引，可用来追踪重复记录被哪一条代表。
representative_ids = clean_df.set_index("review_key")["source_row_index"]
audit["representative_source_row_index"] = (
    audit["review_key"].map(representative_ids).astype("Int64")
)
audit["final_action"] = "保留"
audit["final_reason"] = "按原始 CSV 顺序保留本组第一条（含仅一条记录的组）"
duplicate_ids = candidate_df.loc[duplicate_mask, "source_row_index"]
audit_duplicate_mask = audit["source_row_index"].isin(duplicate_ids)
audit.loc[audit_duplicate_mask, "final_action"] = "重复移除"
audit.loc[audit_duplicate_mask, "final_reason"] = "review_key 相同，保留来源索引更小的代表记录"
audit.loc[exclude_mask, "final_action"] = "整组排除"
audit.loc[exclude_mask, "final_reason"] = "人工审核决定排除此组，无最终代表记录"

# 第六部分：保存前的验收。任何检查失败都会停止，避免保存错误成果。
normalized_duplicates = int(df.duplicated(subset=["review_key"]).sum())
stage_checks = {
    "normalized_duplicate_extra_rows_before_exclusion": normalized_duplicates,
    "conflict_groups": len(actual_conflict_keys),
    "conflict_records": int((relabel_mask | exclude_mask).sum()),
    "rule_covered_records": int(relabel_mask.sum()),
    "changed_labels": int(changed_mask.sum()),
    "confirmed_unchanged_labels": int(confirmed_mask.sum()),
    "excluded_records": int(exclude_mask.sum()),
    "candidate_rows_before_deduplication": len(candidate_df),
}
assert stage_checks == {
    "normalized_duplicate_extra_rows_before_exclusion": 58,
    "conflict_groups": 17, "conflict_records": 40, "rule_covered_records": 38,
    "changed_labels": 16, "confirmed_unchanged_labels": 22,
    "excluded_records": 2, "candidate_rows_before_deduplication": 11985,
}, f"本次复算与已验收结果不一致：{stage_checks}"

assert clean_df["review_key"].is_unique
assert clean_df["source_row_index"].is_unique
assert clean_df["label"].isin([0, 1]).all()
assert not clean_df.isna().any().any()
assert len(clean_df) + int(duplicate_mask.sum()) + int(exclude_mask.sum()) == len(df)
assert len(clean_df) == df["review_key"].nunique() - len(exclude_keys)
assert not clean_df["review_key"].isin(exclude_keys).any()
assert audit["final_action"].value_counts().to_dict() == {
    "保留": len(clean_df), "重复移除": int(duplicate_mask.sum()),
    "整组排除": int(exclude_mask.sum()),
}
assert audit.loc[~exclude_mask, "representative_source_row_index"].notna().all()
assert audit.loc[exclude_mask, "representative_source_row_index"].isna().all()
pd.testing.assert_series_equal(
    representative_ids.sort_index(),
    candidate_df.groupby("review_key")["source_row_index"].min().sort_index(),
    check_names=False,
)
assert clean_df["review"].tolist() == df.loc[clean_df["source_row_index"], "review"].tolist()
assert clean_df["label"].tolist() == df.loc[clean_df["source_row_index"], "reviewed_label"].tolist()
assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == raw_sha256, "原始文件在运行期间发生变化"

# 第七部分：保存成果。每次运行创建独立文件夹，不覆盖过去的结果。
now = datetime.now(timezone(timedelta(hours=8)))
run_id = now.strftime("cleaning_%Y%m%d_%H%M%S_%f")
output_dir = project_root / "data" / "processed" / run_id
output_dir.mkdir(parents=True, exist_ok=False)
docs_dir = project_root / "docs"
docs_dir.mkdir(parents=True, exist_ok=True)

clean_path = output_dir / "waimai_10k_clean.csv"
audit_path = output_dir / "cleaning_audit.csv"
rules_path = output_dir / "manual_review_rules.csv"
summary_path = output_dir / "check_summary.json"
log_path = docs_dir / "experiment_log.md"

rules_df = pd.DataFrame([
    {"review_key": key, "decision": "指定标签", "reviewed_label": label,
     "reason": "用户已确认的人工审核决定"}
    for key, label in label_rules.items()
] + [
    {"review_key": key, "decision": "整组排除", "reviewed_label": pd.NA,
     "reason": "用户已确认暂时排除，不指定统一标签"}
    for key in sorted(exclude_keys)
])
rules_df["reviewed_label"] = rules_df["reviewed_label"].astype("Int64")

clean_df.to_csv(clean_path, index=False, encoding="utf-8-sig")
audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
rules_df.to_csv(rules_path, index=False, encoding="utf-8-sig")

# 从磁盘读回并比较，确认保存后的数据与内存中结果一致。
pd.testing.assert_frame_equal(
    pd.read_csv(clean_path, encoding="utf-8-sig", keep_default_na=False,
                dtype={"review": str, "review_key": str}), clean_df
)
pd.testing.assert_frame_equal(
    pd.read_csv(audit_path, encoding="utf-8-sig", keep_default_na=False,
                dtype={"review": str, "review_key": str,
                       "reviewed_label": "Int64", "representative_source_row_index": "Int64"},
                na_values={"reviewed_label": [""], "representative_source_row_index": [""]}), audit
)
pd.testing.assert_frame_equal(
    pd.read_csv(rules_path, encoding="utf-8-sig", keep_default_na=False,
                dtype={"reviewed_label": "Int64"}, na_values={"reviewed_label": [""]}), rules_df
)

script_bytes = Path(__file__).read_bytes()
(output_dir / "clean_and_save.py").write_bytes(script_bytes)
summary = {
    "run_id": run_id, "run_time_Asia_Shanghai": now.isoformat(),
    "raw_path": str(raw_path), "raw_sha256": raw_sha256,
    "script_sha256": hashlib.sha256(script_bytes).hexdigest(),
    "python_version": sys.version, "pandas_version": pd.__version__,
    "unicode_version": unicodedata.unidata_version,
    "raw_checks": raw_checks, "review_checks": stage_checks,
    "final_checks": {
        "duplicate_records_removed_after_exclusion": int(duplicate_mask.sum()),
        "clean_rows": len(clean_df), "audit_rows": len(audit),
        "negative": int(clean_df["label"].eq(0).sum()),
        "positive": int(clean_df["label"].eq(1).sum()),
        "remaining_duplicate_keys": int(clean_df["review_key"].duplicated().sum()),
        "csv_readback_verified": True, "raw_sha256_unchanged": True,
    },
    "artifacts_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [clean_path, audit_path, rules_path]
    },
    "next_step": "确认本次清洗验收结果后，讨论并执行分层随机划分；尚未划分或训练",
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

limitations = (
    "仅去除空白字符和 Unicode 标点生成 review_key，先按人工审核表指定标签、整组排除不确定评论，"
    "再按原始 CSV 顺序每组保留第一条。建模使用保留记录的原始 review。"
    "此策略可能丢失标点或空白带来的语气、情绪强度乃至语义差异；"
    "未逐组评估代表记录，最早出现不代表最具代表性或标签最准确。"
    "选择过程确定、可复现，不属于随机抽样；原始数据按标签降序排列，也可能影响保留的原文版本。"
    "人工审核决定优先于原标签。未对全量评论开展逐条语义重审。"
)
report = f"""
## 数据清洗验收：{now.isoformat()}

- 运行编号：{run_id}
- 原始文件：{raw_path}
- 原始文件 SHA-256：{raw_sha256}
- 原始记录 11987 条：负面 7987，正面 4000；无缺失、空白评论和完全重复整行，标签降序排列。本次脚本已复核。
- 规范化重复额外记录 58 条；标签冲突 17 组、40 条。
- 人工审核指定标签覆盖 38 条：实际改标 16 条，标签未变 22 条；另整组排除 2 条。
- 排除后候选数据 11985 条；随后去重移除 {int(duplicate_mask.sum())} 条。
- 最终数据 {len(clean_df)} 条：负面 {int(clean_df['label'].eq(0).sum())}，正面 {int(clean_df['label'].eq(1).sum())}。
- 完整审计日志 {len(audit)} 条，记录原始索引、原文、原始/审核后标签、审核原因、最终去向和代表记录索引。来源索引从 0 开始。
- 原始数据中的 58 条重复额外记录包含被排除组的重复；排除与去重数量不能直接重复相减。
- 输出文件夹：{output_dir}
- 输出：waimai_10k_clean.csv、cleaning_audit.csv、manual_review_rules.csv、check_summary.json、脚本副本、此验收记录副本。
- 验收：去重键唯一、记录数守恒、原文保留、原始文件哈希不变、三个 CSV 保存后读回一致。

### 方法与局限

{limitations}

### 暂停点和下次工作

本次数据清洗及落盘验收完成，尚未划分数据集、提取特征或训练模型。下次从本运行文件夹读取清洗数据，先确认标签比例及划分方案，再进行分层随机划分；不能按当前行顺序直接切分。无需重新人工审核这 17 组。
"""
(output_dir / "run_report.md").write_text(report.lstrip(), encoding="utf-8")
with log_path.open("a", encoding="utf-8") as log_file:
    log_file.write("\n" + report)

print("\n第三阶段：保存与验收完成")
print("原始记录数：", len(df))
print("整组排除记录数：", int(exclude_mask.sum()))
print("随后去重移除记录数：", int(duplicate_mask.sum()))
print("最终清洗记录数：", len(clean_df))
print("最终负面 / 正面：", int(clean_df["label"].eq(0).sum()), "/", int(clean_df["label"].eq(1).sum()))
print("完整审计日志记录数：", len(audit))
print("结果文件夹：", output_dir)
print("实验记录：", log_path)
print("原始 CSV 未改变，保存后读回验收通过。今天可在这里暂停。")

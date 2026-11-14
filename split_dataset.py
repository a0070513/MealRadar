# -*- coding: utf-8 -*-
r"""MealRadar：分层随机划分并保存数据集，不训练模型。
在 mealradar 环境中运行：python "D:\MealRadar\split_dataset.py"
固定使用已验收的清洗版本；原始数据和清洗数据均不修改。
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
from itertools import combinations
import hashlib
import json
import sys
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

CLEANING_RUN = "cleaning_20261112_000017_173028"
SEED = 42


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_data(df):
    # 第一步：取整体的 10% 作为测试集，剩下 90%。
    train_val_df, test_df = train_test_split(
        df, test_size=0.10, random_state=SEED,
        shuffle=True, stratify=df["label"],
    )
    # 第二步：从剩下的 90% 中取 1/9，即整体约 10%，作为验证集。
    train_df, val_df = train_test_split(
        train_val_df, test_size=1 / 9, random_state=SEED,
        shuffle=True, stratify=train_val_df["label"],
    )
    return {
        "train": train_df.reset_index(drop=True),
        "validation": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def main(project_root=Path(r"D:\MealRadar")):
    # 一、读取上次已验收的清洗数据，核对文件是否与保存时一致。
    source_dir = project_root / "data" / "processed" / CLEANING_RUN
    source_path = source_dir / "waimai_10k_clean.csv"
    source_hash = file_sha256(source_path)
    cleaning_summary = json.loads((source_dir / "check_summary.json").read_text(encoding="utf-8"))
    assert source_hash == cleaning_summary["artifacts_sha256"][source_path.name], "清洗文件与上次验收时不一致"
    read_options = dict(encoding="utf-8-sig", keep_default_na=False,
                        dtype={"review": str, "review_key": str})
    df = pd.read_csv(source_path, **read_options)
    assert list(df.columns) == ["source_row_index", "label", "review", "review_key"]
    assert len(df) == 11928
    assert df["label"].value_counts().to_dict() == {0: 7954, 1: 3974}
    assert df["source_row_index"].is_unique and df["review_key"].is_unique
    assert not df.isna().any().any()

    # 二、划分。标签用于保持各集合的比例，不是模型输入特征。
    parts = split_data(df)

    # 三、验收：无遗漏、无重叠、原文和标签不变，标签比例接近原数据。
    assert {name: len(part) for name, part in parts.items()} == {
        "train": 9542, "validation": 1193, "test": 1193,
    }
    for left, right in combinations(parts.values(), 2):
        for column in ["source_row_index", "review_key"]:
            assert set(left[column]).isdisjoint(right[column]), f"集合间存在重复的 {column}"
    reconstructed = pd.concat(parts.values(), ignore_index=True)
    pd.testing.assert_frame_equal(
        reconstructed.sort_values("source_row_index").reset_index(drop=True),
        df.sort_values("source_row_index").reset_index(drop=True),
    )
    for part in parts.values():
        assert abs(part["label"].mean() - df["label"].mean()) <= 2 / len(part)

    # 四、保存到新的运行文件夹，并保存每条记录所属的数据集。
    now = datetime.now(timezone(timedelta(hours=8)))
    run_id = now.strftime("split_%Y%m%d_%H%M%S_%f")
    output_dir = project_root / "data" / "processed" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    stats = []
    assignments = []
    artifact_hashes = {}
    for name, part in parts.items():
        path = output_dir / f"{name}.csv"
        part.to_csv(path, index=False, encoding="utf-8-sig")
        pd.testing.assert_frame_equal(pd.read_csv(path, **read_options), part)
        artifact_hashes[path.name] = file_sha256(path)
        assignments.append(part[["source_row_index"]].assign(split=name))
        stats.append({"split": name, "rows": len(part),
                      "negative": int(part["label"].eq(0).sum()),
                      "positive": int(part["label"].eq(1).sum()),
                      "positive_ratio": float(part["label"].mean())})
    manifest = pd.concat(assignments, ignore_index=True).sort_values("source_row_index").reset_index(drop=True)
    manifest_path = output_dir / "split_assignments.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    pd.testing.assert_frame_equal(pd.read_csv(manifest_path, encoding="utf-8-sig"), manifest)
    artifact_hashes[manifest_path.name] = file_sha256(manifest_path)
    assert source_hash == file_sha256(source_path), "源文件在运行期间发生改变"

    script_bytes = Path(__file__).read_bytes()
    (output_dir / "split_dataset.py").write_bytes(script_bytes)
    summary = {
        "run_id": run_id, "time_Asia_Shanghai": now.isoformat(),
        "source_path": str(source_path), "source_sha256": source_hash,
        "cleaning_run": CLEANING_RUN, "seed": SEED,
        "target_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "method": "先分出 10% 测试集，再从剩余部分分出 1/9 验证集；均按 label 分层随机划分",
        "python_version": sys.version, "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__, "statistics": stats,
        "artifacts_sha256": artifact_hashes,
        "script_sha256": hashlib.sha256(script_bytes).hexdigest(),
        "checks": {"source_unchanged": True, "no_missing_or_changed_rows": True,
                   "disjoint_source_indices_and_review_keys": True, "csv_readback_verified": True},
        "limitations": "仅检查来源索引和 review_key 的跨集合重叠，未进行语义近重复或用户/商家分组检查。",
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    table = pd.DataFrame(stats).to_string(index=False)
    report = f"""\n## 数据集划分：{now.isoformat()}

输入：{source_path}
输入 SHA-256：{source_hash}
输出文件夹：{output_dir}
方法：按标签分层随机划分，目标比例 80% / 10% / 10%，随机种子 {SEED}。
随机种子用于复现，不代表最优选择；后续模型比较固定使用本次划分，不根据成绩挑选种子。
各集合保留原有约 2:1 的负面/正面比例，不进行类别平衡采样。

```text
{table}
```

验收：记录无遗漏或改动，来源索引与 review_key 跨集合均无交集，各 CSV 读回一致，源文件未变。
限制：未进行语义近重复或用户/商家分组检查，不能据此声称所有潜在泄漏都已排除。
训练集用于拟合模型及词表/TF-IDF 等从数据学习的转换；验证集用于选择方案；测试集留到方案确定后的最终评估。
本阶段尚未提取特征或训练模型。下一步先讲解文本特征与基线模型，再准备训练代码。
"""
    (output_dir / "run_report.md").write_text(report.lstrip(), encoding="utf-8")
    log_path = project_root / "docs" / "experiment_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(report)

    print("数据集划分与保存验收通过")
    print(table)
    print("来源索引与 review_key 跨集合交集均为 0；记录无遗漏、无改动。")
    print("清洗源文件未改变；CSV 保存后读回一致。")
    print("结果文件夹：", output_dir)
    print("实验记录：", log_path)
    print("尚未提取特征或训练模型。请保留本次划分用于后续实验。")


if __name__ == "__main__":
    main()

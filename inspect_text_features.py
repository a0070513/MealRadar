# -*- coding: utf-8 -*-
r"""MealRadar：从 review 原文提取字符级 TF-IDF，检查并保存特征。
运行：python "D:\MealRadar\inspect_text_features.py"
只读取已确认划分中的训练集和验证集，不读取 test.csv，不训练分类模型。
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import sys
import unicodedata
import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

SPLIT_RUN = "split_20261113_085939_408201"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(project_root=Path(r"D:\MealRadar")):
    # 一、读取本次固定划分，确认文件与划分验收时相同。
    source_dir = project_root / "data" / "processed" / SPLIT_RUN
    split_summary = json.loads((source_dir / "split_summary.json").read_text(encoding="utf-8"))
    frames, input_hashes = {}, {}
    for name, expected_rows in [("train", 9542), ("validation", 1193)]:
        path = source_dir / f"{name}.csv"
        input_hashes[path.name] = sha256(path)
        assert input_hashes[path.name] == split_summary["artifacts_sha256"][path.name]
        frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False,
                            dtype={"review": str, "review_key": str})
        assert len(frame) == expected_rows and frame["source_row_index"].is_unique
        assert frame["review_key"].is_unique and frame["label"].isin([0, 1]).all()
        assert not frame["review"].str.strip().eq("").any()
        frames[name] = frame
    train_df, val_df = frames["train"], frames["validation"]
    for col in ["source_row_index", "review_key"]:
        assert set(train_df[col]).isdisjoint(val_df[col])

    # 二、使用 review 原文。字符片段可包含标点、表情和符号。
    # 不设置停用词，不按文档频率或特征总数筛除片段。
    # 默认字符分析器会把连续多个空白压缩为一个空格；原始 CSV 不变。
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(1, 3), lowercase=False,
        min_df=1, max_df=1.0, max_features=None, dtype=np.float32,
    )
    X_train = vectorizer.fit_transform(train_df["review"])
    # 验证集只使用训练集建立的特征表和 IDF，不参与 fit。
    X_val = vectorizer.transform(val_df["review"])
    feature_names = vectorizer.get_feature_names_out()
    assert X_train.shape[0] == len(train_df) and X_val.shape[0] == len(val_df)
    assert X_train.shape[1] == X_val.shape[1] == len(feature_names)
    assert np.isfinite(X_train.data).all() and np.isfinite(X_val.data).all()

    # 三、检查标点/符号特征。类别 P=标点、S=符号。
    punctuation_features = [f for f in feature_names if any(unicodedata.category(c).startswith("P") for c in f)]
    symbol_features = [f for f in feature_names if any(unicodedata.category(c).startswith("S") for c in f)]
    # 一个片段可以同时属于两类；这两项数量不能直接相加。
    demo_features = vectorizer.build_analyzer()("不好吃！")
    assert "！" in demo_features and "吃！" in demo_features

    # 四、保存转换器、稀疏矩阵以及每行对应的来源索引和标签。
    now = datetime.now(timezone(timedelta(hours=8)))
    output_dir = project_root / "data" / "processed" / now.strftime("features_%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    matrices = {"train": X_train, "validation": X_val}
    stats = {}
    for name, matrix in matrices.items():
        matrix_path = output_dir / f"X_{name}.npz"
        sparse.save_npz(matrix_path, matrix)
        saved = sparse.load_npz(matrix_path)
        assert saved.shape == matrix.shape and (saved != matrix).nnz == 0
        rows = frames[name][["source_row_index", "label"]].reset_index(drop=True)
        rows_path = output_dir / f"{name}_rows.csv"
        rows.to_csv(rows_path, index=False, encoding="utf-8-sig")
        pd.testing.assert_frame_equal(pd.read_csv(rows_path, encoding="utf-8-sig"), rows)
        stats[name] = {"rows": matrix.shape[0], "features": matrix.shape[1],
                       "nonzero_values": matrix.nnz,
                       "zero_feature_rows": int((matrix.getnnz(axis=1) == 0).sum())}
    vectorizer_path = output_dir / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    restored = joblib.load(vectorizer_path)
    probe = train_df["review"].iloc[:5]
    assert (restored.transform(probe) != vectorizer.transform(probe)).nnz == 0
    for filename, digest in input_hashes.items():
        assert sha256(source_dir / filename) == digest
    (output_dir / "inspect_text_features.py").write_bytes(Path(__file__).read_bytes())
    summary = {
        "time_Asia_Shanghai": now.isoformat(), "split_run": SPLIT_RUN,
        "input_sha256": input_hashes, "text_column": "review", "fit_on": "train_only",
        "settings": {"analyzer": "char", "ngram_range": [1, 3], "lowercase": False,
                     "min_df": 1, "max_df": 1.0, "max_features": None, "dtype": "float32"},
        "statistics": stats, "punctuation_feature_count": len(punctuation_features),
        "symbol_feature_count": len(symbol_features),
        "punctuation_feature_examples": punctuation_features[:20],
        "symbol_feature_examples": symbol_features[:20],
        "synthetic_demo": {"text": "不好吃！", "analyzed_fragments": demo_features},
        "python_version": sys.version, "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__, "numpy_version": np.__version__,
        "scipy_version": scipy.__version__, "unicode_version": unicodedata.unidata_version,
        "test_csv_read": False, "classifier_trained": False,
        "notes": "连续多个空白在字符分析阶段压缩为一个空格；未出现于训练特征表的验证片段不新增为特征。",
    }
    (output_dir / "feature_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""\n## 字符级 TF-IDF 特征验收：{now.isoformat()}

划分来源：{source_dir}
特征输出：{output_dir}
输入列为 review 原文，review_key 仅用于重叠检查，不作为模型输入。
配置：char，1～3 字符片段，保留大小写，min_df=1，不限制特征总数，float32。
仅在训练集拟合特征表和 IDF，验证集仅 transform；未读取 test.csv，未训练分类模型。
字符分析器会压缩连续空白，不主动移除标点或符号。未在训练集中出现的片段不新增为验证集特征。
训练矩阵：{X_train.shape}；验证矩阵：{X_val.shape}。
包含 Unicode 标点的特征数量：{len(punctuation_features)}；包含符号的特征数量：{len(symbol_features)}，两类可以重叠。
验证集全零特征行数：{stats['validation']['zero_feature_rows']}。
矩阵行序与对应的 train_rows.csv / validation_rows.csv 一致；保留来源索引和标签以供训练及追溯。
保存后已读回检查矩阵、标签行表和转换器，输入 CSV 哈希未改变。
该方案不能完整表达长距离语义；保留标点特征不代表已证明标点提高分类效果。
下一步在这些训练特征上讲解并建立分类基线，仅使用验证集评估开发方案。
"""
    (output_dir / "run_report.md").write_text(report.lstrip(), encoding="utf-8")
    log_path = project_root / "docs" / "experiment_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(report)
    print("特征提取与保存验收通过（尚未训练分类模型）")
    print("使用文本列：review；特征表和 IDF 仅从训练集学习。")
    print("训练特征矩阵（行数，特征数）：", X_train.shape)
    print("验证特征矩阵（行数，特征数）：", X_val.shape)
    print("含标点的特征数：", len(punctuation_features))
    print("含符号的特征数：", len(symbol_features))
    print("验证集全零特征行数：", stats["validation"]["zero_feature_rows"])
    print("人为示例“ 不好吃！ ”的字符片段：", demo_features)
    print("结果文件夹：", output_dir)
    print("实验记录：", log_path)
    print("源文件未改变；未读取 test.csv；保存后读回验收通过。")


if __name__ == "__main__":
    main()

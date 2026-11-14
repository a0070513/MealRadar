# -*- coding: utf-8 -*-
r"""MealRadar：训练首个逻辑回归基线，仅在验证集评估。
运行：python "D:\MealRadar\train_baseline.py"
复用已保存的 TF-IDF，不读取测试集，不搜索参数或调整分类阈值。
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

FEATURE_RUN = "features_20261113_124640_052892"
SPLIT_RUN = "split_20261113_085939_408201"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1],
            target_names=["负面", "正面"], output_dict=True, zero_division=0),
        "confusion_matrix_true_rows_predicted_columns": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def main(project_root=Path(r"D:\MealRadar")):
    # 一、读取既有特征；核对每一行的来源索引、标签与原始划分一致。
    feature_dir = project_root / "data" / "processed" / FEATURE_RUN
    split_dir = project_root / "data" / "processed" / SPLIT_RUN
    feature_summary = json.loads((feature_dir / "feature_summary.json").read_text(encoding="utf-8"))
    assert feature_summary["split_run"] == SPLIT_RUN
    assert feature_summary["fit_on"] == "train_only" and feature_summary["text_column"] == "review"
    vectorizer = joblib.load(feature_dir / "tfidf_vectorizer.joblib")
    matrices, frames, input_hashes = {}, {}, {}
    for name, rows_expected in [("train", 9542), ("validation", 1193)]:
        csv_path = split_dir / f"{name}.csv"
        assert sha256(csv_path) == feature_summary["input_sha256"][csv_path.name]
        df = pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False,
                         dtype={"review": str, "review_key": str})
        row_path = feature_dir / f"{name}_rows.csv"
        row_table = pd.read_csv(row_path, encoding="utf-8-sig")
        pd.testing.assert_frame_equal(row_table, df[["source_row_index", "label"]])
        matrix_path = feature_dir / f"X_{name}.npz"
        X = sparse.load_npz(matrix_path)
        assert len(df) == X.shape[0] == rows_expected
        assert X.shape[1] == len(vectorizer.vocabulary_)
        assert df["label"].isin([0, 1]).all() and np.isfinite(X.data).all()
        # 只调用 transform，不重新拟合；核对矩阵没有错配或改动。
        difference = X - vectorizer.transform(df["review"])
        assert difference.nnz == 0 or np.max(np.abs(difference.data)) < 1e-6
        matrices[name], frames[name] = X, df
        for path in [csv_path, row_path, matrix_path]:
            input_hashes[str(path)] = sha256(path)
    for path in [feature_dir / "tfidf_vectorizer.joblib", feature_dir / "feature_summary.json"]:
        input_hashes[str(path)] = sha256(path)
    for col in ["source_row_index", "review_key"]:
        assert set(frames["train"][col]).isdisjoint(frames["validation"][col])
    X_train, X_val = matrices["train"], matrices["validation"]
    y_train, y_val = frames["train"]["label"], frames["validation"]["label"]

    # 二、简单对照：永远预测训练集中最多的类别，完全不分析文本。
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_val)
    dummy_metrics = evaluate(y_val, dummy_pred)

    # 三、训练逻辑回归。C 控制正则化强度，第一版固定为 1，不搜索参数。
    # max_iter 是求解器迭代次数上限，不是实际一定要运行的次数。
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000,
                               random_state=42, class_weight=None)
    print("正在用训练集拟合逻辑回归……", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X_train, y_train)

    # 四、预测与评估。训练成绩只作诊断，验证成绩用于评价开发方案。
    val_pred = model.predict(X_val)
    train_metrics = evaluate(y_train, model.predict(X_train))
    val_metrics = evaluate(y_val, val_pred)
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    positive_scores = model.predict_proba(X_val)[:, positive_column]
    # 这里是模型估计概率，不代表已经校准的真实正确率。

    # 五、保存完整文本预测流程，今后输入 review 原文即可预测。
    now = datetime.now(timezone(timedelta(hours=8)))
    output_dir = project_root / "experiments" / now.strftime("baseline_%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    pipeline = Pipeline([("tfidf", vectorizer), ("classifier", model)])
    model_path = output_dir / "text_classifier.joblib"
    joblib.dump(pipeline, model_path)
    restored = joblib.load(model_path)
    assert np.array_equal(restored.predict(frames["validation"]["review"]), val_pred)
    predictions = frames["validation"][["source_row_index", "review", "label"]].copy()
    predictions["predicted_label"] = val_pred
    predictions["positive_probability"] = positive_scores
    predictions["correct"] = predictions["label"].eq(predictions["predicted_label"])
    predictions_path = output_dir / "validation_predictions.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    pd.testing.assert_frame_equal(pd.read_csv(predictions_path, encoding="utf-8-sig", keep_default_na=False,
                                              dtype={"review": str}), predictions,
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    for path, digest in input_hashes.items():
        assert sha256(Path(path)) == digest
    script_bytes = Path(__file__).read_bytes()
    (output_dir / "train_baseline.py").write_bytes(script_bytes)
    metrics = {
        "time_Asia_Shanghai": now.isoformat(), "split_run": SPLIT_RUN,
        "feature_run": FEATURE_RUN, "input_sha256": input_hashes,
        "model_parameters": model.get_params(), "actual_iterations": model.n_iter_.tolist(),
        "dummy_validation": dummy_metrics, "logistic_train": train_metrics,
        "logistic_validation": val_metrics,
        "python_version": sys.version, "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "output_sha256": {model_path.name: sha256(model_path), predictions_path.name: sha256(predictions_path)},
        "script_sha256": hashlib.sha256(script_bytes).hexdigest(),
        "test_csv_read": False, "parameter_search_performed": False,
        "interpretation": "验证集成绩属于开发阶段结果，不是最终测试成绩。Macro-F1 为主要比较指标，另报告准确率及逐类指标。",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison = pd.DataFrame([
        {"model_and_dataset": "多数类对照 / 验证集", "accuracy": dummy_metrics["accuracy"], "macro_f1": dummy_metrics["macro_f1"]},
        {"model_and_dataset": "逻辑回归 / 训练集", "accuracy": train_metrics["accuracy"], "macro_f1": train_metrics["macro_f1"]},
        {"model_and_dataset": "逻辑回归 / 验证集", "accuracy": val_metrics["accuracy"], "macro_f1": val_metrics["macro_f1"]},
    ]).to_string(index=False)
    detail = classification_report(y_val, val_pred, labels=[0, 1], target_names=["负面", "正面"], digits=4, zero_division=0)
    confusion = pd.DataFrame(val_metrics["confusion_matrix_true_rows_predicted_columns"],
                             index=["实际负面", "实际正面"], columns=["预测负面", "预测正面"]).to_string()
    report = f"""\n## 首个分类基线：{now.isoformat()}

划分版本：{SPLIT_RUN}；特征版本：{FEATURE_RUN}。
输出：{output_dir}
复用仅由训练集拟合的字符级 TF-IDF；在训练集拟合逻辑回归。
固定参数：C=1、solver=liblinear、max_iter=1000、random_state=42、class_weight=None。
未重新采样、未搜索参数、未调阈值；使用模型默认分类规则（正面概率大于 0.5 判为正面）。
多数类对照只预测训练集中最多的类别。主要比较指标为 Macro-F1。
训练集指标仅作诊断；验证集指标用于开发阶段比较。未读取测试集，尚未获得最终测试成绩。

```text
{comparison}

逻辑回归验证集逐类指标：
{detail}
混淆矩阵（行=真实标签，列=预测标签）：
{confusion}
```

保存了包含转换器的完整模型、验证集逐条预测、指标及脚本副本。模型重载后预测一致，预测 CSV 读回一致，输入文件哈希未变。
概率未经单独校准；保留标点特征不等于已证明其独立贡献。本阶段只建立首个基线，未宣称模型最优。
下一步检查验证集错误类型，决定是否做有限的方案比较；测试集继续保留。
"""
    (output_dir / "run_report.md").write_text(report.lstrip(), encoding="utf-8")
    log_path = project_root / "docs" / "experiment_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(report)
    print("基线训练、验证与保存完成")
    print(comparison)
    print("\n逻辑回归验证集逐类指标：\n", detail)
    print("混淆矩阵（行=真实，列=预测）：\n", confusion)
    print("实际迭代次数：", model.n_iter_.tolist())
    print("结果文件夹：", output_dir)
    print("实验记录：", log_path)
    print("未读取测试集。上述验证成绩不代表最终测试成绩。")


if __name__ == "__main__":
    main()

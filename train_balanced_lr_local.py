# -*- coding: utf-8 -*-
r"""MealRadar 固定划分 Balanced LR 对照实验，v1.0.0。

Anaconda Prompt:
    conda activate mealradar
    cd /d D:\MealRadar
    python train_balanced_lr_local.py --check-only
    python train_balanced_lr_local.py

可选：--revised-validation "完整修订验证集的真实路径"
只接受已核对的完整 1193 行修订文件；不接受 28 行补丁或仅标签列。
--check-only 只读预检查，不训练、不创建目录。
默认根目录 D:\MealRadar；移动完整实验后可用 --project-root 指定新根目录。
本脚本只读取指定训练/验证数据和基线；不读取测试集、不修改已有文件。
TF-IDF 只 transform。仅新 LR.fit；不调用 Pipeline.fit，不重新训练普通 LR。
输出均写入 experiments/balanced_lr_上海时间戳_随机后缀/，失败目录保留供排查。
指标为开发验证结果；定向修订标签的分数只作补充分析。

核对来源（2026-11-17）：GitHub a0070513/MealRadar 的 train_baseline.py，
本机基线 metrics.json、实际模型、固定 train/validation CSV，以及当前工作目录
validation_revision_28/validation_comparison.json 和 validation_revised28.csv。
以下 SHA-256 锁定这些实验输入，避免误用同名文件；不自动搜索或降级换模型。
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import uuid
import warnings

VERSION = "1.0.0"
SPLIT = "split_20261113_085939_408201"
BASELINE = "baseline_20261114_085341_884160"
EXPECTED_HASHES = {
    "train.csv": "20077226f6c642455d134df90e2d0d71879bf143bba94a51b2ef459e795fc9d2",
    "validation.csv": "892ef1651caaf345b1a3b7cd7af7ee754b8e8d285074123e4880f76e511acd76",
    "text_classifier.joblib": "8519ce0188b3a4cf7e56d34f9d5e295b2fc9e70f641042b60e1c1aab9c576324",
    "revised": "187660190d470a6dd327ecac078a146a42c184b20ebc12f210a4321e1f43701f",
}
EXPECTED_PARAMS = dict(class_weight=None, C=1.0, solver="liblinear",
                       max_iter=1000, random_state=42)


def require(condition, message):
    # 不使用 assert：即使 python -O 运行也保留安全检查。
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_input(path, expected, inputs):
    path = path.resolve()
    require(path.is_file(), f"找不到文件：{path}。请核对 --project-root 或修订文件路径。")
    # 在读取内容之前拒绝明显的测试文件路径；不遍历任何数据目录。
    require(not any("test" in part.lower() for part in path.parts),
            f"拒绝可能属于测试集的输入路径：{path}")
    digest = sha256(path)
    require(digest == expected, f"文件 SHA-256 与已核对实验不一致：{path}\n"
            "请恢复对应实验的原文件；不要通过修改脚本哈希绕过检查。")
    inputs[str(path)] = digest
    return path


def read_frame(path, expected_rows):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    require(len(header) == len(set(header)), f"CSV 有重复列名：{path}")
    required = {"source_row_index", "label", "review", "review_key"}
    require(required.issubset(header), f"{path} 缺列：{sorted(required - set(header))}；实际列：{header}")
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False,
                        dtype={"review": str, "review_key": str})
    require(len(frame) == expected_rows, f"{path} 应为 {expected_rows} 行，实际 {len(frame)} 行。")
    require(frame.label.isin([0, 1]).all(), f"{path} 的 label 必须是数值 0/1。")
    require(set(frame.label) == {0, 1}, f"{path} 必须包含正负两类。")
    frame["label"] = frame.label.astype("int64")
    for col in ["source_row_index", "review_key"]:
        require(frame[col].notna().all() and frame[col].astype(str).str.strip().ne("").all()
                and frame[col].is_unique, f"{path} 的 {col} 为空或重复。")
    require(frame.review.str.strip().ne("").all(), f"{path} 含空白评论；禁止静默删除或重排。")
    return frame


def validate_revision(original, revised):
    # 不排序、不 join、不按位置盲目替换标签：所有非标签列必须逐行完全一致。
    require(list(original.columns) == list(revised.columns), "修订集列名或列顺序与原验证集不同。")
    try:
        pd.testing.assert_frame_equal(original.drop(columns="label"),
                                      revised.drop(columns="label"), check_exact=True)
    except AssertionError as exc:
        raise ValueError("修订验证集的行身份、文本或顺序不一致；已停止，未训练。") from exc
    changed = int(original.label.ne(revised.label).sum())
    require(changed == 28, f"修订标签应恰好改变 28 行，实际 {changed} 行。")
    return revised.label.to_numpy()


def evaluate(y, predicted):
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "classification_report": classification_report(y, predicted, labels=[0, 1],
                                                       output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def plot_comparison(results, label_version, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, name in zip(axes, ["plain_lr", "balanced_lr"]):
        matrix = np.asarray(results[name][label_version]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1193)
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha="center", va="center",
                    color="white" if value > 596 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0 Negative", "1 Positive"],
               yticklabels=["0 Negative", "1 Positive"], xlabel="Predicted label",
               ylabel="True label", title=f"{name} / {label_version}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args):
    global matplotlib, plt
    root = Path(args.project_root).expanduser().resolve()
    require(root.is_dir(), f"项目根目录不存在：{root}")
    inputs = {}
    split_dir = root / "data" / "processed" / SPLIT
    train_path = checked_input(split_dir / "train.csv", EXPECTED_HASHES["train.csv"], inputs)
    val_path = checked_input(split_dir / "validation.csv", EXPECTED_HASHES["validation.csv"], inputs)
    model_path = checked_input(root / "experiments" / BASELINE / "text_classifier.joblib",
                               EXPECTED_HASHES["text_classifier.joblib"], inputs)
    train, validation = read_frame(train_path, 9542), read_frame(val_path, 1193)
    for col in ["source_row_index", "review_key"]:
        require(set(train[col]).isdisjoint(validation[col]), f"训练/验证 {col} 存在交集。")
    labels = {"original": validation.label.to_numpy()}
    if args.revised_validation:
        revised_path = checked_input(Path(args.revised_validation).expanduser(),
                                      EXPECTED_HASHES["revised"], inputs)
        labels["revised28"] = validate_revision(validation, read_frame(revised_path, 1193))

    # 只加载已核对哈希的可信本地模型；版本不兼容直接停止。
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        baseline = joblib.load(model_path)
    require(type(baseline) is Pipeline and [name for name, _ in baseline.steps] == ["tfidf", "classifier"],
            "基线必须是两步 Pipeline：tfidf → classifier；不会猜测或替换模型结构。")
    vectorizer, plain = baseline.named_steps["tfidf"], baseline.named_steps["classifier"]
    require(type(vectorizer) is TfidfVectorizer and type(plain) is LogisticRegression,
            "基线步骤类型必须为 TfidfVectorizer 和 LogisticRegression。")
    check_is_fitted(vectorizer)
    check_is_fitted(plain)
    params = plain.get_params(deep=False)
    for key, value in EXPECTED_PARAMS.items():
        require(params.get(key) == value, f"普通 LR 参数 {key} 应为 {value!r}，实际 {params.get(key)!r}。")
    require(np.array_equal(plain.classes_, [0, 1]), "模型类别顺序应为 [0, 1]。")
    vector_hash = joblib.hash(vectorizer)
    plain_hash = joblib.hash(plain)
    X_train = vectorizer.transform(train.review)
    X_val = vectorizer.transform(validation.review)
    for matrix, rows in [(X_train, 9542), (X_val, 1193)]:
        require(matrix.shape == (rows, plain.n_features_in_) and np.isfinite(matrix.data).all(),
                "TF-IDF 矩阵维度与模型不匹配，或包含非有限数值。")
    print(f"预检查通过：训练 {len(train)} 行，验证 {len(validation)} 行，特征 {X_train.shape[1]}。")
    print(f"原模型：{model_path}\n修订标签评估：{'已启用且通过严格校验' if len(labels) == 2 else '未启用'}")
    print(f"Python {platform.python_version()} / scikit-learn {sklearn.__version__}")
    if args.check_only:
        print("只读预检查结束；未训练、未保存文件。")
        return

    # clone 保留包括 tol、penalty、intercept 等在内的全部超参数，仅改变权重。
    balanced = clone(plain).set_params(class_weight="balanced")
    differences = [key for key, value in params.items() if balanced.get_params()[key] != value]
    require(differences == ["class_weight"], f"出现非预期参数差异：{differences}")
    now = datetime.now(timezone(timedelta(hours=8)))
    output = root / "experiments" / (now.strftime("balanced_lr_%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print(f"本次独立输出目录：{output}", flush=True)
    try:
        # 将绘图缓存也限制在本次新目录，预检查不导入绘图模块、不写缓存。
        os.environ["MPLCONFIGDIR"] = str(output / "matplotlib_cache")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        print("仅训练 Balanced Logistic Regression……", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            balanced.fit(X_train, train.label)
        require(np.array_equal(balanced.classes_, [0, 1]), "新模型类别顺序异常。")
        predictions, probabilities, results = {}, {}, {}
        for name, classifier in [("plain_lr", plain), ("balanced_lr", balanced)]:
            predictions[name] = classifier.predict(X_val)
            probabilities[name] = classifier.predict_proba(X_val)[:, 1]
            results[name] = {}
            for version, y in labels.items():
                result = evaluate(y, predictions[name])
                results[name][version] = result
                print(f"\n{name} / {version}: Accuracy={result['accuracy']:.6f}, Macro-F1={result['macro_f1']:.6f}")
                print(classification_report(y, predictions[name], labels=[0, 1],
                      target_names=["0 负面", "1 正面"], digits=6, zero_division=0))
                print("混淆矩阵（行=真实，列=预测；顺序=[0负面, 1正面]）：")
                print(np.asarray(result["confusion_matrix"]))

        # 直接组装已拟合组件；绝不对 Pipeline 调用 fit。
        full = Pipeline([("tfidf", vectorizer), ("classifier", balanced)])
        saved_model = output / "balanced_pipeline.joblib"
        joblib.dump(full, saved_model)
        restored = joblib.load(saved_model)
        require(np.array_equal(restored.predict(validation.review), predictions["balanced_lr"]),
                "重载冒烟测试失败：预测标签不一致。")
        require(np.allclose(restored.predict_proba(validation.review)[:, 1], probabilities["balanced_lr"],
                            rtol=1e-12, atol=1e-12), "重载冒烟测试失败：概率不一致。")
        require(joblib.hash(restored.named_steps["tfidf"]) == vector_hash,
                "保存重载后的 TF-IDF 状态发生变化。")
        require(joblib.hash(vectorizer) == vector_hash and joblib.hash(plain) == plain_hash,
                "原 TF-IDF 或普通 LR 的内存状态发生变化。")
        table = validation.copy()
        if "revised28" in labels:
            table["revised_label"] = labels["revised28"]
        for name in predictions:
            table[f"{name}_predicted_label"] = predictions[name]
            table[f"{name}_positive_probability"] = probabilities[name]
        csv_path = output / "validation_predictions.csv"
        table.to_csv(csv_path, index=False, encoding="utf-8-sig")
        pd.testing.assert_frame_equal(pd.read_csv(csv_path, encoding="utf-8-sig", keep_default_na=False,
            dtype={"review": str, "review_key": str}), table, check_exact=False, rtol=1e-12, atol=1e-12)
        for version in labels:
            plot_comparison(results, version, output / f"confusion_matrix_{version}.png")
        for path, digest in inputs.items():
            require(sha256(path) == digest, f"运行期间输入文件发生变化：{path}")
        (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        metrics = {
            "status": "success", "script_version": VERSION, "time_Asia_Shanghai": now.isoformat(),
            "split_run": SPLIT, "baseline_run": BASELINE, "input_sha256": inputs,
            "rows": {"train": len(train), "validation": len(validation)},
            "features": int(X_train.shape[1]), "class_order": [0, 1],
            "label_names": {"0": "负面", "1": "正面"},
            "confusion_matrix_axes": "rows=true, columns=predicted",
            "parameters": {"plain_lr": params, "balanced_lr": balanced.get_params(deep=False)},
            "changed_parameters": differences, "balanced_iterations": balanced.n_iter_.tolist(),
            "train_class_counts": {str(k): int(v) for k, v in train.label.value_counts().items()},
            "effective_balanced_weights": {str(k): len(train)/(2*int(v)) for k, v in train.label.value_counts().items()},
            "evaluation": results,
            "balanced_minus_plain": {version: {key: results["balanced_lr"][version][key] - results["plain_lr"][version][key]
                for key in ["accuracy", "macro_f1"]} for version in labels},
            "versions": {"python": sys.version, "platform": platform.platform(),
                "executable": sys.executable, "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
                "scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "joblib": joblib.__version__, "matplotlib": matplotlib.__version__},
            "smoke_test": {"all_1193_labels_equal": True, "probabilities_allclose": True,
                           "tfidf_state_unchanged": True, "prediction_csv_roundtrip": True},
            "input_files_unchanged": True, "test_set_read": False, "tfidf_refitted": False,
            "plain_lr_retrained": False, "parameter_search_performed": False,
            "script_sha256": sha256(__file__),
            "output_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
            "interpretation": "固定验证集开发对照；修订28条来自定向审核，只作补充。未调整阈值，概率未经校准。",
        }
        (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"\n完成：{output}\n保存重载冒烟测试通过；原输入文件哈希未变。")
    except Exception as exc:
        (output / "FAILED.txt").write_text(f"实验未完成，请勿将此目录视为成功结果。\n{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", default=r"D:\MealRadar", help="包含固定实验数据与基线的根目录")
    parser.add_argument("--revised-validation", help="可选：已核对的完整 validation_revised28.csv 路径")
    parser.add_argument("--check-only", action="store_true", help="只读校验路径、数据、模型和 transform，不训练或写文件")
    args = parser.parse_args()
    # 延迟导入：--help 在未安装科学计算依赖时也可用。
    global joblib, np, pd, sklearn, scipy, matplotlib, plt
    global Pipeline, TfidfVectorizer, LogisticRegression, clone, check_is_fitted
    global InconsistentVersionWarning, ConvergenceWarning
    global accuracy_score, f1_score, classification_report, confusion_matrix
    try:
        import joblib
        import numpy as np
        import pandas as pd
        import sklearn
        import scipy
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.base import clone
        from sklearn.utils.validation import check_is_fitted
        from sklearn.exceptions import InconsistentVersionWarning, ConvergenceWarning
        from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
        run(args)
        return 0
    except ImportError as exc:
        print(f"依赖加载失败：{exc}\n请在 Anaconda Prompt 中 conda activate mealradar 后重试。", file=sys.stderr)
    except Exception as exc:
        print(f"实验已停止：{type(exc).__name__}: {exc}\n"
              "请按提示核对原文件与原训练环境；不自动重建数据、不换模型、不覆盖旧实验。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

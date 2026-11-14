"""食评雷达的 Streamlit 入口。"""
import json
import hashlib
import csv
from io import StringIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "models" / "mealradar_pipeline.joblib"
METRICS_PATH = Path(__file__).resolve().parent / "reports" / "metrics.json"
LABEL_NAMES = {0: "负面", 1: "正面"}
MAX_COMMENT = 5000
MAX_BATCH = 5000
EXPECTED_MODEL_SHA256 = "8519ce0188b3a4cf7e56d34f9d5e295b2fc9e70f641042b60e1c1aab9c576324"


@st.cache_resource(show_spinner="正在加载情感分类模型…")
def load_model():
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != EXPECTED_MODEL_SHA256:
        raise ValueError("模型文件与已确认的普通逻辑回归不一致。")
    return joblib.load(MODEL_PATH)


def predict(texts):
    model = load_model()
    labels = model.predict(texts)
    positive_index = list(model.classes_).index(1)
    scores = model.predict_proba(texts)[:, positive_index]
    return labels, scores


def read_upload(raw):
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("gb18030")
    try:
        records = list(csv.reader(StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise pd.errors.ParserError("CSV 引号或记录格式错误") from exc
    if not records or not records[0]:
        raise pd.errors.EmptyDataError("CSV 没有表头")
    header = records[0]
    if len(set(header)) != len(header):
        raise pd.errors.ParserError("CSV 表头重复")
    rows = []
    for row in records[1:]:
        if not row:
            row = [""] * len(header)
        if len(row) != len(header):
            raise pd.errors.ParserError("CSV 字段数量与表头不一致")
        rows.append(row)
    return pd.DataFrame(rows, columns=header, dtype=str)


def single_review():
    st.write("粘贴一条外卖评论，查看模型判断的整体情感。")
    positive, negative = st.columns(2)
    if positive.button("试试正面评论", key="example_positive", width="stretch"):
        st.session_state.review_text = "很好吃，送餐速度快，下次还会再点。"
    if negative.button("试试负面评论", key="example_negative", width="stretch"):
        st.session_state.review_text = "太难吃了，送餐太慢，饭菜都凉了。"
    with st.form("single_review"):
        text = st.text_area("评论内容", max_chars=MAX_COMMENT, height=130,
                            key="review_text",
                            placeholder="例如：配送很及时，餐品还热着，味道也不错。")
        submitted = st.form_submit_button("分析评论", type="primary")
    if not submitted:
        return
    if not text.strip():
        st.warning("请先输入一条评论。")
        return
    try:
        with st.spinner("正在分析…"):
            labels, scores = predict([text])
    except FileNotFoundError:
        st.error("模型文件缺失。请将已训练的模型放入 models 文件夹。")
        return
    except Exception:
        st.error("暂时无法加载模型或完成预测，请检查模型文件和依赖版本。")
        return
    label = LABEL_NAMES[int(labels[0])]
    (st.success if label == "正面" else st.warning)(f"预测结果：{label}")
    st.metric("正面估计分数", f"{scores[0]:.1%}")
    st.caption("分数为模型估计值，未进行概率校准。褒贬混合、反讽或缺少语境时，结果可能不准确。")


def safe_csv_cell(value):
    # 在表格软件打开下载文件时，将公式样式的文本保留为文本。
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def batch_reviews():
    st.write("上传包含 `review` 列的 CSV。单次最多 5,000 条，文件最大 10 MB。")
    upload = st.file_uploader("评论 CSV", type=["csv"], max_upload_size=10)
    if upload is None:
        return
    raw = upload.getvalue()
    submitted = st.button("分析整批评论", type="primary")
    cache = st.session_state.get("batch_results")
    if cache is not None and cache[0] == raw:
        results = cache[1]
    elif not submitted:
        return
    else:
        try:
            frame = read_upload(raw)
        except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
            st.error("无法读取该 CSV。请检查分隔符、表头和文件编码。")
            return
        if "review" not in frame.columns:
            st.error("文件需要包含名为 review 的评论列。")
            return
        if not 1 <= len(frame) <= MAX_BATCH:
            st.warning("请上传 1 到 5,000 条评论。")
            return
        valid = frame.review.str.strip().ne("") & frame.review.str.len().le(MAX_COMMENT)
        if not valid.any():
            st.warning("没有可分析的评论。请检查空白内容和单条长度。")
            return
        results = pd.DataFrame({"review": frame.review, "预测情感": "未分析", "正面估计分数": float("nan")})
        try:
            with st.spinner("正在逐条分析…"):
                labels, scores = predict(frame.loc[valid, "review"].tolist())
        except Exception:
            st.error("批量预测未完成。请检查模型文件和依赖版本。")
            return
        results.loc[valid, "预测情感"] = [LABEL_NAMES[int(label)] for label in labels]
        results.loc[valid, "正面估计分数"] = scores
        st.session_state.batch_results = (raw, results)
    counts = results["预测情感"].value_counts().reindex(["正面", "负面", "未分析"], fill_value=0)
    for col, (name, count) in zip(st.columns(3), counts.items()):
        col.metric(name, int(count))
    st.bar_chart(counts.loc[["正面", "负面"]])
    shown = results.loc[results["预测情感"].eq("负面")] if st.checkbox("仅查看负面评论") else results
    st.dataframe(shown, hide_index=True, width="stretch")
    download = results.copy()
    download["review"] = download.review.map(safe_csv_cell)
    st.download_button("下载全部预测结果", download.to_csv(index=False).encode("utf-8-sig"),
                       "mealradar_predictions.csv", "text/csv")
    st.caption("空白或超过 5,000 字符的评论标为未分析。分数为模型估计值。")


st.set_page_config(page_title="食评雷达 MealRadar", layout="centered")
st.title("食评雷达")
st.caption("MealRadar · 面向大学生的外卖评论情感分析")
single, batch, experiment = st.tabs(["单条分析", "批量分析", "模型实验"])
with single:
    single_review()
with batch:
    batch_reviews()
with experiment:
    st.subheader("字符级 TF-IDF + 普通逻辑回归")
    st.write("使用本机已训练的普通逻辑回归，保留原始评论输入，由已保存的 Pipeline 完成特征转换与分类。网站运行时不训练模型。")
    try:
        report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        if report["pipeline_sha256"] != EXPECTED_MODEL_SHA256:
            raise ValueError("评估报告与最终模型不一致")
        scores = report["final_test"]
        a, b, c = st.columns(3)
        a.metric("测试准确率", f"{scores['accuracy']:.2%}")
        b.metric("测试 Macro-F1", f"{scores['macro_f1']:.4f}")
        c.metric("负面召回率", f"{scores['recall_negative']:.2%}")
        st.write(f"以上结果来自固定测试集的 {scores['rows']:,} 条评论，标签 0 为负面，1 为正面。")
        rows = []
        for key, name in [("0", "负面"), ("1", "正面")]:
            item = scores["classification_report"][key]
            rows.append({"类别": name, "Precision": item["precision"], "Recall": item["recall"],
                         "F1": item["f1-score"], "样本数": int(item["support"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.write("混淆矩阵：行是真实标签，列是预测标签。")
        st.table(pd.DataFrame(scores["confusion_matrix_true_rows_pred_columns"],
                             index=["真实负面", "真实正面"], columns=["预测负面", "预测正面"]))
        st.caption(report["test_note"])
        with st.expander("验证集标签审核说明"):
            st.write("原验证集：Accuracy 89.52%，Macro-F1 0.8765。修订 28 条标签后：Accuracy 91.87%，Macro-F1 0.9030。")
            st.write(report["validation_note"])
    except (OSError, ValueError, KeyError, TypeError):
        st.warning("评估报告暂时不可用，单条和批量分析仍可尝试使用。")
    try:
        case = json.loads((METRICS_PATH.parent / "error_case.json").read_text(encoding="utf-8"))
        if case["pipeline_sha256"] != EXPECTED_MODEL_SHA256:
            raise ValueError("错误案例与模型不一致")
        st.markdown("**真实验证错误案例**")
        st.write(case["review"])
        st.write("参考标签：负面；模型预测：正面。")
        st.write(case["analysis"])
        st.caption(case["future_work"])
    except (OSError, ValueError, KeyError, TypeError):
        st.caption("错误案例暂时不可用。")
    st.write("模型判断整体正负情感，类别不区分口味、配送和包装。")
    st.caption("数据来源：ChineseNlpCorpus / waimai_10k。")

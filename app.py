"""食评雷达的 Streamlit 入口。"""
import json
import hashlib
import csv
from io import StringIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
# Character-fragment helpers are embedded so this app deploys as one file.
# Embedded fragment analysis and presentation helpers.
import unicodedata
import numpy as np
import pandas as pd


def model_parts(model):
    vectorizer = model.named_steps['tfidf']
    classifier = model.named_steps['classifier']
    if list(classifier.classes_) != [0, 1] or vectorizer.analyzer != 'char':
        raise ValueError('需要标签为 0/1 的字符级二分类模型')
    return vectorizer, classifier


def visible_fragment(fragment, min_chars=2):
    return min_chars <= len(fragment) <= 3 and all(
        unicodedata.category(c)[0] in 'LN' for c in fragment)


def explain_review(model, text):
    vectorizer, classifier = model_parts(model)
    x = vectorizer.transform([text]).tocsr()
    names = vectorizer.get_feature_names_out()
    contributions = x.data.astype(float) * classifier.coef_[0, x.indices]
    features = pd.DataFrame({'fragment': names[x.indices], 'tfidf': x.data,
                             'weight': classifier.coef_[0, x.indices],
                             'contribution': contributions})
    intercept = float(classifier.intercept_[0])
    decision = float(model.decision_function([text])[0])
    reconstructed = intercept + float(contributions.sum())
    return dict(features=features, intercept=intercept, decision=decision,
                reconstructed=reconstructed, error=abs(reconstructed-decision),
                verified=bool(np.isclose(reconstructed, decision, rtol=1e-7, atol=1e-8)))


def analyze_groups(model, texts, labels, min_chars=2):
    vectorizer, classifier = model_parts(model)
    names = vectorizer.get_feature_names_out()
    eligible = np.array([visible_fragment(s, min_chars) for s in names])
    labels = np.asarray(labels)
    texts = np.asarray(texts, dtype=object)
    if len(texts) != len(labels):
        raise ValueError('评论与标签数量不一致')
    result = {'sizes': {}}
    for name, direction in [('正面', 1), ('负面', -1)]:
        selected = texts[labels == name].tolist()
        n = len(selected)
        result['sizes'][name] = n
        table = pd.DataFrame(columns=['fragment', 'count', 'coverage', 'support'])
        if n:
            x = vectorizer.transform(selected).tocsr()
            counts = np.asarray(x.getnnz(axis=0)).ravel()
            indices = np.flatnonzero(eligible & (counts > 0))
            # Binary LR has a fixed coefficient sign per feature. Absent features
            # contribute zero; denominator includes every analyzed row in the group.
            means = np.asarray(x.mean(axis=0)).ravel().astype(float)
            support = means * np.maximum(direction * classifier.coef_[0], 0)
            table = pd.DataFrame({'fragment': names[indices], 'count': counts[indices],
                                  'coverage': counts[indices] / n, 'support': support[indices]})
            table = table.sort_values(['count', 'fragment'], ascending=[False, True]).reset_index(drop=True)
        result[name] = table
    return result


def matching_rows(model, texts, fragment):
    """Use the fitted analyzer, including its preprocessing, to match originals."""
    analyzer = model.named_steps['tfidf'].build_analyzer()
    return [i for i, text in enumerate(texts) if fragment in analyzer(text)]

# Embedded fragment analysis and presentation helpers.
from io import BytesIO
from pathlib import Path
import os
import math
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


def chinese_font():
    candidates = [os.environ.get('MEALRADAR_FONT', ''),
                  str(Path(__file__).parent / 'assets/NotoSansCJKsc-Regular.otf'),
                  'C:/Windows/Fonts/msyh.ttc',
                  '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']
    return next((p for p in candidates if p and Path(p).is_file()), None)


def cloud_png(frequencies, font_path):
    """Deterministic non-overlapping layout; font area scales with row count."""
    canvas = Image.new('RGB', (800, 380), '#f4f7fb')
    draw = ImageDraw.Draw(canvas)
    boxes = []
    maximum = max(frequencies.values())
    for index, (word, count) in enumerate(frequencies.items()):
        font = ImageFont.truetype(font_path, max(18, round(72 * math.sqrt(count / maximum))))
        bounds = draw.textbbox((0, 0), word, font=font)
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        for step in range(2400):
            angle, radius = step * .21, step * .15
            x = int(400 + math.cos(angle) * radius * 1.7 - width / 2)
            y = int(190 + math.sin(angle) * radius - height / 2)
            box = (x-5, y-5, x+width+5, y+height+5)
            if box[0] < 0 or box[1] < 0 or box[2] > 800 or box[3] > 380:
                continue
            if any(box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1] for b in boxes):
                continue
            draw.text((x-bounds[0], y-bounds[1]), word, font=font,
                      fill=['#126b70', '#3448a0', '#974b30'][index % 3])
            boxes.append(box)
            break
    output = BytesIO()
    canvas.save(output, format='PNG')
    return output.getvalue()



def cloud_svg(frequencies):
    """Browser-rendered Chinese text needs no server-side font file."""
    from html import escape
    positions = [(133, 55), (400, 55), (667, 55), (230, 145), (570, 145),
                 (133, 235), (400, 235), (667, 235), (230, 325), (570, 325)]
    maximum = max(frequencies.values())
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="380" viewBox="0 0 800 380">',
             '<rect width="800" height="380" fill="#f4f7fb"/>']
    for index, ((word, count), (x, y)) in enumerate(zip(frequencies.items(), positions)):
        size = max(20, round(64 * math.sqrt(count / maximum)))
        color = ['#126b70', '#3448a0', '#974b30'][index % 3]
        parts.append(f'<text x="{x}" y="{y}" text-anchor="middle" dominant-baseline="middle" font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif" font-size="{size}" fill="{color}">{escape(word)}</text>')
    return ''.join(parts) + '</svg>'


def display_table(table, contribution=False):
    columns = ['fragment', 'count', 'coverage'] + (['support'] if contribution else [])
    shown = table[columns].rename(columns={'fragment': '字符片段', 'count': '覆盖条数',
        'coverage': '组内覆盖率', 'support': '平均同方向贡献'})
    shown['组内覆盖率'] = shown['组内覆盖率'].map(lambda v: f'{v:.1%}')
    st.dataframe(shown, hide_index=True, width='stretch')


def render_single(model, text):
    st.subheader('模型真实特征贡献')
    result = explain_review(model, text)
    st.caption('每项贡献 = 模型系数 × 本条评论的 TF-IDF。正值推动正面，负值推动负面；这是模型内部的加法解释，不是语义或因果结论。')
    if not result['verified']:
        st.error('贡献校验未通过，暂不展示解释。')
        return
    st.caption(f"校验通过：截距 {result['intercept']:.8f} + 全部贡献 {result['features'].contribution.sum():.8f} = {result['reconstructed']:.8f}；decision_function = {result['decision']:.8f}；误差 {result['error']:.2e}。")
    for col, direction, name in zip(st.columns(2), [1, -1], ['支持正面', '支持负面']):
        with col:
            st.markdown(f'**{name} · Top 10**')
            frame = result['features']
            frame = frame[frame.contribution * direction > 0].sort_values('contribution', ascending=direction < 0).head(10)
            if frame.empty:
                st.info('没有该方向的非零贡献。')
            else:
                st.dataframe(frame[['fragment', 'contribution']].rename(columns={'fragment': '字符片段', 'contribution': '贡献'}), hide_index=True)
    with st.expander('全部非零特征与计算明细（含标点、空白）'):
        st.dataframe(result['features'].rename(columns={'fragment': '字符片段', 'tfidf': 'TF-IDF', 'weight': '模型系数', 'contribution': '贡献'}), hide_index=True)
    st.caption('上方 Top 10 只展示部分特征；校验使用全部非零特征和截距。贡献不等于概率变化。')


def render_batch(model, results, identity):
    st.divider()
    st.subheader('高频片段')
    st.caption('仅使用当前已拟合 TF-IDF 词表中的字符片段，不做分词或完整短语提取。按预测类别分组，未分析行不参与；一行内重复只计一次，重复评论行仍分别计数。排除含标点、空白及符号的片段。')
    mode = st.radio('字符范围', ['2–3 字符（默认）', '全部 1–3 字符'], horizontal=True)
    top = st.selectbox('高频片段数量', [5, 10])
    minimum = 2 if mode.startswith('2') else 1
    cache_key = (identity, minimum)
    cached = st.session_state.get('fragment_cache')
    if cached is None or cached[0] != cache_key:
        with st.spinner('正在统计字符片段…'):
            data = analyze_groups(model, results.review.tolist(), results['预测情感'].tolist(), minimum)
        st.session_state.fragment_cache = (cache_key, data)
    else:
        data = cached[1]
    for col, name in zip(st.columns(2), ['正面', '负面']):
        with col:
            st.markdown(f"**预测{name} · {data['sizes'][name]} 条**")
            if data['sizes'][name] < 10:
                st.caption('样本少于 10 条，覆盖率容易波动，请结合原文。')
            table = data[name].head(top)
            if table.empty:
                st.info('本组暂无符合范围的片段。')
                continue
            display_table(table)
            font = chinese_font()
            if font:
                st.image(cloud_png(dict(zip(table.fragment, table['count'])), font), caption=f'预测{name}片段词云 · 字号按覆盖条数缩放')
            else:
                st.image(cloud_svg(dict(zip(table.fragment, table['count']))), caption=f'预测{name}片段词云 · 字号按覆盖条数缩放')
    union = list(dict.fromkeys(data['正面'].head(top).fragment.tolist() + data['负面'].head(top).fragment.tolist()))
    if union:
        st.markdown('**正负覆盖率对比**')
        comparison = pd.DataFrame({name: data[name].set_index('fragment').coverage.reindex(union, fill_value=0).astype(float)*100 for name in ['正面', '负面']})
        st.bar_chart(comparison, y_label='组内覆盖率（%）', stack=False)
        st.caption('每组各用自己的评论条数作分母；空组显示 0，不表示观察到零覆盖。')
    st.subheader('模型贡献片段')
    st.caption('与高频统计分开：每组按平均同方向贡献排序取 Top 5。正面取 max(w×TF-IDF, 0)，负面取 max(−w×TF-IDF, 0)，再除以该组全部已分析行数（未出现计 0）。数值是对数几率方向的支持强度，不是概率、完整语义或商家质量。字符范围沿用上方选择。')
    contribution_options = []
    for col, name in zip(st.columns(2), ['正面', '负面']):
        with col:
            st.markdown(f'**支持预测{name} · Top 5**')
            table = data[name]
            table = table[table.support > 0].sort_values(['support', 'fragment'], ascending=[False, True]).head(5)
            contribution_options.extend(table.fragment.tolist())
            if table.empty:
                st.info('本组暂无同方向贡献片段。')
            else:
                display_table(table, True)
    options = list(dict.fromkeys(union + contribution_options))
    if options:
        st.subheader('查看匹配的真实原文')
        fragment = st.selectbox('选择片段（来自上述高频 / 贡献结果）', options, key=f'fragment_{identity}_{minimum}_{top}')
        group = st.radio('原文范围', ['全部已分析', '正面', '负面'], horizontal=True)
        original = results[results['预测情感'].isin(['正面', '负面'])].copy()
        if group != '全部已分析':
            original = original[original['预测情感'] == group]
        original.insert(0, '数据行号', original.index + 1)
        indices = matching_rows(model, original.review.tolist(), fragment)
        st.caption(f'共匹配 {len(indices)} 条；数据行号从表头之后的第一条记录开始，保留重复行。')
        st.dataframe(original.iloc[indices], hide_index=True, width='stretch')


MODEL_PATH = Path(__file__).resolve().parent / "models" / "mealradar_pipeline.joblib"
METRICS_PATH = Path(__file__).resolve().parent / "reports" / "metrics.json"
EXAMPLE_CSV_PATH = Path(__file__).resolve().parent / "examples" / "reviews.csv"
LABEL_NAMES = {0: "负面", 1: "正面"}
MAX_COMMENT = 5000
MAX_BATCH = 5000
EXPECTED_MODEL_SHA256 = "f3a14ef2a2003e7c007ed02738bf6568e167ca0d6c8b332c9482ffbde526ff06"


@st.cache_resource(show_spinner="正在加载情感分类模型…")
def load_model(expected_sha256=EXPECTED_MODEL_SHA256):
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("模型文件与已确认的 Balanced 逻辑回归不一致。")
    return joblib.load(MODEL_PATH)


def predict(texts):
    model = load_model(EXPECTED_MODEL_SHA256)
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
    render_single(load_model(), text)


def safe_csv_cell(value):
    # 在表格软件打开下载文件时，将公式样式的文本保留为文本。
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def batch_reviews():
    st.write("上传 CSV 批量分析评论。单次最多 5,000 条，文件最大 10 MB。")
    st.info("CSV 格式要求：第一行是表头，评论所在列的表头必须为小写 `review`，从第二行开始填写评论，每条评论占一条记录。")
    st.caption("想先试用？下载示例 CSV，再在下方上传，点击“分析整批评论”。")
    if EXAMPLE_CSV_PATH.is_file():
        st.download_button("下载示例 CSV（8 条评论）", EXAMPLE_CSV_PATH.read_bytes(),
                           "reviews.csv", "text/csv", key="download_example_csv")
    else:
        st.caption("示例文件暂不可用，你仍可上传自己的 CSV。")
    upload = st.file_uploader("评论 CSV", type=["csv"], max_upload_size=10)
    if upload is None:
        st.session_state.pop('batch_results', None)
        st.session_state.pop('fragment_cache', None)
        return
    raw = upload.getvalue()
    if len(raw) > 10 * 1024 * 1024:
        st.error('文件最大 10 MB。')
        return
    if st.session_state.get('batch_results', (None,))[0] != (EXPECTED_MODEL_SHA256, raw):
        st.session_state.pop('batch_results', None)
        st.session_state.pop('fragment_cache', None)
    submitted = st.button("分析整批评论", type="primary")
    cache = st.session_state.get("batch_results")
    if cache is not None and cache[0] == (EXPECTED_MODEL_SHA256, raw):
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
        st.session_state.batch_results = ((EXPECTED_MODEL_SHA256, raw), results)
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
    st.caption("空白或超过 5,000 字符的评论标为未分析。")
    st.caption("分数说明：“正面估计分数”是模型估计的正面概率，范围为 0～1。越接近 1，模型越倾向正面；越接近 0，越倾向负面。")
    st.caption("当前模型以 0.5 为默认分界：大于 0.5 预测为正面，小于或等于 0.5 预测为负面。表格显示值可能经过四舍五入，分类以模型原始输出为准。")
    st.caption("分数未经独立概率校准，不代表预测一定正确；接近 0.5 的评论处在分类分界附近，建议结合原文判断。")
    render_batch(load_model(), results, EXPECTED_MODEL_SHA256 + hashlib.sha256(raw).hexdigest())


st.set_page_config(page_title="食评雷达 MealRadar", layout="centered")
st.title("食评雷达")
st.caption("MealRadar · 面向大学生的外卖评论情感分析")
single, batch, experiment = st.tabs(["单条分析", "批量分析", "模型实验"])
with single:
    single_review()
with batch:
    batch_reviews()
with experiment:
    st.subheader("字符级 TF-IDF + Balanced 逻辑回归")
    st.write("使用本人在本机训练的 Balanced 逻辑回归（class_weight=balanced），保留原始评论输入，由已保存的 Pipeline 完成特征转换与分类。网站运行时不训练模型。")
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
        st.caption("选择依据：兼顾正负情感，原验证 Macro-F1 提升；负面召回有所下降。")
        with st.expander("验证集标签审核说明"):
            original = report["validation_original"]
            revised = report["validation_revised28"]
            st.write(f"原验证集：Accuracy {original['accuracy']:.2%}，Macro-F1 {original['macro_f1']:.4f}。"
                     f"修订 28 条标签后：Accuracy {revised['accuracy']:.2%}，Macro-F1 {revised['macro_f1']:.4f}。")
            st.write("修订标签仅用于补充分析，同一组预测保持不变；分数变化不代表模型训练效果提升。")
    except (OSError, ValueError, KeyError, TypeError):
        st.warning("评估报告暂时不可用，单条和批量分析仍可尝试使用。")
    try:
        case = json.loads((METRICS_PATH.parent / "error_case.json").read_text(encoding="utf-8"))
        if case["pipeline_sha256"] != EXPECTED_MODEL_SHA256:
            raise ValueError("错误案例与模型不一致")
        st.markdown("**真实验证错误案例**")
        st.write(case["review"])
        st.write(f"参考标签：{LABEL_NAMES[case['reference_label']]}；模型预测：{LABEL_NAMES[case['predicted_label']]}。")
        st.write(case["analysis"])
        st.caption(case["future_work"])
    except (OSError, ValueError, KeyError, TypeError):
        st.caption("错误案例暂时不可用。")
    st.write("模型判断整体正负情感，类别不区分口味、配送和包装。")
    st.caption("数据来源：ChineseNlpCorpus / waimai_10k。")

# 食评雷达 MealRadar

面向大学生的外卖评论整体正负情感分析应用：单条分析、批量CSV分析、预测分布统计、负面筛选与结果下载。当前本地网页采用本人2026-11-17在mealradar环境训练的**Balanced Logistic Regression**，2026-11-19决定采用。网站加载已保存Pipeline，不在线训练、不调用外部情感分析API。

## 本地运行

Python 3.11，依赖版本见requirements.txt。在本地网页项目目录执行：

```bat
conda activate mealradar
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

模型位于models/mealradar_pipeline.joblib。运行目录应包含app.py，不要与D:\MealRadar训练实验目录混淆。分步验证、上传清单和回退方法见LOCAL_VALIDATION.md。

单条最多5000字符；批量CSV的评论列表头为review，支持UTF-8/GB18030，最多5000行、10MB。空白或过长评论标为未分析。输出分数未经独立概率校准，不能解释为预测正确率。

## 当前模型与选择依据

TF-IDF：字符1～3 gram、lowercase=False、float32、155357个特征，仅在训练集拟合。LR：class_weight='balanced'、C=1.0、solver=liblinear、max_iter=1000、random_state=42；其余参数继承普通LR。0=负面，1=正面。

SHA-256：f3a14ef2a2003e7c007ed02738bf6568e167ca0d6c8b332c9482ffbde526ff06；大小4,201,952字节。

原验证Macro-F1由普通LR的0.8765提高到0.8853；正面召回由75.88%提高到83.67%，负面召回由96.35%降至92.96%。按兼顾两类的产品目标采用Balanced，不宣称全面优于普通LR或已证明其他分布上的效果。详细对照及执行来源见BALANCED_EXPERIMENT.md。

## 数据与评估

来源：[ChineseNlpCorpus / waimai_10k](https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/waimai_10k)。原始11987条，冲突审核、排除歧义和去重后11928条；按标签分层、种子42划分训练/验证/测试为9542/1193/1193条。来源行号和规范化去重键无跨集合交集；未声称排除所有语义、用户或商家层面的泄漏。

| Balanced LR评估口径 | Accuracy | Macro-F1 | 来源 |
|---|---:|---:|---|
| 原验证标签 | 89.86% | 0.8853 | 用户2026-11-17运行 |
| 修订28条验证标签 | 92.04% | 0.9090 | 同哈希模型历史报告，补充分析 |
| 原测试标签 | 89.94% | 0.8862 | 助手2026-11-14历史评估 |

修订验证为基于已知误判的定向审核，非全面盲审；同一组预测只更换参考标签，不能作为训练提升证据。

| 历史测试类别 | Precision | Recall | F1 | 样本数 |
|---|---:|---:|---:|---:|
| 负面 | 0.9204 | 0.9296 | 0.9250 | 796 |
| 正面 | 0.8560 | 0.8388 | 0.8473 | 397 |

测试混淆矩阵[[740,56],[64,333]]，行真实、列预测，顺序负面/正面。网页报告reports/metrics.json与图reports/confusion_matrix.png均为当前Balanced模型口径。

**历史与贡献：** 助手曾于2026-11-14额外训练朴素贝叶斯与Balanced，并执行两次Balanced测试评估。此后用户曾采用普通LR。2026-11-17用户亲自训练并复现Balanced，其模型文件哈希与历史模型完全一致；2026-11-19用户选择切换。用户本次训练未读取测试集，本次网页迁移也未重新训练或评估测试集。不能声称测试集此前从未接触，亦不利用测试成绩继续调参。

普通LR模型与网页备份留在本地，原报告保留于reports/history/plain_lr_metrics_20261114.json；Balanced历史测试原报告保留于reports/history/balanced_test_20261114.json。旧记录只代表当时状态，不等于当前部署选择。

## 错误分析与限制

验证样本9447：“就还行，没有很好吃”，参考负面，当前Balanced仍预测正面。最长3字符片段难以整体表示复杂否定范围，局部正面片段可能干扰；这是解释性分析，并非因果证明。详见reports/error_case.json及reports/error_analysis.csv。

模型对否定、转折、反讽、褒贬混合与标签噪声仍有限制；只判断整体情感，不识别口味/配送维度，不抓取平台评论、不做商家排名。预测分布统计不保证等于真实情感占比。

## 代码与同步

保留清洗、划分、特征和train_baseline.py原流程；新增train_balanced_lr_local.py独立对照脚本。复现实验需本机原数据与固定基线，不要直接运行所有历史脚本。部署只需已训练模型。

仓库：https://github.com/a0070513/MealRadar 。当前修改仅在本地，GitHub由用户验证后手动同步；本次未部署，公网状态未核验。同步清单见LOCAL_VALIDATION.md，MANIFEST.json列出本次文件SHA-256。不要使用旧delivery/source_review覆盖新版。

原始/派生评论、逐条预测和审核记录保留本地，数据与分发说明见data/README.md。AI辅助脚本、网页、评估及材料；本人负责需求决策、人工审核、本机训练与最终模型选择。

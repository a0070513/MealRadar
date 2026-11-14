# 食评雷达 MealRadar

面向大学生的外卖评论情感分析应用：输入评论或上传 CSV，查看整体正负情感，统计分布、筛选负面评论并下载结果。网站使用本人在本机训练的**普通逻辑回归**，不调用外部情感分析 API，也不在网站启动时训练。

## 运行

Python 3.11，依赖固定在 requirements.txt，与训练环境一致。在项目目录执行：

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

本机已配置 mealradar 环境可直接执行第二条。完整模型位于 `models/mealradar_pipeline.joblib`。单条最多 5000 字符；批量 CSV 使用 `review` 列，支持 UTF-8/GB18030，最多 5000 行、10 MB，空白或过长行标为未分析。分数是未独立校准的模型估计值。

## 最终模型

原文件：`D:/MealRadar/experiments/baseline_20261114_085341_884160/text_classifier.joblib`。
SHA-256：`8519ce0188b3a4cf7e56d34f9d5e295b2fc9e70f641042b60e1c1aab9c576324`；大小 4,201,952 字节。

TF-IDF：字符 1～3 gram，lowercase=False，dtype=float32，155357 个特征，仅在训练集拟合。
LogisticRegression：class_weight=None，C=1.0，solver=liblinear，max_iter=1000，random_state=42。
0=负面，1=正面。网站输入评论原文，调用已保存的完整 Pipeline。

## 数据与评估

来源：[ChineseNlpCorpus / waimai_10k](https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/waimai_10k)。11987 条经冲突审核、排除歧义组和去重后剩余 11928 条。按标签分层、种子 42，训练/验证/测试分别为 9542/1193/1193 条。来源行号和规范化去重键无跨集合交集；未声称排除了所有语义、用户或商家层面的泄漏。

| 普通 LR 评估口径 | Accuracy | Macro-F1 |
|---|---:|---:|
| 原验证标签 | 89.52% | 0.8765 |
| 修订 28 条验证标签（补充分析） | 91.87% | 0.9030 |
| 原测试标签 | 90.03% | 0.8816 |

验证修订是定向审核，不是全面盲审；只改变参考标签，没有训练或改变预测。

| 测试类别 | Precision | Recall | F1 | 样本数 |
|---|---:|---:|---:|---:|
| 负面 | 0.8886 | 0.9724 | 0.9286 | 796 |
| 正面 | 0.9317 | 0.7557 | 0.8345 | 397 |

混淆矩阵为 `[[774,22],[97,300]]`，行是真实、列是预测，顺序负面/正面。完整结果在 `reports/metrics.json`，图在 `reports/confusion_matrix.png`。

**实验历史：** 助手此前额外执行过朴素贝叶斯和加权逻辑回归实验，并两次评估加权 LR 测试成绩。最终用户明确采用本人训练的普通 LR。本次固定参数后补齐普通 LR 测试，不根据测试分数重新选模型或调参。不能将本测试描述为从未接触过的盲测，也不能把历史加权指标归给普通模型。

## 错误分析与限制

验证样本 9447：“就还行，没有很好吃”，参考标签负面、预测正面。最长 3 字符片段不能完整表达“没有很好吃”的否定范围，局部正面特征可能造成干扰。这是解释性分析，并非因果证明。详见 `reports/error_case.json` 及 `reports/error_analysis.csv`。

正面召回相对较低；标签有噪声，对否定、转折、反讽和褒贬混合也有限制。只判断整体情感，不识别口味/配送维度，不抓取平台数据、不做商家排名。

## 代码与复现

保留原脚本：review_rules_check.py、apply_review_rules.py、clean_and_save.py、split_dataset.py、inspect_text_features.py、train_baseline.py。流程为清洗→分层划分→训练集拟合 TF-IDF→普通 LR。脚本默认读取本机 D:/MealRadar 与已有运行编号；新环境复现时需配置数据位置及各阶段实际输出编号。不要直接运行所有脚本覆盖旧成果。

部署使用现成模型，无需训练。原始数据、派生评论及逐条审核记录保留本地。本次没有重新训练或清洗。prepare_plain_lr.py 是本地迁移评估工具，有重复执行保护，不是网站入口；旧 finalize_submission.py 不再使用。

```bash
python -m unittest discover -s tests -v
```

## 发布状态

GitHub 仓库：https://github.com/a0070513/MealRadar（当前为私有仓库）。本仓库仅包含代码和汇总材料，模型暂未上传，因此尚不是完整可运行的云端交付。公网部署、Linux 依赖安装和公网交互待完成，步骤见 DEPLOYMENT.md。

上游数据说明称原作者及来源不详，本次未找到明确再分发许可。原始/派生评论不公开打包；模型包含训练词表，暂不公开上传，分发条件待确认。delivery/source_review/ 是不含模型的代码审阅包，不能单独作为可预测网站发布。本机继续使用已核对模型。详见 data/README.md。

本人负责需求决策、标签审核及本机普通 LR 训练运行；AI 辅助脚本、网页、评估及材料整理。历史额外实验由助手执行。



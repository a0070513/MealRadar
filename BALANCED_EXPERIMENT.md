# Balanced LR 对照与采用记录

2026-11-19，用户明确决定网页采用 Balanced Logistic Regression，以兼顾正负情感识别。
选择依据为产品目标与原标签验证结果；已知历史测试成绩，不能把本次切换描述为新的盲测。

| 原验证集（1193条） | 普通 LR | Balanced LR |
|---|---:|---:|
| Accuracy | 0.8952221291 | 0.8985750210 |
| Macro-F1 | 0.8765473491 | 0.8852896466 |
| 负面 Recall | 0.9635220126 | 0.9295597484 |
| 正面 Recall | 0.7587939698 | 0.8366834171 |
| 混淆矩阵（行真实、列预测，负面/正面） | [[766,29],[96,302]] | [[739,56],[65,333]] |

Balanced 改正31条正面误判，新增27条负面误判，净减少4条错误。不是全面改善；若目标改为尽量找全差评，普通LR仍有更高负面召回。不能据此保证其他数据分布上的表现或情感占比估计准确。

固定划分 split_20261113_085939_408201：训练9542条、验证1193条。仅复用基线已拟合TF-IDF的transform，155357个特征；clone原分类器后只将class_weight改为balanced，C=1.0、solver=liblinear、max_iter=1000、random_state=42及其余参数不变。训练类别负面6363、正面3179；对应权重0.7498035518、1.5007864108。

## 执行来源与历史

- 2026-11-14，助手于11:27、12:11执行额外模型比较训练，于11:29、12:12执行Balanced测试评估。保留原溯源记录，不能称只测试过一次。
- 2026-11-14，用户当时选用普通LR；普通LR报告现保留于reports/history/plain_lr_metrics_20261114.json。
- 2026-11-17 13:40，用户在mealradar环境亲自执行train_balanced_lr_local.py，运行编号balanced_lr_20261117_134056_068711_3aca76e7。该次只训练和验证，未启用修订标签评分，未读取测试集。
- 本次模型与两份历史Balanced模型SHA-256完全一致。原验证集逐条标签一致，与第二次历史比较的逐条概率差值为0；不是一份新的独立泛化证据。
- 2026-11-19，按用户决定切换本地网页并整理仓库文件；不重新训练、不读取测试数据、不执行测试集推理、不上传、不部署。

普通LR SHA-256：8519ce0188b3a4cf7e56d34f9d5e295b2fc9e70f641042b60e1c1aab9c576324。

Balanced SHA-256：f3a14ef2a2003e7c007ed02738bf6568e167ca0d6c8b332c9482ffbde526ff06。

历史Balanced测试报告为reports/history/balanced_test_20261114.json，1193条、Accuracy 0.8994132439、Macro-F1 0.8861641221、混淆矩阵[[740,56],[64,333]]。网页复用这份与模型哈希对应的历史结果，不冒充用户本次运行的测试成绩。历史修订28条验证成绩为Accuracy 0.9203688181、Macro-F1 0.9089799344，仅作定向标签审核补充分析。

## 复现与数据要求

train_baseline.py保留普通LR基线流程。train_balanced_lr_local.py是独立对照脚本，不是网站入口。它需要本机完整实验输入，只有仓库源码不能重建原始数据。

```bat
conda activate mealradar
cd /d D:\MealRadar
python train_balanced_lr_local.py --check-only
```

完整训练命令、可选修订标签和输出说明可用python train_balanced_lr_local.py --help查看。已经完成实验时，无需为部署再次训练。

汇总对照保存在reports/balanced_comparison.json；逐条含评论原文的预测保留本地。旧交接文档描述的是各自日期状态，本文件记录最新采用决定，不改写历史。

# Balanced 本地验证与手动同步

2026-11-19完成本地切换。操作只针对网页目录；训练实验仍在D:\MealRadar，未改写原始模型或实验。

## 1. 进入正确目录

如果旧网页正在运行，先在启动它的终端按Ctrl+C停止，避免误看旧进程。打开Anaconda Prompt：

```bat
conda activate mealradar
cd /d "C:\Users\29577\Documents\ChatGPT\项目组招新"
```

这是本次修改的网页目录，不是D:\MealRadar。不要重新运行训练脚本。

## 2. 自动检查

```bat
python -m unittest discover -s tests -v
```

预期8项检查通过，最后显示OK。检查模型/报告归属、真实验证错误案例、页面指标、单条示例、批量预测、缺列、错误CSV、空白行和模型缺失提示；不训练、不读取测试数据。测试框架可能打印missing ScriptRunContext提示，以最终OK或FAILED为准。若FAILED，先保留完整报错，不继续上传。

可选核对模型指纹：

```bat
certutil -hashfile models\mealradar_pipeline.joblib SHA256
```

预期f3a14ef2a2003e7c007ed02738bf6568e167ca0d6c8b332c9482ffbde526ff06。

## 3. 启动网页

```bat
python -m streamlit run app.py
```

打开终端显示的Local URL（通常http://localhost:8501）。终端保持运行。

## 4. 验证“模型实验”页

- 标题为“字符级 TF-IDF + Balanced 逻辑回归”。
- 测试准确率89.94%，测试Macro-F1 0.8862，负面召回92.96%。
- 测试混淆矩阵[[740,56],[64,333]]。
- 页面只展示评估结果及必要解释；完整执行历史保留在BALANCED_EXPERIMENT.md与reports/metrics.json中。
- 展开验证说明：原验证89.86% / 0.8853；历史修订标签92.04% / 0.9090。不要将这些分数当作同一评估。
- 错误案例“就还行，没有很好吃”：参考负面、模型正面；不存在“报告暂时不可用”。

## 5. 验证单条与批量

1. 点击“试试正面评论”，再点“分析评论”，应显示正面。
2. 点击“试试负面评论”，再点“分析评论”，应显示负面。
3. 只输入空格后分析，应提示输入评论。
4. 批量页下载示例CSV，上传该文件并点击分析。应显示8条结果；开启负面筛选后只显示负面，下载仍包含全部结果。
5. 实际打开下载的CSV，检查中文显示、review列、预测情感和正面估计分数。自动检查模拟了上传控件，真实浏览器选择文件及下载由这一步验证。
6. 刷新网页后再分析一次，确认仍为新版本。

## 6. GitHub同步范围（验证完成后由用户执行）

本次未上传或部署。可使用delivery/balanced_20261119干净同步目录，或同名ZIP解压后上传。保持文件夹结构，不要将ZIP本身当作网站源码，也不要把delivery外层目录上传为仓库根目录。清单中包含原有复现源码，便于完整同步；MANIFEST.json包含全部同步文件的SHA-256（不含清单自身）。

本次改动或新增：

- app.py、models/mealradar_pipeline.joblib、models/README.md。
- reports/metrics.json、confusion_matrix.png、error_case.json、error_analysis.csv。
- reports/validation_comparison.png、balanced_comparison.json。
- reports/history/plain_lr_metrics_20261114.json、plain_lr_confusion_matrix.png、balanced_test_20261114.json。
- tests/test_delivery.py、train_balanced_lr_local.py。
- README.md、SUBMISSION_INTRO.md、DEPLOYMENT.md、BALANCED_EXPERIMENT.md、LOCAL_VALIDATION.md、data/README.md、MANIFEST.json、.gitignore。

模型、网页、报告和测试需一起同步；只换模型会触发旧代码的哈希检查。 .gitignore仅放行指定部署模型，其他模型和本地实验继续排除。网页端上传时确认点号开头的.gitignore也被包含。

不要上传整个工作区：local_archive、final_test_private、model_comparison*、validation_revision_28、数据CSV、逐条预测及旧delivery/source_review不属于同步范围。保留仓库中其他未涉及文件，不主动删除。上传前确认是否有已连接仓库的自动部署服务；本次助手没有操作这些服务。

## 保留和回退

原普通LR实验文件仍位于D:\MealRadar\experiments\baseline_20261114_085341_884160\text_classifier.joblib。

切换前网页完整副本位于local_archive/before_balanced_20261119，backup_manifest.json记录每个备份文件的哈希。旧历史目录与溯源记录未改写。

如要回退，停止网页后，将该备份中的app.py、models、reports、tests及原说明文件按原相对位置一并恢复，再启动检查；不能只恢复模型。新Balanced对照记录可作为历史保留，回退时应补充采用状态。不要运行旧finalize_submission.py来回退，也不要改写D:\MealRadar原始实验。

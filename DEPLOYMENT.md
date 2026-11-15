# 发布准备与待办

已建立私有仓库 https://github.com/a0070513/MealRadar 。采用 Streamlit 和现有 Python 模型。

1. GitHub 账户与仓库已建立。上传范围为 delivery/source_review/ 中的代码与汇总材料，不上传整个工作区。
2. 数据源未提供已确认的模型分发条件。
3. 条件明确后，将已确认的普通 LR 放到 models/mealradar_pipeline.joblib，核对 README 中哈希。
4. 在 Streamlit Community Cloud 连接仓库，选择实际分支、app.py 和 Python 3.11。使用 requirements.txt 的训练环境版本；安装失败先查看日志，不随意升级/降级模型依赖。
5. 部署成功后记录真实地址，通过公网验收两类示例、空输入、连续预测、刷新、批量上传/筛选/下载。本地自动检查不能代替公网验收。
6. 在最终提交填入网站与仓库链接，更新工作介绍的部署状态。

[官方部署说明](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)

本机在项目目录、mealradar 环境运行：

```text
python -m streamlit run app.py
```

本机地址通常是 http://localhost:8501，以实际启动输出为准。



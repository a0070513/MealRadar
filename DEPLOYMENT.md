# 本地验证与后续同步

2026-11-19：本地默认模型切换为用户亲自训练的Balanced LR。本轮不上传、不部署。当前公网运行状态未核验。

1. 先按LOCAL_VALIDATION.md运行检查并手动验证网页。
2. 同步当前目录或delivery/balanced_20261119中列出的文件，保留目录结构；不要使用旧delivery/source_review快照。GitHub仓库已有模型，新版上传需同时更新模型、app.py、报告与测试。
3. 模型应位于models/mealradar_pipeline.joblib，SHA-256见models/README.md。普通LR备份及原始数据留在本地。数据分发说明见data/README.md。
4. GitHub上传由用户亲自执行；上传可能触发已有托管服务的自动更新，上传前确认所连服务状态。本次没有操作托管平台。
5. 后续若部署，使用app.py、Python 3.11和requirements.txt固定依赖；不要为部署重新训练或重跑测试集。重启应用使新模型生效，再验收单条、批量、筛选、下载和模型报告。
6. 部署验证后再记录真实网址；本次本地检查不代表公网验收完成。

本地命令：python -m streamlit run app.py。地址以终端实际输出为准，通常为http://localhost:8501。

# 数据来源与分发状态

从 [ChineseNlpCorpus / waimai_10k](https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/waimai_10k) 获取原始数据，review 为原文、label 为情感，0 负面、1 正面。本次原始数据保存在 D:/MealRadar/data/raw，处理结果和审核材料也保留本地，不附入代码审阅包。

2026-11-14 核查：[数据说明](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/waimai_10k/intro.ipynb) 称数据在网上搜集，具体作者、来源不详。仓库 README 和该说明中未找到明确再分发授权，常规 LICENSE 地址返回 404。这不证明所有位置都没有许可，也不等于模型必然不能使用；当前是分发条件尚未确认。

2026-11-19 更新：此前“不含模型”的描述针对旧代码审阅包。当前仓库已有部署模型，本次本地同步包也包含已选Balanced模型；助手未执行上传或部署。模型含训练词表，前述许可核查结论没有因本次技术迁移而改变。原始/派生评论CSV及完整审核资料仍只保留本地；同步包只附汇总指标和已有短错误案例。


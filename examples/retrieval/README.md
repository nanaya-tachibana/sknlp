数据来源: https://github.com/IceFlameWorm/NLP_Datasets

1. bert-12, chinese_roberta_wwm_ext_L-12_H-768_A-12, 学习率 2e-5
2. bert-3l, RoBERTa-tiny3L768-clue, 学习率 2e-4

评估方法: [Spearman correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)

| 方法\数据集     | LCQMC | BQ                    |
| --------------- | ----- | --------------------- |
| bert-12         | 59.11 | 31.88                 |
| bert-12(无监督) | 70.87 | 43.04                 |
| bert-3l(有监督) | 76.21 | 59.03                 |
| bert-12(有监督) | 77.82 | 56.81(大概是过拟合了) |

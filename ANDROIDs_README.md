# Androids 语音研究模型

训练目标：Androids 数据集的抑郁相关患者组（PT）与健康对照组（HC）。患者组包含不同诊断，包括双相障碍抑郁期；不能称为单相抑郁模型或个人诊断工具。

使用同一篇意大利语朗读材料的 112 段录音。按数据集提供的 `fold-lists.csv` 划分：前四折 90 人训练，第五折 22 人留出测试。模型超参数仅在训练组内部四折交叉验证选择。测试组有患者组 7 人、健康对照 15 人；AUC 0.848、平衡准确率 0.829、灵敏度 0.857、特异度 0.800。测试样本少，指标不稳定；不能代表中文、其他录音任务或真实临床筛查效果。

依赖：`numpy scipy scikit-learn`。在本目录运行：

```text
python androids_screen.py train "F:\数模真题\Androids-Corpus" androids_model.pkl
python androids_screen.py predict androids_model.pkl "录音文件.wav"
```

预测输入只支持 16-bit PCM WAV。输出 `patient_group_score_0_100` 是研究用模型分数，不是患病概率。测试集已经用于本次评估，后续调整特征或模型后需要新的独立受试者来检验泛化能力。原始 Androids 录音保留在 F 盘，未复制到交付文件夹；使用时遵守数据集的学术非商业及不得再分发条款。

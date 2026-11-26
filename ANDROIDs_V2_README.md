# Androids 语音分类实验 V2

研究目标是 Androids Corpus 意大利语朗读任务中的混合患者组（PT，含多种抑郁相关诊断）与健康对照组（HC）区分，不用于个人诊断。运行时须将 `androids_screen_v2.py` 与 `voice_screen.py` 放在同一目录。

## 改进内容

- 保留官方第 5 折（22 人）作为测试折；仅在第 1–4 折（90 人）内部按官方折比较三组特征及逻辑回归正则化参数。
- 对新录音检查时长、音量和削波；不足 5 秒或音量过低时拒绝评分。
- 输出混淆矩阵，并可导出测试折逐样本 CSV，便于分析误判。

## 运行

```text
python androids_screen_v2.py train "F:\数模真题\Androids-Corpus" androids_model_v2.pkl --report androids_test_v2.csv
python androids_screen_v2.py predict androids_model_v2.pkl "录音文件.wav"
```

输入须为 16-bit PCM WAV。训练选择仍为全部特征、C=0.1；训练集内部按官方折交叉验证平均 AUC 为 0.792。第 5 折 AUC 为 0.848、平衡准确率为 0.829、灵敏度为 0.857、特异度为 0.800；混淆矩阵 TN=12、FP=3、FN=1、TP=6。与旧版第 5 折成绩相同，**没有证据表明本次改动提升了分类性能**。

第 5 折结果此前已被查看，新版对它的结果只能作为探索性比较，不能再称作全新独立验证。输出的 0–100 分是未校准的模型分数，不是患病概率；意大利语实验结果不能推广至中文。原始录音、逐样本测试表和模型文件不随公开代码仓库发布。

"""Exploratory Androids reading-task model; not a clinical diagnostic tool."""
import argparse
import csv
import json
import pickle
import wave
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from voice_screen import features


def quality(path):
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError("仅支持 16-bit PCM WAV")
        sr, channels, count = f.getframerate(), f.getnchannels(), f.getnframes()
        if sr <= 0 or channels <= 0 or count == 0:
            raise ValueError("空录音或无效采样率")
        x = np.frombuffer(f.readframes(count), dtype="<i2").reshape(-1, channels).mean(axis=1)
    duration = count / sr
    if duration < 5:
        raise ValueError("录音不足 5 秒，无法评估")
    rms = float(np.sqrt(np.mean((x / 32768.0) ** 2)))
    peak = float(np.max(np.abs(x)) / 32768.0)
    warnings = []
    if rms < 0.005:
        warnings.append("音量过低或静音过多")
    if peak > 0.99:
        warnings.append("可能存在削波失真")
    return {"duration_s": round(duration, 1), "rms": round(rms, 4),
            "peak": round(peak, 4), "warnings": warnings}


def make_model(c):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler(), LogisticRegression(C=c, max_iter=2000,
                         class_weight="balanced"))


def records_for(root):
    folds = {}
    with (root / "fold-lists.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = csv.reader(f)
        next(rows)
        next(rows)
        for row in rows:
            for k in range(5):
                name = row[k].strip().strip("'") if k < len(row) else ""
                if name:
                    folds[name] = k + 1
    records = []
    for group, label in (("HC", 0), ("PT", 1)):
        for wav in sorted((root / "Reading-Task" / "audio" / group).glob("*.wav")):
            if wav.stem not in folds:
                raise ValueError(f"缺少官方分折: {wav.name}")
            records.append((wav, label, folds[wav.stem]))
    return records


def evaluate(y, score):
    tn, fp, fn, tp = confusion_matrix(y, score >= 0.5, labels=[0, 1]).ravel()
    return {"auc": round(float(roc_auc_score(y, score)), 3),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, score >= 0.5)), 3),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "sensitivity": round(float(tp / (tp + fn)), 3),
            "specificity": round(float(tn / (tn + fp)), 3)}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("dataset_dir")
    t.add_argument("model_path")
    t.add_argument("--report", help="可选：逐样本测试 CSV 输出路径")
    p = sub.add_parser("predict")
    p.add_argument("model_path")
    p.add_argument("wav")
    args = ap.parse_args()
    if args.command == "predict":
        with open(args.model_path, "rb") as f:
            saved = pickle.load(f)  # 仅加载可信来源的模型文件
        q = quality(args.wav)
        if q["rms"] < 0.005:
            raise ValueError("录音音量过低，已拒绝评分")
        cols = saved["columns"]
        score = float(saved["model"].predict_proba([np.asarray(features(args.wav))[cols]])[0, 1])
        print(json.dumps({"patient_group_score_0_100": round(100 * score, 1),
                          "audio_quality": q,
                          "note": "仅为研究用模型输出；不是患病概率或临床诊断"},
                         ensure_ascii=False, indent=2))
        return

    records = records_for(Path(args.dataset_dir))
    X = np.asarray([features(wav) for wav, _, _ in records])
    y = np.asarray([label for _, label, _ in records])
    fold = np.asarray([k for _, _, k in records])
    train, test = fold != 5, fold == 5
    if len(records) != 112 or len(set(y[test])) != 2:
        raise ValueError("数据数量或官方第五折类别不符合预期")

    # All model selection uses only official folds 1-4; fold 5 is evaluated once.
    candidates = [("all", np.arange(X.shape[1])),
                  ("acoustic_only", np.arange(1, 9)),
                  ("no_duration", np.arange(1, X.shape[1]))]
    choices = []
    for name, cols in candidates:
        for c in (0.01, 0.1, 1.0):
            scores = []
            for k in (1, 2, 3, 4):
                tr, val = train & (fold != k), fold == k
                model = make_model(c)
                model.fit(X[tr][:, cols], y[tr])
                scores.append(float(roc_auc_score(y[val], model.predict_proba(X[val][:, cols])[:, 1])))
            choices.append((float(np.mean(scores)), name, c, cols))
    cv_auc, name, c, cols = max(choices, key=lambda item: item[0])
    model = make_model(c)
    model.fit(X[train][:, cols], y[train])
    score = model.predict_proba(X[test][:, cols])[:, 1]
    report = {"target": "Androids混合患者组PT vs健康对照HC",
              "train_people": int(train.sum()), "test_people": int(test.sum()),
              "test_patient_group": int(y[test].sum()), "test_controls": int((1-y[test]).sum()),
              "selected_features": name, "selected_C": c,
              "training_cv_auc": round(cv_auc, 3), **evaluate(y[test], score),
              "note": "第五折结果已被项目查看；新版结果仅为探索性比较，不能称为全新独立验证"}
    with open(args.model_path, "wb") as f:
        pickle.dump({"model": model, "columns": cols, "report": report}, f)
    if args.report:
        with open(args.report, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "true_group", "score_0_100", "predicted_group_at_50", "correct"])
            for i in np.where(test)[0]:
                s = float(model.predict_proba([X[i, cols]])[0, 1])
                pred = int(s >= 0.5)
                writer.writerow([records[i][0].name, "PT" if y[i] else "HC", round(100*s, 1),
                                 "PT" if pred else "HC", int(pred == y[i])])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Androids reading-task research baseline: depression-related patients vs controls."""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from voice_screen import features


def pipeline(c):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler(), LogisticRegression(C=c, max_iter=2000,
                         class_weight="balanced"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("dataset_dir")
    t.add_argument("model_path")
    p = sub.add_parser("predict")
    p.add_argument("model_path")
    p.add_argument("wav")
    a = ap.parse_args()
    if a.command == "predict":
        with open(a.model_path, "rb") as f:
            saved = pickle.load(f)
        score = float(saved["model"].predict_proba([features(a.wav)])[0, 1])
        print(json.dumps({"patient_group_score_0_100": round(score * 100, 1),
                          "test_metrics": saved["test_metrics"],
                          "note": "研究用患者组相似度；不是抑郁症诊断或患病概率"},
                         ensure_ascii=False, indent=2))
        return
    root = Path(a.dataset_dir)
    fold_file = root / "fold-lists.csv"
    folds = {}
    with open(fold_file, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
        next(reader)
        for row in reader:
            for index in range(5):
                name = row[index].strip().strip("'") if index < len(row) else ""
                if name:
                    folds[name] = index + 1
    records = []
    for group, label in (("HC", 0), ("PT", 1)):
        for wav in sorted((root / "Reading-Task" / "audio" / group).glob("*.wav")):
            stem = wav.stem
            if stem not in folds:
                raise ValueError(f"缺少官方分组: {stem}")
            records.append((wav, label, folds[stem]))
    X, y, fold = [], [], []
    for i, (wav, label, k) in enumerate(records, 1):
        X.append(features(wav))
        y.append(label)
        fold.append(k)
        if i % 20 == 0:
            print(f"已处理 {i}/{len(records)} 段录音", file=sys.stderr, flush=True)
    X, y, fold = np.asarray(X), np.asarray(y), np.asarray(fold)
    train, test = fold != 5, fold == 5
    if len(set(y[test])) != 2:
        raise ValueError("官方第 5 折测试集缺少某一类")
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    choices = [0.01, 0.1, 1.0]
    cv_scores = [cross_val_score(pipeline(c), X[train], y[train], cv=cv, scoring="roc_auc").mean()
                 for c in choices]
    chosen = choices[int(np.argmax(cv_scores))]
    model = pipeline(chosen)
    model.fit(X[train], y[train])
    prob = model.predict_proba(X[test])[:, 1]
    tn, fp, fn, tp = confusion_matrix(y[test], prob >= .5, labels=[0, 1]).ravel()
    report = {"target": "Androids患者组（含多种抑郁相关诊断）vs健康对照",
              "train_people": int(train.sum()), "test_people": int(test.sum()),
              "test_patient_group": int(y[test].sum()), "test_controls": int((1-y[test]).sum()),
              "auc": round(float(roc_auc_score(y[test], prob)), 3),
              "balanced_accuracy": round(float(balanced_accuracy_score(y[test], prob >= .5)), 3),
              "sensitivity": round(float(tp / (tp + fn)), 3),
              "specificity": round(float(tn / (tn + fp)), 3),
              "selected_C": chosen}
    with open(a.model_path, "wb") as f:
        pickle.dump({"model": model, "test_metrics": report}, f)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

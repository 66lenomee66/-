"""Research prototype for voice-based depression risk screening; not a diagnosis."""
import argparse
import csv
import json
import pickle
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from scipy.fft import dct
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def features(path):
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError("仅支持 16-bit PCM WAV")
        sr = f.getframerate()
        x = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(float)
        x = x.reshape(-1, f.getnchannels()).mean(axis=1) / 32768
    if len(x) / sr < 1:
        raise ValueError("录音至少需要 1 秒")
    if sr != 16000:
        from math import gcd
        g = gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g)
        sr = 16000
    x = x - x.mean()
    frame, hop = 400, 160
    frames = np.stack([x[i:i + frame] for i in range(0, len(x) - frame + 1, hop)])
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    threshold = max(0.008, np.percentile(rms, 80) * 0.2)
    voiced = rms > threshold
    # Autocorrelation pitch estimates on active frames; invalid estimates stay missing.
    pitches = []
    for f in frames[voiced][::5]:
        f = f * np.hanning(len(f))
        ac = np.correlate(f, f, mode="full")[len(f) - 1:]
        lo, hi = int(sr / 350), int(sr / 75)
        lag = lo + np.argmax(ac[lo:hi])
        if ac[lag] > 0.35 * ac[0]:
            pitches.append(sr / lag)
    p = np.asarray(pitches)
    runs = np.diff(np.r_[False, ~voiced, False].astype(int))
    pauses = (np.where(runs == -1)[0] - np.where(runs == 1)[0]) * hop / sr
    pauses = pauses[pauses >= 0.2]
    spec = np.abs(np.fft.rfft(frames[voiced] * np.hanning(frame), n=512)) ** 2
    if len(spec):
        hz = np.linspace(0, sr / 2, spec.shape[1])
        mel = lambda v: 2595 * np.log10(1 + v / 700)
        edges = 700 * (10 ** (np.linspace(mel(0), mel(sr / 2), 16) / 2595) - 1)
        bands = np.stack([np.maximum(0, np.minimum((hz-a)/max(b-a, 1), (c-hz)/max(c-b, 1)))
                          for a,b,c in zip(edges[:-2], edges[1:-1], edges[2:])], axis=1)
        mfcc = dct(np.log(np.maximum(spec @ bands, 1e-10)), type=2, norm="ortho", axis=1)[:, 1:7]
        cepstral = np.r_[mfcc.mean(axis=0), mfcc.std(axis=0)].tolist()
    else:
        cepstral = [np.nan] * 12
    return [len(x) / sr, float(voiced.mean()), float(np.median(rms)),
            float(np.std(rms)), float(np.median(p)) if len(p) else np.nan,
            float(np.std(p)) if len(p) else np.nan,
            len(pauses) / (len(x) / sr), float(np.mean(pauses)) if len(pauses) else 0.0,
            float(pauses.sum() / (len(x) / sr))] + cepstral


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("manifest", help="CSV: path,patient_id,label; label 0/1")
    train.add_argument("model")
    pred = sub.add_parser("predict")
    pred.add_argument("model")
    pred.add_argument("wav")
    args = ap.parse_args()
    if args.command == "train":
        base = Path(args.manifest).resolve().parent
        with open(args.manifest, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        usable = []
        for r in rows:
            try:
                usable.append((r, features(base / r["path"])))
            except ValueError as e:
                print(f"跳过 {r['path']}: {e}")
        by_person = {}
        for r, v in usable:
            by_person.setdefault(r["patient_id"], []).append((r, v))
        people = sorted(by_person)
        X, y, splits = [], [], []
        for person in people:
            items = by_person[person]
            labels = {int(r["label"]) for r, _ in items}
            if len(labels) != 1:
                raise ValueError(f"受试者 {person} 的标签不一致")
            X.append(np.nanmean(np.asarray([v for _, v in items]), axis=0))
            y.append(labels.pop())
            splits.append(items[0][0].get("official_split", ""))
        X, y, groups = np.asarray(X), np.asarray(y), np.asarray(people)
        if len(np.unique(groups)) < 10 or set(y) != {0, 1}:
            raise ValueError("至少需要 10 名不同受试者，且包含两类标签")
        if set(splits) == {"t", "v"}:
            tr = np.flatnonzero(np.asarray(splits) == "t")
            te = np.flatnonzero(np.asarray(splits) == "v")
        else:
            split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
            tr, te = next(split.split(X, y, groups))
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            raise ValueError("划分后训练集或测试集缺少某一类；请增加受试者")
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                              StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        report = {"train_patients": int(len(tr)), "test_patients": int(len(te)),
                  "balanced_accuracy": float(balanced_accuracy_score(y[te], prob >= 0.5)),
                  "roc_auc": float(roc_auc_score(y[te], prob))}
        with open(args.model, "wb") as f:
            pickle.dump({"model": model, "validation": report}, f)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        with open(args.model, "rb") as f:
            saved = pickle.load(f)  # Only load model files you trust.
        model = saved["model"] if isinstance(saved, dict) else saved
        validation = saved.get("validation") if isinstance(saved, dict) else None
        with wave.open(args.wav, "rb") as recording:
            duration = recording.getnframes() / recording.getframerate()
        if duration < 20:
            raise ValueError("正式演示录音建议至少 20 秒；当前录音过短")
        score = float(model.predict_proba([features(args.wav)])[0, 1])
        print(json.dumps({"screening_score_0_100": round(score * 100, 1),
                          "validation": validation,
                          "note": "研究用筛查分数；不是实际患病概率，不能用于诊断"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

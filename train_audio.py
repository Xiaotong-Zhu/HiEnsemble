import os
import json
import numpy as np
import librosa
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

AUDIO_DIR = 'data/audio'
TEXT_DIR = 'data/text'
FEATURE_DIR = 'data/audio_features'
INFO_PATH = 'data/info.json'

os.makedirs(FEATURE_DIR, exist_ok=True)

RESULTS_DIR = 'results/audio'
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_labels(info_path):
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    return {item['id']: item['label'] for item in info}

def load_timestamps(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_segment_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    features = [
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(chroma, axis=1), np.std(chroma, axis=1),
        np.mean(rms, axis=1), np.std(rms, axis=1),
        np.mean(zcr, axis=1), np.std(zcr, axis=1),
        np.mean(centroid, axis=1), np.std(centroid, axis=1),
        np.mean(bandwidth, axis=1), np.std(bandwidth, axis=1),
        np.mean(contrast, axis=1), np.std(contrast, axis=1),
        np.mean(rolloff, axis=1), np.std(rolloff, axis=1),
        np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1)
    ]

    return np.concatenate(features)


def process_sample(audio_path, timestamp_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"读取失败: {audio_path}，错误: {e}")
        return None

    timestamps = load_timestamps(timestamp_path)
    features = []
    for seg in timestamps:
        start = float(seg['start'])
        end = float(seg['end'])
        y_seg = y[int(start * sr): int(end * sr)]

        if len(y_seg) < 2048:
            print(f"片段太短（{len(y_seg)}），跳过")
            continue

        feat = extract_segment_features(y_seg, sr)
        features.append(feat)

    if not features:
        return None

    matrix = np.array(features)
    return np.concatenate([np.mean(matrix, axis=0),
                           np.std(matrix, axis=0),
                           np.max(matrix, axis=0),
                           np.min(matrix, axis=0)])

def main():
    id2label = load_labels(INFO_PATH)
    audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(('wav', 'mp3', 'MP3'))])
    print(len(audio_files))

    features_list = []
    labels = []
    id_set = []

    existing = [f for f in os.listdir(FEATURE_DIR) if f.endswith('.npy')]
    if existing:
        print("检测到已有音频特征文件，跳过特征提取阶段...")
        for fname in existing:
            sample_id = os.path.splitext(fname)[0]
            if sample_id not in id2label:
                continue
            feat = np.load(os.path.join(FEATURE_DIR, fname))
            features_list.append(feat)
            labels.append(id2label[sample_id])
            id_set.append(sample_id)
    else:
        print("正在提取音频特征...")
        for file in tqdm(audio_files, desc="处理样本"):
            sample_id = os.path.splitext(file)[0]
            if sample_id not in id2label:
                continue

            audio_path = os.path.join(AUDIO_DIR, file)
            timestamp_path = os.path.join(TEXT_DIR, f"{sample_id}.json")

            if not os.path.exists(timestamp_path):
                continue

            feature_vector = process_sample(audio_path, timestamp_path)
            if feature_vector is None:
                continue

            np.save(os.path.join(FEATURE_DIR, f"{sample_id}.npy"), feature_vector)
            features_list.append(feature_vector)
            labels.append(id2label[sample_id])


    X = np.array(features_list)
    y = np.array(labels)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=16)

    accs, f1s, precisions, recalls, aucs = [], [], [], [], []
    print(len(X))

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        selector = SelectKBest(f_classif, k=85)
        X_train_reduced = selector.fit_transform(X_train_scaled, y_train)
        X_test_reduced = selector.transform(X_test_scaled)

        print(f"\nFold {fold} - 特征维度从 {X.shape[1]} 降为 {X_train_reduced.shape[1]}")

        clf = RandomForestClassifier(n_estimators=30, random_state=16)
        clf.fit(X_train_reduced, y_train)
        y_pred = clf.predict(X_test_reduced)
        y_prob = clf.predict_proba(X_test_reduced)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        except:
            auc = None

        cm = confusion_matrix(y_test, y_pred)

        accs.append(acc)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        aucs.append(auc if auc is not None else np.nan)

        print(f"\nFold {fold} Metrics:")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC: {auc:.4f}" if auc is not None else "AUC: Not available for this fold.")
        print("Confusion Matrix:")
        print(cm)

        fold_file = os.path.join(RESULTS_DIR, f"fold_{fold}.txt")
        with open(fold_file, 'w', encoding='utf-8') as f_out:
            f_out.write(f"=== Fold {fold} ===\n")
            f_out.write("Train IDs:\n")
            f_out.write(", ".join([id_set[i] for i in train_idx]) + "\n")
            f_out.write("Test IDs:\n")
            f_out.write(", ".join([id_set[i] for i in test_idx]) + "\n\n")
            f_out.write(f"Accuracy: {acc:.4f}\n")
            f_out.write(f"F1 Score: {f1:.4f}\n")
            f_out.write(f"Precision: {precision:.4f}\n")
            f_out.write(f"Recall: {recall:.4f}\n")
            f_out.write(f"AUC: {auc:.4f}\n" if auc is not None else "AUC: Not available\n")
            f_out.write("Confusion Matrix:\n")
            f_out.write(str(cm) + "\n")

    print("\nAverage Metrics Across Folds:")
    print(f"Avg Accuracy:  {np.mean(accs):.4f}")
    print(f"Avg F1 Score:  {np.mean(f1s):.4f}")
    print(f"Avg Precision:{np.mean(precisions):.4f}")
    print(f"Avg Recall:   {np.mean(recalls):.4f}")
    if not np.isnan(aucs).all():
        print(f"Avg AUC:      {np.nanmean(aucs):.4f}")
    else:
        print("Avg AUC:      Not available")

    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Average Metrics Across Folds:\n")
        f.write(f"Avg Accuracy:  {np.mean(accs):.4f}\n")
        f.write(f"Avg F1 Score:  {np.mean(f1s):.4f}\n")
        f.write(f"Avg Precision:{np.mean(precisions):.4f}\n")
        f.write(f"Avg Recall:   {np.mean(recalls):.4f}\n")
        if not np.isnan(aucs).all():
            f.write(f"Avg AUC:      {np.nanmean(aucs):.4f}\n")
        else:
            f.write("Avg AUC:      Not available\n")

if __name__ == "__main__":
    main()

import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import hanlp
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.preprocessing import StandardScaler


HanLP = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)

text_dir = 'data/text'
feature_dir = 'data/text_features'
info_path = 'data/info.json'
w2v_path = 'weights/w2v.model'

RESULTS_DIR = 'results/text'
os.makedirs(RESULTS_DIR, exist_ok=True)

if not os.path.exists(feature_dir):
    os.makedirs(feature_dir)


def load_json_sentences(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item['text'] for item in data if item['text'].strip()]


def load_labels(info_path):
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    return {item['id']: item['label'] for item in info}


def train_or_load_w2v(sentences, model_path="w2v.model"):
    tokenized = [[w for w in HanLP(s)['tok'] if w.strip()] for s in sentences]
    if os.path.exists(model_path):
        model = Word2Vec.load(model_path)
    else:
        model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, sg=1, workers=4, epochs=20)
        model.save(model_path)
    return model


def syntactic_features(sentences):
    pos_counter = {
        'n': 0, 'v': 0, 'a': 0, 'd': 0, 'r': 0,
        'm': 0, 'q': 0, 'p': 0, 'c': 0, 'u': 0, 'e': 0, 'y': 0
    }
    total = 0
    for s in sentences:
        result = HanLP(s)
        pos_tags = result['pos']
        for tag in pos_tags:
            total += 1
            for key in pos_counter:
                if tag.startswith(key):
                    pos_counter[key] += 1
                    break
    if total == 0:
        return np.zeros(len(pos_counter))
    return np.array([pos_counter[k] / total for k in pos_counter])



def semantic_features(sentences, w2v_model, k_list=[3, 4, 5, 6, 7]):
    tokens = [w for s in sentences for w in HanLP(s)['tok'] if w in w2v_model.wv]
    
    if len(tokens) < max(k_list) + 1:
        return np.zeros(len(k_list) * 5)

    features = []

    for k in k_list:
        sims = []
        for i in range(len(tokens) - k):
            vec1 = w2v_model.wv[tokens[i]]
            vec2 = w2v_model.wv[tokens[i + k]]
            sim = cosine_similarity([vec1], [vec2])[0][0]
            sims.append(sim)
        
        if sims:
            sims = np.array(sims)
            features.extend([
                np.min(sims),
                np.max(sims),
                np.mean(sims),
                np.std(sims),
                np.percentile(sims, 90),
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
    
    return np.array(features)


def main():
    json_paths = sorted([os.path.join(text_dir, f) for f in os.listdir(text_dir) if f.endswith('.json')])
    print(len(json_paths))
    id2label = load_labels(info_path)
    id_set = []

    features_list = []
    labels = []

    pre_extracted = [f for f in os.listdir(feature_dir) if f.endswith(".npy")]
    if pre_extracted:
        print("检测到已有特征文件，跳过特征提取阶段。")
        for fname in pre_extracted:
            sample_id = os.path.splitext(fname)[0]
            feature_path = os.path.join(feature_dir, fname)
            if sample_id in id2label:
                features_list.append(np.load(feature_path))
                labels.append(id2label[sample_id])
                id_set.append(sample_id)
    else:
        print("未检测到特征文件，开始进行特征提取...")

        all_sentences = []
        for path in json_paths:
            all_sentences.extend(load_json_sentences(path))
        w2v_model = train_or_load_w2v(all_sentences, w2v_path)

        for path in tqdm(json_paths, desc="提取特征"):
            sample_id = os.path.splitext(os.path.basename(path))[0]
            if sample_id not in id2label:
                continue

            feature_file = os.path.join(feature_dir, f"{sample_id}.npy")
            sentences = load_json_sentences(path)
            syntactic = syntactic_features(sentences)
            semantic = semantic_features(sentences, w2v_model)
            features = np.concatenate([syntactic, semantic])
            np.save(feature_file, features)

            features_list.append(features)
            labels.append(id2label[sample_id])
            id_set.append(sample_id)

    X = np.array(features_list)[:, 12:]
    y = np.array(labels)

    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=16)

    accs, f1s, precisions, recalls, aucs = [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # selector = SelectKBest(f_classif, k=8)
        selector = PCA(n_components=9) 
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        print(f"\nFold {fold} - 特征维度从 {X.shape[1]} 降为 {X_train.shape[1]}")

        clf = RandomForestClassifier(n_estimators=34, random_state=16)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import shap
import joblib

import seaborn as sns

plt.rcParams["font.family"] = "Times New Roman"

AUDIO_FEATURE_DIR = 'data/audio_features'
TEXT_FEATURE_DIR = 'data/text_features'
RESULTS_DIR = 'results/fusion_late'
AUDIO_FEATURE_NAME_PATH = 'data/audio_feature_name.txt'
TEXT_FEATURE_NAME_PATH = 'data/text_feature_name.txt'
INFO_PATH = 'data/info.json'

SHAP_SAMPLE_LIMIT = 9
SHAP_NSAMPLES = 100

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_features(feature_dir, id_list):
    features = []
    for sample_id in id_list:
        fpath = os.path.join(feature_dir, f"{sample_id}.npy")
        try:
            feat = np.load(fpath)
            if np.isnan(feat).all():
                continue
            features.append(feat)
        except Exception as e:
            print(f"加载失败: {sample_id}, 错误: {e}")
    return np.array(features)

def load_labels(info_path):
    import json
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    return {item['id']: item['label'] for item in info}

def preprocess(X_audio, X_text, y, audio_feature_names, text_feature_names, k_audio=85, k_text_pca=9):
    scaler_audio = StandardScaler()
    scaler_text = StandardScaler()
    X_audio_scaled = scaler_audio.fit_transform(X_audio)
    X_text_scaled = scaler_text.fit_transform(X_text)

    selector_audio = SelectKBest(f_classif, k=k_audio)
    X_audio_selected = selector_audio.fit_transform(X_audio_scaled, y)
    selected_audio_names = [audio_feature_names[i] for i, m in enumerate(selector_audio.get_support()) if m]

    pca_text = PCA(n_components=k_text_pca)
    X_text_pca = pca_text.fit_transform(X_text_scaled)
    pca_components = pca_text.components_

    return (
        X_audio_selected, X_text_pca,
        scaler_audio, scaler_text,
        selector_audio, pca_text,
        selected_audio_names, [f'text_pca_{i}' for i in range(k_text_pca)],
        pca_components, text_feature_names,
        X_text_scaled
    )

def compute_text_raw_importance(pca_components, pca_feature_importances, text_feature_names):
    raw_importances = np.abs(pca_components).T @ pca_feature_importances
    return dict(zip(text_feature_names, raw_importances))

def get_merge_feature_names(audio_feature_names, text_feature_names):
    return audio_feature_names + text_feature_names

def get_selectkbest_selected_feature_names(selector_kbest, all_feature_names):
    selected_mask = selector_kbest.get_support()
    return [all_feature_names[i] for i, m in enumerate(selected_mask) if m]

def save_confusion_matrix(y_true, y_pred, labels, filename, fold):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.to_csv(filename)

    fig = plt.figure(figsize=(1.7, 1.7))

    gs = fig.add_gridspec(len(labels), len(labels))

    ax = fig.add_subplot(gs[:, :])
    cax = ax.imshow(cm, cmap='Blues', aspect='equal')

    ax.set_title(f'Fold {fold}', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='black', fontsize=12)

    plt.subplots_adjust(top=0.85, bottom=0.3, left=0.15, right=0.95)

    plt.savefig(filename.replace('.csv', '.png'))
    plt.close()

from matplotlib.colors import LinearSegmentedColormap

def plot_feature_importance(importance_dict, title, filename, topk=10):
    sorted_items = sorted(importance_dict.items(), key=lambda x: -x[1])
    feature_names = [k for k, v in sorted_items[:topk]]
    importances = [v for k, v in sorted_items[:topk]]

    pastel_colors = ["#3D5A80", "#98C1D9", "#E0FBFC"]
    cmap = LinearSegmentedColormap.from_list("pastel_gradient", pastel_colors, N=len(feature_names))
    gradient_colors = [cmap(i / len(feature_names)) for i in range(len(feature_names))]

    plt.figure(figsize=(max(6, 0.4 * len(feature_names)), 4))
    sns.set_theme(style="whitegrid")

    sns.barplot(x=feature_names, y=importances, palette=gradient_colors)

    plt.title(title, fontsize=14, fontname="Times New Roman")
    plt.xlabel('Features', fontsize=12, fontname="Times New Roman")
    plt.ylabel('Importance', fontsize=12, fontname="Times New Roman")
    plt.xticks(rotation=45, ha='right', fontsize=10, fontname="Times New Roman")
    plt.yticks(fontsize=10, fontname="Times New Roman")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    plt.close()


from scipy import stats
import matplotlib.font_manager as fm
def plot_fold_metrics(fold_metrics):
    metrics_df = pd.DataFrame(fold_metrics)
    
    plt.figure(figsize=(5, 4))

    for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
        plt.plot(metrics_df['fold'], metrics_df[metric], label=metric, marker='o', linestyle='-', linewidth=2)


    plt.title('Fold-wise Metric', fontsize=14, fontname="Times New Roman")
    plt.xlabel('Fold', fontsize=13, fontname="Times New Roman")
    plt.ylabel('Metric Value', fontsize=13, fontname="Times New Roman")
    plt.legend(title='Metrics', loc='best', fontsize=13, prop=fm.FontProperties(family='Times New Roman'))


    plt.ylim(0.0, 1.0)


    plt.grid(False)


    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)


    plt.tight_layout()

    for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
        values = metrics_df[metric].values
        for i in range(len(values) - 1):

            t_stat, p_value = stats.ttest_ind([values[i]], [values[i + 1]])
            if p_value < 0.05:  
                plt.text(i + 0.5, max(values) - 0.03, "*", fontsize=14, color='red', ha='center', fontname="Times New Roman")

    plt.savefig(os.path.join(RESULTS_DIR, 'fold_metrics_fluctuations_with_significance.png'), dpi=300)
    plt.close()

def main():
    audio_feature_names = np.loadtxt(AUDIO_FEATURE_NAME_PATH, dtype=str).tolist()
    text_feature_names = np.loadtxt(TEXT_FEATURE_NAME_PATH, dtype=str).tolist()

    id2label = load_labels(INFO_PATH)
    common_ids = sorted([f.split('.')[0] for f in os.listdir(AUDIO_FEATURE_DIR) if f.endswith('.npy')])
    X_audio = load_features(AUDIO_FEATURE_DIR, common_ids)
    X_text = load_features(TEXT_FEATURE_DIR, common_ids)[:, 12:]
    y = np.array([id2label[i] for i in common_ids])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=16)

    global_audio_imp, global_text_imp, global_merge_imp, global_fuser_imp = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    fold_metrics = []
    confusion_matrices = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_audio, y), 1):
        print(f"===== Fold {fold} =====")

        X_audio_train, X_audio_test = X_audio[train_idx], X_audio[test_idx]
        X_text_train, X_text_test = X_text[train_idx], X_text[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_audio_proc, X_text_pca_proc, scaler_audio, scaler_text, selector_audio, pca_text, selected_audio_names, text_pca_names, pca_components, text_feature_names, X_text_scaled_train = preprocess(
            X_audio_train, X_text_train, y_train, audio_feature_names, text_feature_names
        )

        clf_audio = RandomForestClassifier(n_estimators=30, random_state=16)
        clf_audio.fit(X_audio_proc, y_train)
        audio_importances = dict(zip(selected_audio_names, clf_audio.feature_importances_))

        clf_text = RandomForestClassifier(n_estimators=34, random_state=16)
        clf_text.fit(X_text_pca_proc, y_train)
        text_raw_importances = compute_text_raw_importance(pca_components, clf_text.feature_importances_, text_feature_names)

        X_train_merge = np.concatenate([X_audio_proc, X_text_pca_proc], axis=1)
        X_test_merge = np.concatenate([ 
            selector_audio.transform(scaler_audio.transform(X_audio_test)),
            pca_text.transform(scaler_text.transform(X_text_test))
        ], axis=1)

        merge_feature_names = selected_audio_names + text_pca_names

        selector_merge_pca = PCA(n_components=8)
        X_train_pca = selector_merge_pca.fit_transform(X_train_merge)
        X_test_pca = selector_merge_pca.transform(X_test_merge)
        clf_merge_pca = RandomForestClassifier(n_estimators=12, random_state=16)
        clf_merge_pca.fit(X_train_pca, y_train)

        selector_merge_kbest = SelectKBest(f_classif, k=min(86, X_train_merge.shape[1]))
        X_train_kbest = selector_merge_kbest.fit_transform(X_train_merge, y_train)
        X_test_kbest = selector_merge_kbest.transform(X_test_merge)
        selected_merge_feature_names = get_selectkbest_selected_feature_names(selector_merge_kbest, merge_feature_names)
        clf_merge_kbest = RandomForestClassifier(n_estimators=15, random_state=16)
        clf_merge_kbest.fit(X_train_kbest, y_train)

        train_stack = np.hstack([ 
            clf_text.predict_proba(X_text_pca_proc),
            clf_merge_pca.predict_proba(X_train_pca),
            clf_merge_kbest.predict_proba(X_train_kbest)
        ])
        test_stack = np.hstack([ 
            clf_text.predict_proba(pca_text.transform(scaler_text.transform(X_text_test))),
            clf_merge_pca.predict_proba(X_test_pca),
            clf_merge_kbest.predict_proba(X_test_kbest)
        ])
        fuser = RandomForestClassifier(n_estimators=57, random_state=16)
        fuser.fit(train_stack, y_train)
        y_pred = fuser.predict(test_stack)


        fold_model_dir = os.path.join(RESULTS_DIR, f"fold_{fold}_models")
        os.makedirs(fold_model_dir, exist_ok=True)

        joblib.dump(clf_audio, os.path.join(fold_model_dir, "clf_audio.pkl"))
        joblib.dump(clf_text, os.path.join(fold_model_dir, "clf_text.pkl"))
        joblib.dump(clf_merge_pca, os.path.join(fold_model_dir, "clf_merge_pca.pkl"))
        joblib.dump(clf_merge_kbest, os.path.join(fold_model_dir, "clf_merge_kbest.pkl"))
        joblib.dump(fuser, os.path.join(fold_model_dir, "fuser.pkl"))


        joblib.dump(selector_audio, os.path.join(fold_model_dir, "selector_audio.pkl"))
        joblib.dump(pca_text, os.path.join(fold_model_dir, "pca_text.pkl"))
        joblib.dump(scaler_audio, os.path.join(fold_model_dir, "scaler_audio.pkl"))
        joblib.dump(scaler_text, os.path.join(fold_model_dir, "scaler_text.pkl"))
        joblib.dump(selector_merge_kbest, os.path.join(fold_model_dir, "selector_merge_kbest.pkl"))
        joblib.dump(selector_merge_pca, os.path.join(fold_model_dir, "selector_merge_pca.pkl"))

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        try:
            roc = roc_auc_score(y_test, fuser.predict_proba(test_stack), multi_class='ovr', average='macro')
        except:
            roc = 0.0

        fold_metrics.append({
            'fold': fold,
            'accuracy': acc,
            'f1': f1,
            'precision': prec,
            'recall': rec,
            'roc_auc': roc
        })

        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        confusion_matrices.append(cm)

        save_confusion_matrix(y_test, y_pred, labels=np.unique(y), filename=os.path.join(RESULTS_DIR, f"confusion_matrix_fold{fold}.csv"), fold=fold)

        for k, v in audio_importances.items():
            global_audio_imp[k] += v
        for k, v in text_raw_importances.items():
            global_text_imp[k] += v
        for i, name in enumerate(selected_merge_feature_names):
            global_merge_imp[name] += clf_merge_kbest.feature_importances_[i]

        fuser_feature_names = [f"text_prob_{i}" for i in range(train_stack.shape[1]//3)] + \
                               [f"pca_prob_{i}" for i in range(train_stack.shape[1]//3)] + \
                               [f"kbest_prob_{i}" for i in range(train_stack.shape[1]//3)]
        for name, imp in zip(fuser_feature_names, fuser.feature_importances_):
            global_fuser_imp[name] += imp


    pd.DataFrame(fold_metrics).to_csv(os.path.join(RESULTS_DIR, "fold_metrics.csv"), index=False)

    metrics_df = pd.DataFrame(fold_metrics)
    print("\nAverage Metrics Across Folds:")
    print(f"Avg Accuracy:  {metrics_df['accuracy'].mean():.10f}")
    print(f"Avg F1 Score:  {metrics_df['f1'].mean():.10f}")
    print(f"Avg Precision:{metrics_df['precision'].mean():.10f}")
    print(f"Avg Recall:   {metrics_df['recall'].mean():.10f}")
    print(f"Avg AUC:      {metrics_df['roc_auc'].mean():.10f}")


    def save_importance(importance_dict, filename):
        sorted_items = sorted(importance_dict.items(), key=lambda x: -x[1])
        with open(os.path.join(RESULTS_DIR, filename), 'w', encoding='utf-8') as f:
            for k, v in sorted_items:
                f.write(f"{k}\t{v:.6f}\n")

    save_importance(global_audio_imp, "global_audio_importance.txt")
    save_importance(global_text_imp, "global_text_importance.txt")
    save_importance(global_merge_imp, "global_merge_importance.txt")
    save_importance(global_fuser_imp, "global_fuser_importance.txt")


    plot_feature_importance(global_audio_imp, "Top Audio Feature Importances", "audio_feature_importance.png")
    plot_feature_importance(global_text_imp, "Top Text Feature Importances", "text_feature_importance.png")
    plot_feature_importance(global_merge_imp, "Top 10 Multimodal Feature Importances", "merge_feature_importance.png")
    plot_feature_importance(global_fuser_imp, "Top Fuser Feature Importances", "fuser_feature_importance.png")

    plot_fold_metrics(fold_metrics)

    print("\n特征重要性图和指标波动图已保存！")

if __name__ == "__main__":
    main()

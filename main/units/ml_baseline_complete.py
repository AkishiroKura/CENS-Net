# -*- coding: utf-8 -*-
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, f1_score, 
                             roc_auc_score, confusion_matrix)
from dataloader import T2RDataset
import random

# 固定随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Loading dataset...")
DATA_ROOT = r"F:\\workspace1\\dataset"
dataset = T2RDataset(DATA_ROOT)

# 清理数据
clean = []
for i, d in enumerate(dataset):
    if not hasattr(d, "x") or d.x is None or d.x.size(0) == 0:
        continue
    if torch.isnan(d.x).any() or (hasattr(d, "edge_attr") and d.edge_attr is not None and torch.isnan(d.edge_attr).any()):
        continue
    if not hasattr(d, "y") or d.y is None:
        continue
    if not torch.is_tensor(d.y):
        d.y = torch.tensor(d.y)
    d.y = d.y.long()
    if d.y.min() < 0 or d.y.max() >= 2:
        continue
    if not hasattr(d, "global_feat") or d.global_feat is None:
        continue
    clean.append(d)

dataset = clean
print(f"Dataset cleaned: {len(dataset)} samples")

# 提取全局特征
X = np.array([d.global_feat.cpu().numpy().flatten() for d in dataset])
y = np.array([d.y.item() for d in dataset])

print(f"Feature shape: {X.shape}")
print(f"Label distribution: Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")

# 定义模型
models = {
    'SVM': SVC(kernel='rbf', C=1.0, random_state=SEED),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=SEED),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=SEED)
}

# 5折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

results = {}

for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    fold_metrics = {
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'f1': [],
        'auc': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
        
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # 计算指标
        acc = accuracy_score(y_test, y_pred)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = float('nan')
        
        fold_metrics['accuracy'].append(acc)
        fold_metrics['sensitivity'].append(sensitivity)
        fold_metrics['specificity'].append(specificity)
        fold_metrics['precision'].append(prec)
        fold_metrics['f1'].append(f1)
        fold_metrics['auc'].append(auc)
        
        print(f"Fold {fold}: acc={acc:.4f}, sen={sensitivity:.4f}, spe={specificity:.4f}, "
              f"prec={prec:.4f}, f1={f1:.4f}, auc={auc:.4f}")
    
    # 计算平均值和标准差
    results[model_name] = {}
    for metric, values in fold_metrics.items():
        values_clean = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
        mean_val = np.mean(values_clean)
        std_val = np.std(values_clean)
        results[model_name][metric] = (mean_val, std_val)
        print(f"{metric.capitalize():12s}: {mean_val:.4f} ± {std_val:.4f}")

# 写入汇总文件
print(f"\n{'='*60}")
print("Writing summary to ml_baseline_complete_summary.txt...")
with open('ml_baseline_complete_summary.txt', 'w', encoding='utf-8') as f:
    for model_name, metrics in results.items():
        f.write(f"\n{model_name} Results\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy  : {metrics['accuracy'][0]:.4f} ± {metrics['accuracy'][1]:.4f}\n")
        f.write(f"Sensitivity: {metrics['sensitivity'][0]:.4f} ± {metrics['sensitivity'][1]:.4f}\n")
        f.write(f"Specificity: {metrics['specificity'][0]:.4f} ± {metrics['specificity'][1]:.4f}\n")
        f.write(f"Precision : {metrics['precision'][0]:.4f} ± {metrics['precision'][1]:.4f}\n")
        f.write(f"F1        : {metrics['f1'][0]:.4f} ± {metrics['f1'][1]:.4f}\n")
        f.write(f"AUC       : {metrics['auc'][0]:.4f} ± {metrics['auc'][1]:.4f}\n")

print("Done!")

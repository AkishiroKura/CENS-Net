import torch
import torch.nn as nn
from dataloader import T2RDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import random

from models import GINEBaseline, DualPathModel
from train import train_model

# ===== 完全固定随机性，确保结果100%可重现 =====
SEED = 42

# 1. Python内置random
random.seed(SEED)

# 2. NumPy
np.random.seed(SEED)

# 3. PyTorch CPU
torch.manual_seed(SEED)

# 4. PyTorch GPU (如果使用CUDA)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # 多GPU情况
    
    # 5. cuDNN确定性设置（可能轻微降低性能，但保证可重现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Random seed set to {SEED} for reproducibility")
print("="*60)

DATA_ROOT = r"F:\\workspace1\\dataset"

dataset = T2RDataset(DATA_ROOT)

# 简单清理：移除空图、含 NaN 或标签不合法的样本；确保 y 为 long
clean = []
for i, d in enumerate(dataset):
    # 检查节点
    if not hasattr(d, "x") or d.x is None or d.x.size(0) == 0:
        print(f"Skipping sample {i}: empty node feature")
        continue
    # 检查 NaN
    if torch.isnan(d.x).any() or (hasattr(d, "edge_attr") and d.edge_attr is not None and torch.isnan(d.edge_attr).any()):
        print(f"Skipping sample {i}: NaN in features/edge_attr")
        continue
    # 标签检查
    if not hasattr(d, "y") or d.y is None:
        print(f"Skipping sample {i}: missing label")
        continue
    if not torch.is_tensor(d.y):
        d.y = torch.tensor(d.y)
    d.y = d.y.long()
    # 检查标签范围（可选）
    if d.y.min() < 0 or d.y.max() >= 2:  # num_classes=2
        print(f"Skipping sample {i}: label out of range ({d.y})")
        continue
    # 检查全局特征
    if not hasattr(d, "global_feat") or d.global_feat is None:
        print(f"Skipping sample {i}: missing global_feat")
        continue
    clean.append(d)
if len(clean) == 0:
    raise RuntimeError("No valid samples after cleaning dataset.")
dataset = clean

print(f"Dataset size: {len(dataset)}")
print(f"Sample: {dataset[0]}")

from torch_geometric.loader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== 5折交叉验证 =====
n = len(dataset)
if n < 5:
    raise RuntimeError("需要至少 5 个样本以进行 5 折交叉验证")

# 提取标签用于分层分割
labels = np.array([d.y.item() for d in dataset])

# 5折分层交叉验证（使用固定的SEED确保数据划分一致）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print(f"Total samples: {n}")
print("=" * 60)
print("Starting 5-Fold Cross-Validation")
print("=" * 60)

for fold, (train_indices, val_indices) in enumerate(skf.split(range(n), labels)):
    print(f"\n{'='*60}")
    print(f"Fold {fold + 1}/5")
    print(f"{'='*60}")
    
    # 构建当前折的数据集
    train_set = [dataset[i] for i in train_indices]
    val_set = [dataset[i] for i in val_indices]
    
    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False) if len(val_set) > 0 else None
    
    # 打印当前折的数据分布
    train_labels_fold = labels[train_indices]
    val_labels_fold = labels[val_indices]
    print(f"Train samples: {len(train_set)}")
    print(f"  - Class 0: {(train_labels_fold == 0).sum()}, Class 1: {(train_labels_fold == 1).sum()}")
    print(f"Val samples: {len(val_set)}")
    print(f"  - Class 0: {(val_labels_fold == 0).sum()}, Class 1: {(val_labels_fold == 1).sum()}")
    
    # 每折使用不同的随机种子，确保模型初始化独立但可重现
    fold_seed = SEED + fold * 1000
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fold_seed)
        torch.cuda.manual_seed_all(fold_seed)
    np.random.seed(fold_seed)
    random.seed(fold_seed)
    print(f"Using fold seed: {fold_seed}")
    
    # 计算类别权重（处理数据不平衡）
    class_counts = np.bincount(train_labels_fold)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * 2  # 归一化
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion_fold = nn.CrossEntropyLoss(weight=class_weights_tensor)
    print(f"Class weights: {class_weights}")
    
    # 创建新模型 (DualPathModel - GNN + 特征图编码器)
    model_fold = DualPathModel(
        gnn_hidden=32,
        global_hidden=32,
        global_out=16,
        num_classes=2,
        dropout=0.5
    ).to(device)
    optimizer_fold = torch.optim.Adam(model_fold.parameters(), lr=5e-4, weight_decay=1e-4)
    
    # 学习率调度器：防止卡在局部最优
    scheduler_fold = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_fold, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # 训练当前折
    train_model(model_fold, train_loader, val_loader, device, criterion_fold, optimizer_fold, 
                save_path=f"best_model_fold{fold+1}.pt",
                log_path=f"training_log_fold{fold+1}.csv",
                results_path=f"best_results_fold{fold+1}.txt",
                fold_num=fold+1,
                preds_save_path=f"best_preds_fold{fold+1}.npz",
                save_dir=".")

print(f"\n{'='*60}")
print("5-Fold Cross-Validation Complete!")
print(f"{'='*60}")

# ---------- Cross-validation summary & comparison plots ----------
from plot import plot_loss_comparison, plot_roc_comparison, plot_confusion_matrix_average
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

fold_metrics = {
    'accuracy': [],
    'f1': [],
    'precision': [],
    'recall': [],
    'auc': []
}
log_paths = []
pred_npz_paths = []
for fold in range(1, 6):
    log_paths.append(f"training_log_fold{fold}.csv")
    pred_npz_paths.append(f"best_preds_fold{fold}.npz")

# 计算每折指标
for i, p in enumerate(pred_npz_paths, start=1):
    try:
        data = np.load(p, allow_pickle=True)
        preds = data['preds']
        labels_fold = data['labels']
        probs = data['probs'] if 'probs' in data.files else None
        acc = accuracy_score(labels_fold, preds)
        f1v = f1_score(labels_fold, preds, average='weighted')
        prec = precision_score(labels_fold, preds, average='weighted', zero_division=0)
        rec = recall_score(labels_fold, preds, average='weighted')
        if probs is not None and len(probs) > 0:
            try:
                aucv = roc_auc_score(labels_fold, probs)
            except Exception:
                aucv = float('nan')
        else:
            aucv = float('nan')
        fold_metrics['accuracy'].append(acc)
        fold_metrics['f1'].append(f1v)
        fold_metrics['precision'].append(prec)
        fold_metrics['recall'].append(rec)
        fold_metrics['auc'].append(aucv)
        print(f"Fold {i}: acc={acc:.4f}, f1={f1v:.4f}, prec={prec:.4f}, rec={rec:.4f}, auc={aucv:.4f}")
    except Exception as e:
        print(f"Warning: could not load predictions for fold {i}: {e}")

# 写入汇总文件
with open('cross_validation_summary.txt', 'w', encoding='utf-8') as f:
    f.write('Cross-Validation Summary (5 folds)\n')
    f.write('='*50 + '\n')
    for k, vals in fold_metrics.items():
        vals_clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if len(vals_clean) == 0:
            mean_v = float('nan')
            std_v = float('nan')
        else:
            mean_v = np.mean(vals_clean)
            std_v = np.std(vals_clean)
        f.write(f"{k.capitalize():10s}: {mean_v:.4f} ± {std_v:.4f}\n")

print("Saved cross_validation_summary.txt")

# ===== 生成全数据集的t-SNE可视化 =====
print("\nGenerating full dataset t-SNE...")
try:
    # 找到F1分数最高的fold作为最佳模型
    best_fold = None
    best_f1 = -1
    for k, vals in fold_metrics.items():
        if k == 'f1':
            vals_clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if len(vals_clean) > 0:
                fold_f1s = vals_clean
                max_f1_idx = np.argmax(fold_f1s)
                best_f1 = fold_f1s[max_f1_idx]
                best_fold = max_f1_idx + 1
                break
    
    if best_fold is not None:
        print(f"Using best model from fold {best_fold} (F1={best_f1:.4f}) for full dataset t-SNE")
        
        # 加载最佳模型
        best_model_path = f"best_model_fold{best_fold}.pt"
        model_full = DualPathModel(
            gnn_hidden=32,
            global_hidden=32,
            global_out=16,
            num_classes=2,
            dropout=0.5
        ).to(device)
        model_full.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
        
        # 对整个数据集提取特征
        full_loader = DataLoader(dataset, batch_size=16, shuffle=False)  # 更大batch加速
        from evaluate import extract_features
        features, labels_full = extract_features(model_full, full_loader, device)
        
        if len(features) > 0:
            # 生成全数据集t-SNE
            from plot import plot_tsne
            n_samples = len(features)
            perplexity = min(30, max(5, n_samples // 3))
            print(f"Generating t-SNE for {n_samples} samples (perplexity={perplexity})")
            plot_tsne(features, labels_full, save_dir=".", perplexity=perplexity, fold_num=None, suffix="full_dataset")
            print("Saved full dataset t-SNE: tsne_full_dataset.png")
        else:
            print("Warning: No features extracted for full dataset t-SNE")
    else:
        print("Warning: Could not determine best fold for full dataset t-SNE")
        
except Exception as e:
    print(f"Warning: Failed to generate full dataset t-SNE: {e}")

# 生成汇总图片
try:
    plot_loss_comparison(log_paths, save_path='loss_curves_comparison.png')
    plot_roc_comparison(pred_npz_paths, save_path='roc_curves_comparison.png')
    plot_confusion_matrix_average(pred_npz_paths, save_path='confusion_matrix_average.png')
    print("Saved comparison figures: loss_curves_comparison.png, roc_curves_comparison.png, confusion_matrix_average.png")
except Exception as e:
    print(f"Warning: failed to generate plots: {e}")
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import T2RDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random

from models import GINEBaseline, DualPathModel, GCNBaseline, GINBaseline, GATBaseline, BrainNetCNN, ContrastPoolNet
from train import train_model

# ===== 完全固定随机性，确保结果100%可重现 =====
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Random seed set to {SEED} for reproducibility")
print("="*60)

# ==========================================
# COMPARISON STUDY CONFIGURATION
# ==========================================
# Options:
# "gcn" : Graph Convolutional Network (Baseline)
# "gin" : Graph Isomorphism Network (Baseline)
# "gat" : Graph Attention Network (Baseline)
# "brainnetcnn" : BrainNetCNN (Kawahara et al., 2017)
# "contrastpool" : ContrastPool (Xu et al., IEEE TMI 2024)
COMPARISON_MODE = "contrastpool"  # Change this to run different experiments
# ==========================================

print(f"Running Comparison Study: Mode = {COMPARISON_MODE}")

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

# 如果是 BrainNetCNN 模式，需要转换数据格式
if COMPARISON_MODE == "brainnetcnn":
    from brainnet_utils import convert_dataset_to_adjacency, BrainNetDataLoader
    print("Converting dataset to adjacency matrices for BrainNetCNN...")
    # 找出最大节点数
    max_nodes = max([d.x.size(0) for d in dataset])
    print(f"Max nodes in dataset: {max_nodes}")
    dataset_adj = convert_dataset_to_adjacency(dataset, max_nodes=max_nodes)
    print(f"Converted {len(dataset_adj)} samples to adjacency format")

# ===== 5折交叉验证 =====
n = len(dataset)
if n < 5:
    raise RuntimeError("需要至少5个样本以进行5折交叉验证")

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
    if COMPARISON_MODE == "brainnetcnn":
        train_set = [dataset_adj[i] for i in train_indices]
        val_set = [dataset_adj[i] for i in val_indices]
        # 使用自定义 DataLoader
        train_loader = BrainNetDataLoader(train_set, batch_size=8, shuffle=True)
        val_loader = BrainNetDataLoader(val_set, batch_size=8, shuffle=False) if len(val_set) > 0 else None
    else:
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
    
    # ==========================================
    # MODEL SELECTION BASED ON COMPARISON MODE
    # ==========================================
    if COMPARISON_MODE == "gcn":
        model_fold = GCNBaseline(
            in_dim=14,
            hidden_dim=32,
            num_classes=2,
            dropout=0.5
        ).to(device)
    elif COMPARISON_MODE == "gin":
        model_fold = GINBaseline(
            in_dim=14,
            hidden_dim=32,
            num_classes=2,
            dropout=0.5
        ).to(device)
    elif COMPARISON_MODE == "gat":
        model_fold = GATBaseline(
            in_dim=14,
            hidden_dim=32,
            num_classes=2,
            dropout=0.5,
            heads=4
        ).to(device)
    elif COMPARISON_MODE == "brainnetcnn":
        # BrainNetCNN使用简化的单通道邻接矩阵（只有边特征第一维）
        num_edge_features = 1
        model_fold = BrainNetCNN(
            num_nodes=max_nodes,
            num_edge_features=num_edge_features,
            num_classes=2,
            dropout=0.5
        ).to(device)
    elif COMPARISON_MODE == "contrastpool":
        # ContrastPool: Xu et al., IEEE TMI 2024
        # 找出最大节点数
        max_nodes_cp = max([d.x.size(0) for d in dataset])
        model_fold = ContrastPoolNet(
            in_dim=14,
            hidden_dim=32,
            num_classes=2,
            max_num_nodes=max_nodes_cp,
            pool_ratio=0.5,
            n_layers=2,
            dropout=0.3,
            lambda1=0.01
        ).to(device)
        
        # 初始化对比学习（使用训练集计算类别对比特征）
        train_labels_np = np.array([dataset[i].y.item() for i in train_indices])
        train_dataset = [dataset[i] for i in train_indices]
        model_fold.cal_contrast(train_dataset, train_labels_np, device, num_classes=2)
        print("ContrastPool: initialized contrastive features from training set")
    else:
        raise ValueError(f"Unknown comparison mode: {COMPARISON_MODE}")
    
    print(f"Model initialized: {COMPARISON_MODE}")
    
    optimizer_fold = torch.optim.Adam(model_fold.parameters(), lr=5e-4, weight_decay=1e-4)
    
    # 学习率调度器：防止卡在局部最优
    scheduler_fold = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_fold, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Define output filenames with comparison prefix
    prefix = f"comparison_{COMPARISON_MODE}_"
    
    # 训练当前折
    if COMPARISON_MODE == "brainnetcnn":
        from train_brainnet import train_brainnetcnn
        train_brainnetcnn(model_fold, train_loader, val_loader, device, criterion_fold, optimizer_fold, 
                    save_path=f"{prefix}best_model_fold{fold+1}.pt",
                    log_path=f"{prefix}training_log_fold{fold+1}.csv",
                    results_path=f"{prefix}best_results_fold{fold+1}.txt",
                    fold_num=fold+1,
                    preds_save_path=f"{prefix}best_preds_fold{fold+1}.npz")
    elif COMPARISON_MODE == "contrastpool":
        from train_contrastpool import train_contrastpool
        train_contrastpool(model_fold, train_loader, val_loader, device, criterion_fold, optimizer_fold, 
                    save_path=f"{prefix}best_model_fold{fold+1}.pt",
                    log_path=f"{prefix}training_log_fold{fold+1}.csv",
                    results_path=f"{prefix}best_results_fold{fold+1}.txt",
                    fold_num=fold+1,
                    preds_save_path=f"{prefix}best_preds_fold{fold+1}.npz")
    else:
        train_model(model_fold, train_loader, val_loader, device, criterion_fold, optimizer_fold, 
                    save_path=f"{prefix}best_model_fold{fold+1}.pt",
                    log_path=f"{prefix}training_log_fold{fold+1}.csv",
                    results_path=f"{prefix}best_results_fold{fold+1}.txt",
                    fold_num=fold+1,
                    preds_save_path=f"{prefix}best_preds_fold{fold+1}.npz")

print(f"\n{'='*60}")
print("5-Fold Cross-Validation Complete!")
print(f"{'='*60}")

# ---------- Cross-validation summary & comparison plots ----------
from plot import plot_loss_comparison, plot_roc_comparison, plot_confusion_matrix_average
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np

fold_metrics = {
    'accuracy': [],
    'sensitivity': [],
    'specificity': [],
    'f1': [],
    'precision': [],
    'auc': []
}
log_paths = []
pred_npz_paths = []
prefix = f"comparison_{COMPARISON_MODE}_"

for fold in range(1, 6):
    log_paths.append(f"{prefix}training_log_fold{fold}.csv")
    pred_npz_paths.append(f"{prefix}best_preds_fold{fold}.npz")

# 计算每折指标
for i, p in enumerate(pred_npz_paths, start=1):
    try:
        data = np.load(p, allow_pickle=True)
        preds = data['preds']
        labels = data['labels']
        probs = data['probs'] if 'probs' in data.files else None
        
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        
        # 计算基本指标
        acc = accuracy_score(labels, preds)
        f1v = f1_score(labels, preds, average='weighted')
        prec = precision_score(labels, preds, average='weighted', zero_division=0)
        
        # 计算 Sensitivity 和 Specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # 计算 AUC
        if probs is not None and len(probs) > 0:
            try:
                aucv = roc_auc_score(labels, probs)
            except Exception:
                aucv = float('nan')
        else:
            aucv = float('nan')
        
        # 保存指标
        fold_metrics['accuracy'].append(acc)
        fold_metrics['sensitivity'].append(sensitivity)
        fold_metrics['specificity'].append(specificity)
        fold_metrics['f1'].append(f1v)
        fold_metrics['precision'].append(prec)
        fold_metrics['auc'].append(aucv)
        
        print(f"Fold {i}: acc={acc:.4f}, sen={sensitivity:.4f}, spe={specificity:.4f}, f1={f1v:.4f}, prec={prec:.4f}, auc={aucv:.4f}")
    except Exception as e:
        print(f"Warning: could not load predictions for fold {i}: {e}")

# 写入汇总文件
summary_file = f"comparison_{COMPARISON_MODE}_summary.txt"
with open(summary_file, 'w') as f:
    f.write(f'Comparison Study Summary: {COMPARISON_MODE}\n')
    f.write('===============================\n')
    for k, vals in fold_metrics.items():
        vals_clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if len(vals_clean) == 0:
            mean_v = float('nan')
            std_v = float('nan')
        else:
            mean_v = np.mean(vals_clean)
            std_v = np.std(vals_clean)
        f.write(f"{k.capitalize():10s}: {mean_v:.4f} ± {std_v:.4f}\n")

print(f"Saved summary to {summary_file}")

# 生成汇总图片
plot_loss_comparison(log_paths, save_path=f'{prefix}loss_curves_comparison.png')
plot_roc_comparison(pred_npz_paths, save_path=f'{prefix}roc_curves_comparison.png')
plot_confusion_matrix_average(pred_npz_paths, save_path=f'{prefix}confusion_matrix_average.png')

print(f"Saved comparison figures with prefix {prefix}")

"""
可视化训练过程：Loss曲线 和 预测vs真实值（科研风格）
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.manifold import TSNE
import seaborn as sns

matplotlib.use('Agg')  # 使用非交互后端
sns.set_style("whitegrid")  # 科研风格背景
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.figsize'] = (8, 6)


def plot_training_curves(log_path="training_log.csv", save_dir=".", fold_num=None):
    """
    绘制 Loss 曲线（科研风格）
    
    Args:
        log_path: 训练日志 CSV 文件路径
        save_dir: 图片保存目录
        fold_num: 折号（用于多折交叉验证）
    """
    # 读取日志
    df = pd.read_csv(log_path)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Loss 曲线
    ax.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', linewidth=2.5, 
            markersize=5, color='#1f77b4', markerfacecolor='white', markeredgewidth=2)
    ax.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s', linewidth=2.5, 
            markersize=5, color='#ff7f0e', markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if fold_num is not None:
        save_path = f"{save_dir}/loss_curve_fold{fold_num}.png"
    else:
        save_path = f"{save_dir}/loss_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved loss curve to {save_path}")
    plt.close()


def plot_predictions(all_preds, all_labels, all_probs=None, save_dir=".", title="Predictions vs Ground Truth", fold_num=None):
    """
    绘制预测vs真实值的多个图表（科研风格），单独保存每个图
    
    Args:
        all_preds: 预测值列表
        all_labels: 真实标签列表
        all_probs: 预测概率列表（用于ROC曲线）
        save_dir: 图片保存目录
        title: 图表标题
        fold_num: 折号（用于多折交叉验证）
    """
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 1. 混淆矩阵（单独图）
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax, 
                xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'],
                annot_kws={'fontsize': 14, 'fontweight': 'bold'}, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    if fold_num is not None:
        save_path = f"{save_dir}/confusion_matrix_fold{fold_num}.png"
    else:
        save_path = f"{save_dir}/confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {save_path}")
    plt.close()
    
    # 2. 预测vs真实值分布对比（单独图）
    fig, ax = plt.subplots(figsize=(9, 6))
    classes = ['Class 0', 'Class 1']
    true_counts = [all_labels.count(0), all_labels.count(1)]
    pred_counts = [all_preds.count(0), all_preds.count(1)]
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, true_counts, width, label='Ground Truth', alpha=0.8, color='#2E86AB', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, pred_counts, width, label='Predictions', alpha=0.8, color='#A23B72', edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution: Ground Truth vs Predictions', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    # place legend at the upper-right corner
    ax.legend(loc='upper right', ncol=1,
              frameon=True, fancybox=True, shadow=True, fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    if fold_num is not None:
        save_path = f"{save_dir}/pred_distribution_fold{fold_num}.png"
    else:
        save_path = f"{save_dir}/pred_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved prediction distribution to {save_path}")
    plt.close()
    
    # 3. 运行精度曲线（单独图）
    fig, ax = plt.subplots(figsize=(10, 6))
    correctness = [1 if pred == label else 0 for pred, label in zip(all_preds, all_labels)]
    cumsum_correct = np.cumsum(correctness)
    sample_indices = np.arange(1, len(all_labels) + 1)
    running_accuracy = cumsum_correct / sample_indices
    
    ax.plot(sample_indices, running_accuracy, linewidth=2.5, color='#06D6A0', alpha=0.8, label='Cumulative Accuracy')
    final_acc = sum(correctness) / len(correctness)
    ax.axhline(y=final_acc, color='#EF476F', linestyle='--', linewidth=2.5, alpha=0.9, 
               label=f'Final Accuracy: {final_acc:.4f}')
    
    # 填充置信区间效果（可选）
    window = max(5, len(all_labels) // 20)
    smooth_acc = pd.Series(running_accuracy).rolling(window=window, center=True, min_periods=1).mean()
    ax.fill_between(sample_indices, running_accuracy, smooth_acc, alpha=0.2, color='#06D6A0')
    
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Prediction Accuracy', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    if fold_num is not None:
        save_path = f"{save_dir}/accuracy_curve_fold{fold_num}.png"
    else:
        save_path = f"{save_dir}/accuracy_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy curve to {save_path}")
    plt.close()
    
    # 4. ROC 曲线 (如果有预测概率)
    if all_probs is not None and len(all_probs) > 0:
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # 计算 ROC 曲线
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        # 绘制 ROC 曲线
        ax.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier', alpha=0.7)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        if fold_num is not None:
            save_path = f"{save_dir}/roc_curve_fold{fold_num}.png"
        else:
            save_path = f"{save_dir}/roc_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
        plt.close()


if __name__ == "__main__":
    plot_training_curves("training_log.csv", ".")


def plot_tsne(features, labels, save_dir=".", perplexity=30, fold_num=None, suffix=None):
    """
    绘制 t-SNE 降维可视化
    
    Args:
        features: (N, D) 的特征矩阵
        labels: (N,) 的标签数组
        save_dir: 图片保存目录
        perplexity: t-SNE 参数
        fold_num: 折号（用于多折交叉验证）
        suffix: 文件名后缀（如"full_dataset"）
    """
    print("Computing t-SNE... (this may take a while)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000, verbose=1)
    features_tsne = tsne.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 分别绘制两类
    colors = ['#2E86AB', '#A23B72']
    class_names = ['AD (Class 0)', 'FTD (Class 1)']
    
    for label, color, name in zip([0, 1], colors, class_names):
        mask = labels == label
        count = mask.sum()
        ax.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                  c=color, label=f'{name} ({count} samples)', alpha=0.7, s=60, 
                  edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
    
    # 更好的标题
    total_samples = len(features)
    if fold_num is not None:
        title = f't-SNE Feature Visualization (Fold {fold_num})'
    elif suffix == "full_dataset":
        title = 't-SNE Feature Visualization'
    else:
        title = 't-SNE Feature Visualization'
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # 文件名处理
    if fold_num is not None:
        save_path = f"{save_dir}/tsne_fold{fold_num}.png"
    elif suffix is not None:
        save_path = f"{save_dir}/tsne_{suffix}.png"
    else:
        save_path = f"{save_dir}/tsne.png"
        
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE visualization to {save_path}")
    plt.close()


# --- 交叉验证汇总绘图函数 ---

def plot_loss_comparison(log_paths, save_path="loss_curves_comparison.png"):
    """绘制多折 Validation Loss 对比，并绘制平均曲线和标准差阴影"""
    fig, ax = plt.subplots(figsize=(10, 6))
    all_vals = []
    for p in log_paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if 'val_loss' in df.columns:
            vals = df['val_loss'].astype(float).values
            epochs = df['epoch'].astype(int).values
            ax.plot(epochs, vals, linewidth=1.5, alpha=0.6)
            all_vals.append(vals)
    if len(all_vals) == 0:
        print("No val_loss series found for comparison.")
        return
    # 对齐到最小长度
    min_len = min([len(v) for v in all_vals])
    arr = np.array([v[:min_len] for v in all_vals])
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    epochs = np.arange(min_len)
    ax.plot(epochs, mean, color='#d62728', linewidth=2.5, label='Mean Val Loss')
    ax.fill_between(epochs, mean - std, mean + std, color='#d62728', alpha=0.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison (per-fold)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved loss comparison to {save_path}")
    plt.close()


def plot_roc_comparison(pred_npz_paths, save_path="roc_curves_comparison.png"):
    """绘制多折 ROC 曲线对比，并计算平均 ROC 与 AUC 的均值和标准差"""
    import numpy as _np
    from sklearn.metrics import roc_curve, auc
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(8, 7))
    for p in pred_npz_paths:
        try:
            data = _np.load(p, allow_pickle=True)
            probs = data['probs']
            labels = data['labels']
            if probs is None or len(probs) == 0:
                continue
            fpr, tpr, _ = roc_curve(labels, probs)
            ax.plot(fpr, tpr, color='gray', alpha=0.4)
            auc_score = auc(fpr, tpr)
            aucs.append(auc_score)
            # interp
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
        except Exception:
            continue
    if len(tprs) == 0:
        print("No ROC data found for comparison.")
        plt.close()
        return
    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='#1f77b4', linewidth=2.5, label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    ax.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), color='#1f77b4', alpha=0.2)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC comparison to {save_path}")
    plt.close()


def plot_confusion_matrix_average(pred_npz_paths, save_path="confusion_matrix_average.png"):
    """计算每折混淆矩阵并绘制平均混淆矩阵（非归一化计数）"""
    from sklearn.metrics import confusion_matrix as _cm
    cms = []
    for p in pred_npz_paths:
        try:
            data = np.load(p, allow_pickle=True)
            preds = data['preds']
            labels = data['labels']
            cm = _cm(labels, preds)
            cms.append(cm)
        except Exception:
            continue
    if len(cms) == 0:
        print("No confusion matrices to average.")
        return
    # 对齐大小（如果矩阵大小不一致，pad到 2x2）
    cms_arr = np.array([c if c.shape == (2,2) else np.zeros((2,2)) for c in cms])
    mean_cm = cms_arr.mean(axis=0)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(mean_cm, annot=True, fmt='.2f', cmap='Blues', ax=ax, xticklabels=['Class 0','Class 1'], yticklabels=['Class 0','Class 1'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Average Confusion Matrix (per-fold mean)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved average confusion matrix to {save_path}")
    plt.close()

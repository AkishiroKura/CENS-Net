"""
CAM 热力图可视化 - 类似论文中的多样本激活模式展示
========================================================
为 AD 和 FTD 各选择多个样本，生成类似附件图片的 CAM 热力图
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Batch
import torch.nn.functional as F
import os


def compute_cam_for_sample(model, data, target_class):
    """
    计算单个样本的 CAM 激活图
    
    Args:
        model: 训练好的模型
        data: 单个数据样本
        target_class: 目标类别 (0=AD, 1=FTD)
    
    Returns:
        cam_map: (num_nodes,) CAM 激活值，范围 [0, 1]
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 准备输入
    x_orig = data.x.clone().requires_grad_(True)
    data_copy = data.clone()
    data_copy.x = x_orig
    batch_data = Batch.from_data_list([data_copy]).to(device)
    
    # 前向传播
    output = model(batch_data)
    
    # 反向传播获取梯度
    model.zero_grad()
    output[0, target_class].backward()
    
    # CAM = 梯度绝对值的和（按特征维度）
    cam_map = x_orig.grad.abs().sum(dim=-1).cpu().numpy()
    
    # 归一化到 [0, 1]
    cam_map = cam_map / (cam_map.max() + 1e-10)
    
    return cam_map


def plot_cam_heatmaps(model, dataset, device, output_dir='results_explainability',
                      n_samples=None, figsize=None, cmap='viridis'):
    """
    生成类似附件图片的 CAM 热力图可视化
    
    Args:
        model: 训练好的模型
        dataset: 数据集
        device: 设备
        output_dir: 输出目录
        n_samples: 每个类别显示的样本数（None 表示使用所有样本）
        figsize: 图片尺寸（自动根据样本数调整）
        cmap: 色图（'viridis', 'plasma', 'inferno', 'magma', 'cividis'）
    
    Notes:
        - Y轴表示脑区（brain regions）
        - X轴表示被试（subjects）
        - 颜色深浅表示各脑区对分类决策的贡献度
    """
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device).eval()
    
    print("=" * 60)
    print("Generating CAM Heatmap Visualization")
    print("=" * 60)
    
    # 分离 AD 和 FTD 样本
    ad_samples = [d for d in dataset if d.y.item() == 0]
    ftd_samples = [d for d in dataset if d.y.item() == 1]
    
    # 选择样本（如果 n_samples 为 None，使用所有样本）
    if n_samples is None or n_samples >= len(ad_samples):
        ad_selected = ad_samples
        print(f"\nUsing all {len(ad_samples)} AD samples")
    else:
        np.random.seed(42)
        ad_indices = np.random.choice(len(ad_samples), size=n_samples, replace=False)
        ad_selected = [ad_samples[i] for i in ad_indices]
        print(f"\nRandomly selected {len(ad_selected)} AD samples")
    
    if n_samples is None or n_samples >= len(ftd_samples):
        ftd_selected = ftd_samples
        print(f"Using all {len(ftd_samples)} FTD samples")
    else:
        np.random.seed(42)
        ftd_indices = np.random.choice(len(ftd_samples), size=n_samples, replace=False)
        ftd_selected = [ftd_samples[i] for i in ftd_indices]
        print(f"Randomly selected {len(ftd_selected)} FTD samples")
    
    # 计算所有样本的 CAM
    print("\nComputing CAM for AD samples...")
    ad_cams = []
    for i, data in enumerate(ad_selected):
        cam = compute_cam_for_sample(model, data, target_class=0)
        ad_cams.append(cam)
        print(f"  AD Sample {i+1}: CAM shape {cam.shape}, range [{cam.min():.3f}, {cam.max():.3f}]")
    
    print("\nComputing CAM for FTD samples...")
    ftd_cams = []
    for i, data in enumerate(ftd_selected):
        cam = compute_cam_for_sample(model, data, target_class=1)
        ftd_cams.append(cam)
        print(f"  FTD Sample {i+1}: CAM shape {cam.shape}, range [{cam.min():.3f}, {cam.max():.3f}]")
    
    # 检查所有样本的节点数是否一致
    ad_shapes = [cam.shape[0] for cam in ad_cams]
    ftd_shapes = [cam.shape[0] for cam in ftd_cams]
    all_shapes = ad_shapes + ftd_shapes
    
    if len(set(all_shapes)) > 1:
        print(f"\n⚠️  Warning: Samples have different number of nodes!")
        print(f"   AD shapes: {ad_shapes}")
        print(f"   FTD shapes: {ftd_shapes}")
        
        # 使用最小的节点数进行截断
        min_nodes = min(all_shapes)
        print(f"   Truncating all samples to {min_nodes} nodes")
        
        ad_cams = [cam[:min_nodes] for cam in ad_cams]
        ftd_cams = [cam[:min_nodes] for cam in ftd_cams]
    
    # 转换为矩阵（samples × nodes）
    ad_cams = np.array(ad_cams)  # shape: (n_ad_samples, num_nodes)
    ftd_cams = np.array(ftd_cams)  # shape: (n_ftd_samples, num_nodes)
    
    # 只保留纤维束节点（前25个），去掉脑区节点
    fiber_nodes = 25
    if ad_cams.shape[1] > fiber_nodes:
        print(f"\nExtracting fiber tract nodes (0-{fiber_nodes-1}) from {ad_cams.shape[1]} total nodes")
        ad_cams = ad_cams[:, :fiber_nodes]
        ftd_cams = ftd_cams[:, :fiber_nodes]
    
    print(f"\nFinal CAM matrices (fiber tracts only):")
    print(f"  AD: {ad_cams.shape}")
    print(f"  FTD: {ftd_cams.shape}")
    
    # 转置矩阵：从 (samples × nodes) 变为 (nodes × samples)
    ad_cams_T = ad_cams.T
    ftd_cams_T = ftd_cams.T
    print(f"\nTransposed for visualization:")
    print(f"  AD: {ad_cams_T.shape} (nodes × samples)")
    print(f"  FTD: {ftd_cams_T.shape} (nodes × samples)")
    
    # 自动调整图片尺寸
    if figsize is None:
        width = 20  # 更大的宽度
        height = 14  # 更大的高度
        figsize = (width, height)
        print(f"\nAuto-adjusted figure size: {figsize}")
    
    # ===== 绘制热力图 =====
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # AD 热力图（上）
    im1 = axes[0].imshow(ad_cams_T, aspect='auto', cmap=cmap, interpolation='nearest')
    axes[0].set_xlabel('AD Subjects', fontsize=16, fontweight='bold')
    axes[0].set_xticks([])
    axes[0].set_ylabel('Fiber Tracts', fontsize=16, fontweight='bold')
    for label in axes[0].get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(12)
    axes[0].set_title('Class Activation Map - AD', fontsize=18, fontweight='bold', 
                      color='#3498DB', pad=20)
    axes[0].text(0.5, -0.08, '(a) Alzheimer\'s Disease', transform=axes[0].transAxes, fontsize=18, fontweight='bold', ha='center', va='top')
    
    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=axes[0], orientation='vertical', pad=0.02)
    cbar1.set_label('Activation', fontsize=14, fontweight='bold')
    cbar1.ax.tick_params(labelsize=12)
    
    # FTD 热力图（下）
    im2 = axes[1].imshow(ftd_cams_T, aspect='auto', cmap=cmap, interpolation='nearest')
    axes[1].set_xlabel('FTD Subjects', fontsize=16, fontweight='bold')
    axes[1].set_xticks([])
    axes[1].set_ylabel('Fiber Tracts', fontsize=16, fontweight='bold')
    for label in axes[1].get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(12)
    axes[1].set_title('Class Activation Map - FTD', fontsize=18, fontweight='bold',
                      color='#E74C3C', pad=20)
    axes[1].text(0.5, -0.08, '(b) Frontotemporal Dementia', transform=axes[1].transAxes, fontsize=18, fontweight='bold', ha='center', va='top')
    
    # 添加颜色条
    cbar2 = plt.colorbar(im2, ax=axes[1], orientation='vertical', pad=0.02)
    cbar2.set_label('Activation', fontsize=14, fontweight='bold')
    cbar2.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(output_dir, 'cam_heatmap_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    plt.show()
    
    return fig, ad_cams, ftd_cams


def plot_cam_with_node_labels(model, dataset, device, node_names=None,
                               output_dir='results_explainability',
                               n_samples=None, top_k=20, cmap='viridis'):
    """
    生成带节点标签的 CAM 热力图（只显示 top-k 个最显著节点）
    
    Args:
        node_names: 节点名称列表（可选）
        n_samples: 每个类别显示的样本数（None 表示使用所有样本）
        top_k: 只显示前 k 个最重要的节点
    """
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device).eval()
    
    print("=" * 60)
    print("Generating CAM Heatmap with Node Labels (Top-K)")
    print("=" * 60)
    
    # 分离样本
    ad_samples = [d for d in dataset if d.y.item() == 0]
    ftd_samples = [d for d in dataset if d.y.item() == 1]
    
    # 选择样本
    if n_samples is None or n_samples >= len(ad_samples):
        ad_selected = ad_samples
    else:
        np.random.seed(42)
        ad_indices = np.random.choice(len(ad_samples), size=n_samples, replace=False)
        ad_selected = [ad_samples[i] for i in ad_indices]
    
    if n_samples is None or n_samples >= len(ftd_samples):
        ftd_selected = ftd_samples
    else:
        np.random.seed(42)
        ftd_indices = np.random.choice(len(ftd_samples), size=n_samples, replace=False)
        ftd_selected = [ftd_samples[i] for i in ftd_indices]
    
    # 计算 CAM
    ad_cams = [compute_cam_for_sample(model, data, 0) for data in ad_selected]
    ftd_cams = [compute_cam_for_sample(model, data, 1) for data in ftd_selected]
    
    # 检查节点数一致性
    ad_shapes = [cam.shape[0] for cam in ad_cams]
    ftd_shapes = [cam.shape[0] for cam in ftd_cams]
    all_shapes = ad_shapes + ftd_shapes
    
    if len(set(all_shapes)) > 1:
        print(f"⚠️  Warning: Samples have different node counts: {set(all_shapes)}")
        min_nodes = min(all_shapes)
        print(f"   Truncating to {min_nodes} nodes")
        ad_cams = [cam[:min_nodes] for cam in ad_cams]
        ftd_cams = [cam[:min_nodes] for cam in ftd_cams]
    
    # 转换为数组
    ad_cams = np.array(ad_cams)
    ftd_cams = np.array(ftd_cams)
    
    # 找出平均激活最高的 top-k 节点
    combined_mean = np.concatenate([ad_cams, ftd_cams], axis=0).mean(axis=0)
    top_indices = np.argsort(combined_mean)[::-1][:top_k]
    
    # 提取 top-k 节点的激活
    ad_cams_top = ad_cams[:, top_indices]
    ftd_cams_top = ftd_cams[:, top_indices]
    
    # 节点标签
    if node_names is not None:
        labels = [node_names[i] if i < len(node_names) else f'Node{i}' for i in top_indices]
    else:
        labels = [f'Node{i}' for i in top_indices]
    
    # 简化标签（如果太长）
    labels = [label[:20] + '...' if len(label) > 20 else label for label in labels]
    
    # 自动调整图片尺寸
    width = max(16, top_k * 0.5)
    height = max(12, (len(ad_selected) + len(ftd_selected)) * 0.35)
    figsize = (width, height)
    
    # 绘制
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # AD
    im1 = axes[0].imshow(ad_cams_top, aspect='auto', cmap=cmap, interpolation='nearest')
    axes[0].set_ylabel('AD Subjects', fontsize=14, fontweight='bold')
    axes[0].set_yticks(range(len(ad_selected)))
    axes[0].set_yticklabels([f'S{i+1}' for i in range(len(ad_selected))], fontsize=11, fontweight='bold')
    axes[0].set_title(f'CAM - AD (Top {top_k} Nodes)', fontsize=16, fontweight='bold', color='#3498DB')
    axes[0].text(0.5, -0.08, '(a)', transform=axes[0].transAxes, fontsize=18, fontweight='bold', ha='center', va='top')
    cbar1 = plt.colorbar(im1, ax=axes[0], pad=0.02)
    cbar1.set_label('Activation', fontsize=12, fontweight='bold')
    cbar1.ax.tick_params(labelsize=10)
    
    # FTD
    im2 = axes[1].imshow(ftd_cams_top, aspect='auto', cmap=cmap, interpolation='nearest')
    axes[1].set_ylabel('FTD Subjects', fontsize=14, fontweight='bold')
    axes[1].set_yticks(range(len(ftd_selected)))
    axes[1].set_yticklabels([f'S{i+1}' for i in range(len(ftd_selected))], fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Brain Regions', fontsize=14, fontweight='bold')
    axes[1].set_title(f'CAM - FTD (Top {top_k} Nodes)', fontsize=16, fontweight='bold', color='#E74C3C')
    axes[1].text(0.5, -0.08, '(b)', transform=axes[1].transAxes, fontsize=18, fontweight='bold', ha='center', va='top')
    axes[1].set_xticks(range(top_k))
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    cbar2 = plt.colorbar(im2, ax=axes[1], pad=0.02)
    cbar2.set_label('Activation', fontsize=12, fontweight='bold')
    cbar2.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'cam_heatmap_top{top_k}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    print(__doc__)
    print("\n使用示例:")
    print("="*60)
    print("""
from cam_heatmap_vis import plot_cam_heatmaps
from models import DualPathModel
from dataloader import T2RDataset
import torch

# 加载模型和数据
model = DualPathModel(gnn_hidden=32, global_hidden=32, global_out=16, num_classes=2, dropout=0.5)
model.load_state_dict(torch.load('best_model_fold1.pt'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = T2RDataset('F:\\\\workspace1\\\\dataset')

# 生成 CAM 热力图
plot_cam_heatmaps(model, dataset, device, n_samples=4, cmap='viridis')

# 生成带节点标签的版本
plot_cam_with_node_labels(model, dataset, device, n_samples=4, top_k=20)
    """)

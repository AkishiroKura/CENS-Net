"""
可解释性可视化模块（论文版）
=============================================
专注于论文常用的三种可视化：

1. Integrated Gradients 特征重要性 - 基于梯度的准确方法
2. Grad-CAM 节点重要性热力图 - 展示脑区重要性
3. 连接矩阵热力图 - 展示重要的脑网络连接
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Batch
import os

# 论文风格设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2


class GradCAM:
    """Grad-CAM for GNN - 计算节点重要性"""
    
    def __init__(self, model):
        self.model = model
    
    def compute(self, data, target_class=None):
        """
        计算节点重要性分数
        
        Returns:
            node_importance: (num_nodes,) 归一化的节点重要性
            pred: 预测类别
            prob: 目标类别概率
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # 包装成 batch 以兼容 DualPathModel
        x_orig = data.x.clone().requires_grad_(True)
        data_copy = data.clone()
        data_copy.x = x_orig
        batch_data = Batch.from_data_list([data_copy]).to(device)
        
        output = self.model(batch_data)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # 节点重要性 = 梯度绝对值
        node_importance = x_orig.grad.abs().sum(dim=-1).cpu().numpy()
        node_importance = node_importance / (node_importance.max() + 1e-10)
        
        pred = output.argmax(dim=1).item()
        prob = F.softmax(output, dim=1)[0, target_class].item()
        
        return node_importance, pred, prob


class GradientSHAP:
    """使用GradientSHAP计算特征重要性（基于梯度的SHAP）"""
    
    def __init__(self, model, n_samples=50):
        """
        Args:
            model: 待解释的模型
            n_samples: baseline采样数量
        """
        self.model = model
        self.n_samples = n_samples
        self.device = next(model.parameters()).device
        print(f"  GradientSHAP initialized (n_samples={n_samples})")
    
    def compute(self, data_list, target_class=None):
        """
        计算样本级特征重要性
        
        Args:
            data_list: 数据样本列表
            target_class: 目标类别 (0=AD, 1=FTD)
        
        Returns:
            all_importance: (n_samples, num_features) 每个样本的特征重要性
        """
        self.model.eval()
        all_importance = []
        
        for data in data_list:
            data = data.to(self.device)
            num_features = data.x.size(1)
            
            # 对多个随机插值点计算梯度并平均
            shap_values = torch.zeros(data.x.size(0), num_features).to(self.device)
            
            # 使用零向量作为baseline
            baseline_x = torch.zeros_like(data.x)
            
            for _ in range(self.n_samples):
                # 随机选择插值点
                alpha = torch.rand(1).to(self.device)
                
                # 插值
                interpolated_x = baseline_x + alpha * (data.x - baseline_x)
                interpolated_x.requires_grad_(True)
                
                # 前向传播
                data_interp = data.clone()
                data_interp.x = interpolated_x
                batch_data = Batch.from_data_list([data_interp])
                
                output = self.model(batch_data)
                
                # 获取目标类别
                if target_class is None:
                    tc = output.argmax(dim=1).item()
                else:
                    tc = target_class
                
                # 反向传播
                self.model.zero_grad()
                output[0, tc].backward()
                
                # 累积梯度
                if interpolated_x.grad is not None:
                    shap_values += interpolated_x.grad.detach()
            
            # 平均梯度
            avg_grads = shap_values / self.n_samples
            
            # GradientSHAP: (x - baseline) * gradient
            feature_shap = (data.x * avg_grads).abs().sum(dim=0).cpu().numpy()
            
            all_importance.append(feature_shap)
        
        return np.array(all_importance)  # (n_samples, num_features)


# ============================================================
# 特征分组定义：物理信息 vs 扩散信息
# ============================================================

def get_feature_groups():
    """定义特征组：物理信息 vs 扩散信息
    
    物理信息：纤维束追踪得到的物理特性
        - number_of_tracts (Tracts): 纤维束数量
        - curl: 纤维束弯曲度
        - intersect_ratio (Ratio): 交叉比例
    
    扩散信息：微观结构的扩散特性
        - qa: 各向异性量化
        - md: 平均扩散率
        - rd: 径向扩散率
        - gfa: 广义各向异性分数
    """
    # 7个指标，每个2种统计量(μ, σ)，共14个特征（已删除max）
    # 顺序: Tracts_μ, Tracts_σ, Curl_μ, Curl_σ, 
    #       QA_μ, QA_σ, MD_μ, MD_σ, RD_μ, RD_σ,
    #       GFA_μ, GFA_σ, Ratio_μ, Ratio_σ
    # 删除了max特征，从21维降至14维
    return {
        'Physical': [0, 1, 2, 3, 12, 13],  # Tracts, Curl, Ratio (6个特征)
        'Diffusion': [4, 5, 6, 7, 8, 9, 10, 11]  # QA, MD, RD, GFA (8个特征)
    }

def aggregate_to_groups(importance_matrix, feature_groups):
    """
    将原始特征重要性聚合到特征组
    
    Args:
        importance_matrix: (n_samples, n_features) 原始特征重要性
        feature_groups: dict {group_name: [feature_indices]}
    
    Returns:
        group_importance: dict {group_name: (n_samples,)} 组级别重要性
    """
    group_importance = {}
    for group_name, indices in feature_groups.items():
        # 组内特征求和
        group_importance[group_name] = importance_matrix[:, indices].sum(axis=1)
    return group_importance


# ============================================================
# 论文级可视化函数
# ============================================================

def plot_feature_grouped(ad_importance, ftd_importance, save_path=None):
    """
    绘制特征组级别的重要性对比（带样本分布）
    
    Args:
        ad_importance: (n_samples, n_features) AD样本的特征重要性
        ftd_importance: (n_samples, n_features) FTD样本的特征重要性
        save_path: 保存路径
    """
    feature_groups = get_feature_groups()
    
    # 聚合到组
    ad_groups = aggregate_to_groups(ad_importance, feature_groups)
    ftd_groups = aggregate_to_groups(ftd_importance, feature_groups)
    
    # 准备绘图数据
    plot_data = []
    for group_name in ['Microstructure', 'Connectivity', 'Morphology']:
        for val in ad_groups[group_name]:
            plot_data.append({'Group': group_name, 'Importance': val, 'Class': 'AD'})
        for val in ftd_groups[group_name]:
            plot_data.append({'Group': group_name, 'Importance': val, 'Class': 'FTD'})
    
    df = pd.DataFrame(plot_data)
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # 左图：分组对比（boxplot）
    ax1 = axes[0]
    group_order = ['Microstructure', 'Connectivity', 'Morphology']
    palette = {'AD': '#3498DB', 'FTD': '#E74C3C'}
    
    sns.boxplot(data=df, x='Group', y='Importance', hue='Class', 
                order=group_order, palette=palette, ax=ax1, linewidth=1.5)
    
    ax1.set_title('Feature Group Importance Distribution', fontweight='bold', fontsize=16)
    ax1.set_xlabel('Feature Group', fontsize=14, fontweight='bold')
    ax1.set_ylabel('SHAP Value (normalized)', fontsize=14, fontweight='bold')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.legend(title='Class', fontsize=12, title_fontsize=13)
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='y', labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    
    # 右图：平均值条形图（便于快速比较）
    ax2 = axes[1]
    ad_means = [ad_groups[g].mean() for g in group_order]
    ftd_means = [ftd_groups[g].mean() for g in group_order]
    
    x = np.arange(len(group_order))
    width = 0.35
    
    ax2.bar(x - width/2, ad_means, width, label='AD', color='#3498DB', 
            edgecolor='black', linewidth=1.2)
    ax2.bar(x + width/2, ftd_means, width, label='FTD', color='#E74C3C',
            edgecolor='black', linewidth=1.2)
    
    ax2.set_title('Mean Feature Group Importance', fontweight='bold', fontsize=16)
    ax2.set_xlabel('Feature Group', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Mean SHAP Value', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(group_order, fontsize=12)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.legend(fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='y', labelsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Feature group plot: {save_path}")
    
    return fig


def plot_feature_importance(importance, feature_names=None, top_k=15,
                            title='Feature Importance (SHAP)', save_path=None):
    """
    特征重要性条形图（横向，按重要性排序）
    
    论文常用格式：正负值不同颜色，按绝对值排序
    """
    n = len(importance)
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n)]
    
    idx = np.argsort(np.abs(importance))[::-1][:top_k]
    vals = importance[idx]
    names = [feature_names[i] for i in idx]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#E74C3C' if v > 0 else '#3498DB' for v in vals]
    
    ax.barh(range(len(vals)), vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('SHAP Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    from matplotlib.patches import Patch
    legend = [Patch(facecolor='#E74C3C', label='Positive'),
              Patch(facecolor='#3498DB', label='Negative')]
    ax.legend(handles=legend, loc='lower right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def plot_node_importance(node_importance, node_names=None, title='Node Importance (Grad-CAM)',
                         save_path=None, top_k=20):
    """
    节点重要性条形图（显示 Top-K 重要节点名称）
    """
    n = len(node_importance)
    if node_names is None:
        node_names = [f'Node_{i}' for i in range(n)]
    
    # 取 Top-K
    idx = np.argsort(node_importance)[::-1][:top_k]
    vals = node_importance[idx]
    names = [node_names[i] for i in idx]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlBu_r(vals / vals.max())
    
    ax.barh(range(len(vals)), vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def plot_connectivity_matrix(adj, node_importance=None, node_names=None,
                             title='Brain Connectivity Matrix', save_path=None):
    """
    脑网络连接矩阵热力图（带脑区标签）
    """
    n = adj.shape[0]
    
    # 简化名称
    if node_names is not None:
        short_names = []
        for name in node_names:
            # 简化：去掉 Association_ 前缀，缩写左右
            s = name.replace('Association_', '').replace('_', ' ')
            s = s.replace(' L', '_L').replace(' R', '_R')
            if len(s) > 15:
                s = s[:12] + '..'
            short_names.append(s)
    else:
        short_names = [f'{i}' for i in range(n)]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if node_importance is not None:
        weight = np.outer(node_importance, node_importance)
        adj = adj * weight
    
    mask = adj == 0
    sns.heatmap(adj, cmap='RdYlBu_r', center=0, ax=ax,
                mask=mask, square=True,
                cbar_kws={'label': 'Connection Weight', 'shrink': 0.6},
                xticklabels=short_names, yticklabels=short_names)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6, fontweight='bold')
    ax.set_xlabel('Brain Region', fontsize=12)
    ax.set_ylabel('Brain Region', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def plot_feature_by_category(ad_imp, ftd_imp, feature_names, save_path=None):
    """
    分类特征重要性：物理特征 vs 扩散张量特征
    
    物理特征：Tracts（纤维束数量）, Curl（曲率）, Ratio（交叉比）
    扩散张量：QA, MD, RD, GFA
    
    将同一特征的 μ/max/σ 合并为一个值（取绝对值最大的）
    """
    # 定义特征前缀
    physical_features = ['Tracts', 'Curl', 'Ratio']
    diffusion_features = ['QA', 'MD', 'RD', 'GFA']
    
    def aggregate_feature(importance, feature_names, prefix):
        """将同一前缀的 μ/max/σ 合并（取平均绝对值，保留符号）"""
        idx = [i for i, n in enumerate(feature_names) if n.startswith(prefix)]
        if not idx:
            return 0.0
        vals = importance[idx]
        # 取绝对值之和作为重要性，符号取主导方向
        return np.sum(vals)
    
    # 合并特征
    ad_phys = {f: aggregate_feature(ad_imp, feature_names, f) for f in physical_features}
    ad_diff = {f: aggregate_feature(ad_imp, feature_names, f) for f in diffusion_features}
    ftd_phys = {f: aggregate_feature(ftd_imp, feature_names, f) for f in physical_features}
    ftd_diff = {f: aggregate_feature(ftd_imp, feature_names, f) for f in diffusion_features}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # AD - 物理特征
    names = list(ad_phys.keys())
    vals = list(ad_phys.values())
    colors = ['#E74C3C' if v > 0 else '#3498DB' for v in vals]
    axes[0, 0].barh(range(len(vals)), vals, color=colors, edgecolor='black', height=0.6)
    axes[0, 0].set_yticks(range(len(vals)))
    axes[0, 0].set_yticklabels(names, fontsize=13, fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].axvline(x=0, color='black', linewidth=0.8)
    axes[0, 0].set_title('AD - Physical', fontweight='bold', fontsize=14, color='#3498DB')
    axes[0, 0].set_xlabel('SHAP Value', fontsize=12, fontweight='bold')
    axes[0, 0].tick_params(axis='x', labelsize=11)
    
    # AD - 扩散张量特征
    names = list(ad_diff.keys())
    vals = list(ad_diff.values())
    colors = ['#E74C3C' if v > 0 else '#3498DB' for v in vals]
    axes[0, 1].barh(range(len(vals)), vals, color=colors, edgecolor='black', height=0.6)
    axes[0, 1].set_yticks(range(len(vals)))
    axes[0, 1].set_yticklabels(names, fontsize=13, fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].axvline(x=0, color='black', linewidth=0.8)
    axes[0, 1].set_title('AD - Diffusion Tensor', fontweight='bold', fontsize=14, color='#3498DB')
    axes[0, 1].set_xlabel('SHAP Value', fontsize=12, fontweight='bold')
    axes[0, 1].tick_params(axis='x', labelsize=11)
    
    # FTD - 物理特征
    names = list(ftd_phys.keys())
    vals = list(ftd_phys.values())
    colors = ['#E74C3C' if v > 0 else '#3498DB' for v in vals]
    axes[1, 0].barh(range(len(vals)), vals, color=colors, edgecolor='black', height=0.6)
    axes[1, 0].set_yticks(range(len(vals)))
    axes[1, 0].set_yticklabels(names, fontsize=13, fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].axvline(x=0, color='black', linewidth=0.8)
    axes[1, 0].set_title('FTD - Physical', fontweight='bold', fontsize=14, color='#E74C3C')
    axes[1, 0].set_xlabel('SHAP Value', fontsize=12, fontweight='bold')
    axes[1, 0].tick_params(axis='x', labelsize=11)
    
    # FTD - 扩散张量特征
    names = list(ftd_diff.keys())
    vals = list(ftd_diff.values())
    colors = ['#E74C3C' if v > 0 else '#3498DB' for v in vals]
    axes[1, 1].barh(range(len(vals)), vals, color=colors, edgecolor='black', height=0.6)
    axes[1, 1].set_yticks(range(len(vals)))
    axes[1, 1].set_yticklabels(names, fontsize=13, fontweight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].axvline(x=0, color='black', linewidth=0.8)
    axes[1, 1].set_title('FTD - Diffusion Tensor', fontweight='bold', fontsize=14, color='#E74C3C')
    axes[1, 1].set_xlabel('SHAP Value', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(axis='x', labelsize=11)
    
    plt.suptitle('Feature Importance by Category', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def plot_class_comparison(ad_imp, ftd_imp, feature_names=None, top_k=12,
                          save_path=None):
    """
    AD vs FTD 特征重要性对比图（双列条形图）
    """
    n = len(ad_imp)
    if feature_names is None:
        feature_names = [f'F{i+1}' for i in range(n)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # AD
    idx_ad = np.argsort(np.abs(ad_imp))[::-1][:top_k]
    axes[0].barh(range(len(idx_ad)), ad_imp[idx_ad],
                 color='#3498DB', edgecolor='black', linewidth=0.8)
    axes[0].set_yticks(range(len(idx_ad)))
    axes[0].set_yticklabels([feature_names[i] for i in idx_ad], fontsize=13, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].axvline(x=0, color='black', linewidth=0.8)
    axes[0].set_xlabel('SHAP Value', fontsize=14, fontweight='bold')
    axes[0].set_title('AD', fontsize=16, fontweight='bold', color='#3498DB')
    axes[0].tick_params(axis='x', labelsize=11)
    axes[0].text(0.5, -0.08, '(a)', transform=axes[0].transAxes, fontsize=16, fontweight='bold', ha='center', va='top')
    
    # FTD
    idx_ftd = np.argsort(np.abs(ftd_imp))[::-1][:top_k]
    axes[1].barh(range(len(idx_ftd)), ftd_imp[idx_ftd],
                 color='#E74C3C', edgecolor='black', linewidth=0.8)
    axes[1].set_yticks(range(len(idx_ftd)))
    axes[1].set_yticklabels([feature_names[i] for i in idx_ftd], fontsize=13, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].axvline(x=0, color='black', linewidth=0.8)
    axes[1].set_xlabel('SHAP Value', fontsize=14, fontweight='bold')
    axes[1].set_title('FTD', fontsize=16, fontweight='bold', color='#E74C3C')
    axes[1].tick_params(axis='x', labelsize=11)
    axes[1].text(0.5, -0.08, '(b)', transform=axes[1].transAxes, fontsize=16, fontweight='bold', ha='center', va='top')
    
    plt.suptitle('Feature Importance Comparison', fontsize=18, fontweight='bold', y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


# ============================================================
# 辅助函数：读取节点名称
# ============================================================

def get_node_names(csv_path):
    """
    从 CSV 文件读取纤维束和脑区名称
    
    Returns:
        fiber_names: 纤维束名称列表（行名）
        region_names: 脑区名称列表（列名）
    """
    df = pd.read_csv(csv_path, index_col=0)
    fiber_names = list(df.index)
    region_names = list(df.columns)
    return fiber_names, region_names


# ============================================================
# 一键分析函数
# ============================================================

def run_analysis(model, dataset, device, output_dir='results_explainability',
                 n_samples=15, sample_csv_path=None):
    """
    运行完整可解释性分析，生成论文图表
    
    输出：
    - feature_by_category.png           AD/FTD 特征对比（按类别分组）
    - results.npz                       数值结果
    """
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device).eval()
    
    print("=" * 50)
    print("Running Explainability Analysis")
    print("=" * 50)
    
    # 分类样本
    ad_samples = [d for d in dataset if d.y.item() == 0][:n_samples]
    ftd_samples = [d for d in dataset if d.y.item() == 1][:n_samples]
    
    # 特征名（已删除max特征，从21维降至14维）
    feature_names = [
        'Tracts_μ', 'Tracts_σ',
        'Curl_μ', 'Curl_σ',
        'QA_μ', 'QA_σ',
        'MD_μ', 'MD_σ',
        'RD_μ', 'RD_σ',
        'GFA_μ', 'GFA_σ',
        'Ratio_μ', 'Ratio_σ'
    ]
    
    # 1. GradientSHAP 特征重要性
    print("\n[1/2] Computing feature importance with GradientSHAP...")
    
    explainer = GradientSHAP(model, n_samples=50)
    
    # 计算样本级特征重要性
    ad_importance = explainer.compute(ad_samples, target_class=0)
    ftd_importance = explainer.compute(ftd_samples, target_class=1)
    
    feature_names = feature_names[:ad_importance.shape[1]]
    
    # 聚合到物理信息 vs 扩散信息
    feature_groups = get_feature_groups()
    
    # 计算每组的总重要性
    ad_physical = ad_importance[:, feature_groups['Physical']].sum(axis=1).mean()
    ad_diffusion = ad_importance[:, feature_groups['Diffusion']].sum(axis=1).mean()
    ftd_physical = ftd_importance[:, feature_groups['Physical']].sum(axis=1).mean()
    ftd_diffusion = ftd_importance[:, feature_groups['Diffusion']].sum(axis=1).mean()
    
    # 计算每个特征的平均重要性（用于细节分析）
    ad_feat_mean = ad_importance.mean(axis=0)
    ftd_feat_mean = ftd_importance.mean(axis=0)
    
    # 按大指标聚合（每个指标有2个统计量：μ, σ，已删除max）
    # 顺序: Tracts(0-1), Curl(2-3), QA(4-5), MD(6-7), RD(8-9), GFA(10-11), Ratio(12-13)
    metric_indices = {
        'Tracts': [0, 1],
        'Curl': [2, 3],
        'QA': [4, 5],
        'MD': [6, 7],
        'RD': [8, 9],
        'GFA': [10, 11],
        'Ratio': [12, 13]
    }
    
    # 物理信息指标
    physical_metrics = ['Tracts', 'Curl', 'Ratio']
    # 扩散信息指标
    diffusion_metrics = ['QA', 'MD', 'RD', 'GFA']
    
    # 计算每个大指标的重要性（用平均值，更公平）
    ad_metrics = {m: ad_feat_mean[metric_indices[m]].mean() for m in metric_indices.keys()}
    ftd_metrics = {m: ftd_feat_mean[metric_indices[m]].mean() for m in metric_indices.keys()}
    
    # 同时记录各指标中最重要的统计量
    stat_names = ['μ', 'σ']
    ad_top_stat = {}
    ftd_top_stat = {}
    for m in metric_indices.keys():
        vals = ad_feat_mean[metric_indices[m]]
        ad_top_stat[m] = stat_names[np.argmax(vals)]
        vals = ftd_feat_mean[metric_indices[m]]
        ftd_top_stat[m] = stat_names[np.argmax(vals)]
    
    # === 可视化1: 两大类特征组对比 ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # AD 特征组重要性
    groups = ['Physical\n(Tracts, Curl, Ratio)', 'Diffusion\n(QA, MD, RD, GFA)']
    vals = [ad_physical, ad_diffusion]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = axes[0].bar(range(len(groups)), vals, color=colors, edgecolor='black', 
                       linewidth=1.5, alpha=0.85, width=0.6)
    axes[0].set_xticks(range(len(groups)))
    axes[0].set_xticklabels(groups, fontsize=13, fontweight='bold')
    axes[0].set_title('AD - Feature Importance by Category', fontweight='bold', 
                      fontsize=16, color='#2C3E50', pad=15)
    axes[0].set_ylabel('Total Importance (SHAP)', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_ylim(0, max(vals) * 1.2)
    axes[0].tick_params(axis='y', labelsize=11)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, vals)):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    axes[0].text(0.5, -0.15, '(a)', transform=axes[0].transAxes, fontsize=16, fontweight='bold', ha='center', va='top')
    
    # FTD 特征组重要性
    vals = [ftd_physical, ftd_diffusion]
    
    bars = axes[1].bar(range(len(groups)), vals, color=colors, edgecolor='black', 
                       linewidth=1.5, alpha=0.85, width=0.6)
    axes[1].set_xticks(range(len(groups)))
    axes[1].set_xticklabels(groups, fontsize=13, fontweight='bold')
    axes[1].set_title('FTD - Feature Importance by Category', fontweight='bold', 
                      fontsize=16, color='#2C3E50', pad=15)
    axes[1].set_ylabel('Total Importance (SHAP)', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_ylim(0, max(vals) * 1.2)
    axes[1].tick_params(axis='y', labelsize=11)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, vals)):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    axes[1].text(0.5, -0.15, '(b)', transform=axes[1].transAxes, fontsize=16, fontweight='bold', ha='center', va='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Feature importance plot: {output_dir}/feature_importance.png")
    
    # === 可视化2: 各类别下Top指标详细分解 ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # AD - 物理信息的各指标
    phys_names = physical_metrics
    phys_vals = [ad_metrics[m] for m in phys_names]
    
    bars = axes[0, 0].barh(range(len(phys_names)), phys_vals, 
                     color='#FF6B6B', edgecolor='black', alpha=0.85)
    axes[0, 0].set_yticks(range(len(phys_names)))
    axes[0, 0].set_yticklabels(phys_names, fontsize=13, fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title('AD - Physical Metrics', fontweight='bold', fontsize=15, pad=10)
    axes[0, 0].set_xlabel('Importance', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    axes[0, 0].tick_params(axis='x', labelsize=11)
    for i, v in enumerate(phys_vals):
        axes[0, 0].text(v * 0.95, i, 
                       f'{v:.3f}', 
                       va='center', ha='right', fontsize=12, fontweight='bold', color='#1a1a1a')
    
    axes[0, 0].text(0.0, -0.18, '(a)', transform=axes[0, 0].transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
    
    # AD - 扩散信息的各指标
    diff_names = diffusion_metrics
    diff_vals = [ad_metrics[m] for m in diff_names]
    
    bars = axes[0, 1].barh(range(len(diff_names)), diff_vals, 
                     color='#4ECDC4', edgecolor='black', alpha=0.85)
    axes[0, 1].set_yticks(range(len(diff_names)))
    axes[0, 1].set_yticklabels(diff_names, fontsize=13, fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_title('AD - Diffusion Metrics', fontweight='bold', fontsize=15, pad=10)
    axes[0, 1].set_xlabel('Importance', fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)
    axes[0, 1].tick_params(axis='x', labelsize=11)
    for i, v in enumerate(diff_vals):
        axes[0, 1].text(v * 0.95, i, 
                       f'{v:.3f}', 
                       va='center', ha='right', fontsize=12, fontweight='bold', color='#1a1a1a')
    
    axes[0, 1].text(0.0, -0.18, '(b)', transform=axes[0, 1].transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
    
    # FTD - 物理信息的各指标
    phys_vals = [ftd_metrics[m] for m in phys_names]
    
    bars = axes[1, 0].barh(range(len(phys_names)), phys_vals, 
                     color='#FF6B6B', edgecolor='black', alpha=0.85)
    axes[1, 0].set_yticks(range(len(phys_names)))
    axes[1, 0].set_yticklabels(phys_names, fontsize=13, fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title('FTD - Physical Metrics', fontweight='bold', fontsize=15, pad=10)
    axes[1, 0].set_xlabel('Importance', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    axes[1, 0].tick_params(axis='x', labelsize=11)
    for i, v in enumerate(phys_vals):
        axes[1, 0].text(v * 0.95, i, 
                       f'{v:.3f}', 
                       va='center', ha='right', fontsize=12, fontweight='bold', color='#1a1a1a')
    
    axes[1, 0].text(0.0, -0.18, '(c)', transform=axes[1, 0].transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
    
    # FTD - 扩散信息的各指标
    diff_vals = [ftd_metrics[m] for m in diff_names]
    
    bars = axes[1, 1].barh(range(len(diff_names)), diff_vals, 
                     color='#4ECDC4', edgecolor='black', alpha=0.85)
    axes[1, 1].set_yticks(range(len(diff_names)))
    axes[1, 1].set_yticklabels(diff_names, fontsize=13, fontweight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title('FTD - Diffusion Metrics', fontweight='bold', fontsize=15, pad=10)
    axes[1, 1].set_xlabel('Importance', fontsize=14, fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    axes[1, 1].tick_params(axis='x', labelsize=11)
    for i, v in enumerate(diff_vals):
        axes[1, 1].text(v * 0.95, i, 
                       f'{v:.3f}', 
                       va='center', ha='right', fontsize=12, fontweight='bold', color='#1a1a1a')
    
    axes[1, 1].text(0.0, -0.18, '(d)', transform=axes[1, 1].transAxes, fontsize=18, fontweight='bold', ha='left', va='top')
    
    plt.suptitle('Detailed Feature Importance by Metric Category', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_details.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Detailed feature plot: {output_dir}/feature_importance_details.png")
    
    print("\n[2/2] Saving results...")
    
    # 保存数值（包含样本级数据和聚合结果）
    save_data = {
        'ad_importance_samples': ad_importance,  # (n_samples, n_features)
        'ftd_importance_samples': ftd_importance,
        'ad_physical': ad_physical,  # 物理信息总重要性
        'ad_diffusion': ad_diffusion,  # 扩散信息总重要性
        'ftd_physical': ftd_physical,
        'ftd_diffusion': ftd_diffusion,
        'feature_names': feature_names
    }
    np.savez(os.path.join(output_dir, 'results.npz'), **save_data)
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_dir}/")
    print("  ✓ feature_importance.png (Physical vs Diffusion overview)")
    print("  ✓ feature_importance_details.png (Breakdown by metric)")
    print(f"\n  AD:  Physical={ad_physical:.3f}, Diffusion={ad_diffusion:.3f}")
    print(f"  FTD: Physical={ftd_physical:.3f}, Diffusion={ftd_diffusion:.3f}")
    print(f"\n  Top metrics in Physical (AD): {sorted([(m, ad_metrics[m]) for m in physical_metrics], key=lambda x: x[1], reverse=True)}")
    print(f"  Top metrics in Diffusion (AD): {sorted([(m, ad_metrics[m]) for m in diffusion_metrics], key=lambda x: x[1], reverse=True)}")
    print("=" * 60)
    
    return {
        'ad_importance_samples': ad_importance,
        'ftd_importance_samples': ftd_importance,
        'ad_physical': ad_physical,
        'ad_diffusion': ad_diffusion,
        'ftd_physical': ftd_physical,
        'ftd_diffusion': ftd_diffusion
    }


# ============================================================
# BrainNet Viewer 导出
# ============================================================

def export_to_brainnet_viewer(node_importance, node_names, output_path, 
                               adjacency_matrix=None, label=''):
    """
    导出为 BrainNet Viewer 格式（严格保持节点顺序）
    
    Args:
        node_importance: (num_nodes,) 节点重要性分数
        node_names: 节点名称列表
        output_path: 输出文件路径（不含扩展名）
        adjacency_matrix: (num_nodes, num_nodes) 邻接矩阵（可选）
        label: 标签（如 'AD' 或 'FTD'）
    
    输出文件：
        - {output_path}.node: 节点文件（MNI坐标 + 重要性）
        - {output_path}.edge: 边文件（邻接矩阵）
        - {output_path}_mapping.txt: 节点映射关系（验证用）
    """
    # HCP842 纤维束的 MNI 坐标（标准名称）
    TRACT_COORDS = {
        'Acoustic_Radiation_L': (-42, -25, 8),
        'Acoustic_Radiation_R': (42, -25, 8),
        'Arcuate_Fasciculus_L': (-45, -30, 25),
        'Arcuate_Fasciculus_R': (45, -30, 25),
        'Cingulum_L': (-8, -20, 35),
        'Cingulum_R': (8, -20, 35),
        'Cortico_Spinal_Tract_L': (-20, -20, 45),
        'Cortico_Spinal_Tract_R': (20, -20, 45),
        'Fornix_L': (-5, -5, 15),
        'Fornix_R': (5, -5, 15),
        'Frontal_Aslant_Tract_L': (-18, 20, 40),
        'Frontal_Aslant_Tract_R': (18, 20, 40),
        'Inferior_Fronto_Occipital_Fasciculus_L': (-35, 0, -5),
        'Inferior_Fronto_Occipital_Fasciculus_R': (35, 0, -5),
        'Inferior_Longitudinal_Fasciculus_L': (-45, -40, -10),
        'Inferior_Longitudinal_Fasciculus_R': (45, -40, -10),
        'Middle_Longitudinal_Fasciculus_L': (-45, -45, 15),
        'Middle_Longitudinal_Fasciculus_R': (45, -45, 15),
        'Optic_Radiation_L': (-30, -75, 5),
        'Optic_Radiation_R': (30, -75, 5),
        'Uncinate_Fasciculus_L': (-30, 15, -15),
        'Uncinate_Fasciculus_R': (30, 15, -15),
        'Vertical_Occipital_Fasciculus_L': (-35, -70, 15),
        'Vertical_Occipital_Fasciculus_R': (35, -70, 15),
        'Superior_Longitudinal_Fasciculus_I_L': (-30, -10, 45),
        'Superior_Longitudinal_Fasciculus_I_R': (30, -10, 45),
        'Superior_Longitudinal_Fasciculus_II_L': (-40, -25, 35),
        'Superior_Longitudinal_Fasciculus_II_R': (40, -25, 35),
        'Superior_Longitudinal_Fasciculus_III_L': (-50, -35, 25),
        'Superior_Longitudinal_Fasciculus_III_R': (50, -35, 25),
        'Anterior_Commissure': (0, 10, -5),
        'Middle_Cerebellar_Peduncle': (0, -40, -30),
    }
    
    def normalize_name(name):
        """
        将数据集中的节点名称转换为标准格式
        例如: Association_ArcuateFasciculusL -> Arcuate_Fasciculus_L
        """
        # 移除前缀
        name = name.replace('Association_', '').replace('Projection_', '')
        name = name.replace('Commissural_', '').replace('Striatum_', '')
        
        # 处理大小写和下划线
        # ArcuateFasciculusL -> Arcuate_Fasciculus_L
        import re
        # 在大写字母前插入下划线
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name)
        # 处理 L/R 后缀
        if name.endswith('_L') or name.endswith('_R'):
            pass  # 已经正确
        elif name.endswith('L'):
            name = name[:-1] + '_L'
        elif name.endswith('R'):
            name = name[:-1] + '_R'
        
        return name
    
    # 严格按照输入顺序处理所有节点
    n_nodes = len(node_importance)
    coords = []
    values = []
    labels = []
    missing_coords = []
    
    print(f"\n  处理 {n_nodes} 个节点...")
    
    for i in range(n_nodes):
        # 获取原始名称
        if node_names is not None and i < len(node_names):
            orig_name = node_names[i]
        else:
            orig_name = f"Node_{i+1}"
        
        # 标准化名称
        norm_name = normalize_name(orig_name)
        
        # 查找坐标
        coord = None
        # 1. 尝试直接匹配
        if orig_name in TRACT_COORDS:
            coord = TRACT_COORDS[orig_name]
        # 2. 尝试标准化后匹配
        elif norm_name in TRACT_COORDS:
            coord = TRACT_COORDS[norm_name]
        # 3. 尝试去掉前缀后匹配
        else:
            for key in TRACT_COORDS:
                if key in orig_name or key in norm_name:
                    coord = TRACT_COORDS[key]
                    break
        
        # 如果还是找不到，使用默认坐标
        if coord is None:
            coord = (0, 0, 0)
            missing_coords.append(i)
        
        coords.append(coord)
        values.append(node_importance[i])
        labels.append(orig_name)  # 保留原始名称用于显示
    
    # 警告缺失坐标
    if missing_coords:
        print(f"  警告: {len(missing_coords)}/{n_nodes} 个节点缺少MNI坐标，使用(0,0,0)占位")
        if len(missing_coords) <= 10:
            print(f"  缺失节点: {[labels[i] for i in missing_coords]}")
    else:
        print(f"  ✓ 所有节点都找到了MNI坐标")
    
    # 归一化重要性分数到 [1, 6] 范围（BrainNet Viewer 的节点大小）
    values = np.array(values)
    values_norm = 1 + 5 * (values - values.min()) / (values.max() - values.min() + 1e-8)
    
    # ===== 导出 .node 文件 =====
    # 格式: X Y Z Color Size Label
    node_file = output_path + '.node'
    with open(node_file, 'w') as f:
        for i in range(n_nodes):
            x, y, z = coords[i]
            color = values[i]  # 原始重要性作为颜色
            size = values_norm[i]  # 归一化后的大小
            label_str = labels[i]
            f.write(f"{x:.2f}\t{y:.2f}\t{z:.2f}\t{color:.6f}\t{size:.2f}\t{label_str}\n")
    
    print(f"Saved node file: {node_file}")
    print(f"  - {n_nodes} nodes (按输入顺序)")
    
    # ===== 导出节点映射文件（用于验证）=====
    mapping_file = output_path + '_mapping.txt'
    with open(mapping_file, 'w') as f:
        f.write("节点映射关系验证文件\n")
        f.write("="*60 + "\n")
        f.write(f"Label: {label}\n")
        f.write(f"Total nodes: {n_nodes}\n")
        f.write(f"Missing MNI coords: {len(missing_coords)}\n")
        f.write("="*60 + "\n\n")
        f.write("Index\tImportance\tMNI_X\tMNI_Y\tMNI_Z\tLabel\n")
        for i in range(n_nodes):
            x, y, z = coords[i]
            f.write(f"{i}\t{values[i]:.6f}\t{x:.1f}\t{y:.1f}\t{z:.1f}\t{labels[i]}\n")
    
    print(f"Saved mapping file: {mapping_file}")
    print(f"  - 用于验证节点对应关系")
    
    # ===== 导出 .edge 文件 =====
    if adjacency_matrix is not None:
        edge_file = output_path + '.edge'
        
        # 确保邻接矩阵大小严格匹配
        adj_shape = adjacency_matrix.shape
        if adj_shape[0] != n_nodes or adj_shape[1] != n_nodes:
            print(f"  警告: 邻接矩阵大小 {adj_shape} 与节点数 {n_nodes} 不匹配")
            if adj_shape[0] >= n_nodes and adj_shape[1] >= n_nodes:
                adj_sub = adjacency_matrix[:n_nodes, :n_nodes]
                print(f"  已截取前 {n_nodes}x{n_nodes} 部分")
            else:
                # 补齐为 n_nodes x n_nodes
                adj_sub = np.zeros((n_nodes, n_nodes))
                min_size = min(adj_shape[0], n_nodes)
                adj_sub[:min_size, :min_size] = adjacency_matrix[:min_size, :min_size]
                print(f"  已补齐为 {n_nodes}x{n_nodes}")
        else:
            adj_sub = adjacency_matrix
        
        # 保存为文本格式（制表符分隔）
        np.savetxt(edge_file, adj_sub, fmt='%.6f', delimiter='\t')
        print(f"Saved edge file: {edge_file}")
        print(f"  - {n_nodes}x{n_nodes} adjacency matrix")
    
    return node_file


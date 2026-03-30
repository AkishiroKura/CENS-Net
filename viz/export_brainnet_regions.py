"""
导出脑区节点到BrainNet Viewer（基于纤维束importance映射）
=========================================================
将纤维束的重要性映射到脑区，然后在BrainNet中显示脑区节点
用于论文/汇报中的脑区可视化
"""

import numpy as np
import pandas as pd
import os
import re


def export_to_brainnet_viewer(node_importance, node_names, output_path, label='', top_k=10):
    """
    导出为 BrainNet Viewer 格式
    
    Args:
        node_importance: (num_nodes,) 节点重要性分数
        node_names: 节点名称列表
        output_path: 输出文件路径（不含扩展名）
        label: 标签（如 'AD' 或 'FTD'）
        top_k: 只保留前K个最显著的节点
    
    Returns:
        node_file: 生成的 .node 文件路径
    """
    # HCP842模板的MNI坐标
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
        'Anterior_Commissure': (0, 10, -5),
        'Middle_Cerebellar_Peduncle': (0, -40, -30),
    }
    
    def normalize_name(name):
        """标准化名称格式"""
        name = name.replace('_', ' ').replace('-', ' ')
        return name
    
    def shorten_label(name):
        """简化标签（用点号区分左右避免下划线显示混乱）"""
        abbreviations = {
            'Acoustic Radiation': 'AR',
            'Arcuate Fasciculus': 'AF',
            'Cingulum': 'Cin',
            'Cortico Spinal Tract': 'CST',
            'Fornix': 'Fx',
            'Frontal Aslant Tract': 'FAT',
            'Inferior Fronto Occipital Fasciculus': 'IFOF',
            'Inferior Longitudinal Fasciculus': 'ILF',
            'Middle Longitudinal Fasciculus': 'MLF',
            'Optic Radiation': 'OR',
            'Uncinate Fasciculus': 'UF',
            'Vertical Occipital Fasciculus': 'VOF',
            'Anterior Commissure': 'AC',
            'Middle Cerebellar Peduncle': 'MCP',
        }
        
        # 先判断左右（用原始名字）
        suffix = ''
        if name.endswith('_L') or name.endswith(' L') or '_L_' in name or ' L ' in name:
            suffix = '.L'
        elif name.endswith('_R') or name.endswith(' R') or '_R_' in name or ' R ' in name:
            suffix = '.R'
        
        # 再标准化名字匹配缩写
        name_norm = normalize_name(name)
        for full, abbr in abbreviations.items():
            if full.lower() in name_norm.lower():
                return abbr + suffix if suffix else abbr
        return name[:10].replace('_', '.')
    
    # 选择Top-K节点
    if top_k and top_k < len(node_importance):
        top_indices = np.argsort(node_importance)[::-1][:top_k]
        top_indices = sorted(top_indices)
    else:
        top_indices = range(len(node_importance))
    
    print(f"\n  总节点数: {len(node_importance)}，保留前 {len(top_indices)} 个最显著节点")
    
    # 生成.node文件
    node_file = output_path + '.node'
    mapping_file = output_path + '_mapping.txt'
    
    nodes_data = []
    missing_coords = 0
    
    for rank, idx in enumerate(sorted(np.argsort(node_importance)[::-1][:len(top_indices)]), 1):
        name = node_names[idx]
        importance = node_importance[idx]
        
        # 查找MNI坐标
        coord = TRACT_COORDS.get(name, None)
        if coord is None:
            # 尝试标准化名称后再查找
            name_norm = name.replace('_', ' ')
            for key in TRACT_COORDS.keys():
                if key.lower().replace('_', ' ') == name_norm.lower():
                    coord = TRACT_COORDS[key]
                    break
        
        if coord is None:
            coord = (0, 0, 0)
            missing_coords += 1
        
        # 归一化importance到1-6范围（节点大小）
        size = 1 + 5 * (importance - node_importance.min()) / (node_importance.max() - node_importance.min() + 1e-10)
        label_short = shorten_label(name)
        
        nodes_data.append({
            'rank': rank,
            'orig_idx': idx,
            'x': coord[0],
            'y': coord[1],
            'z': coord[2],
            'color': importance,
            'size': size,
            'label': label_short,
            'importance': importance
        })
    
    # 写入.node文件
    with open(node_file, 'w') as f:
        for node in nodes_data:
            f.write(f"{node['x']}\t{node['y']}\t{node['z']}\t{node['color']:.6f}\t{node['size']:.2f}\t{node['label']}\n")
    
    print(f"  ✓ 所有节点都找到了MNI坐标" if missing_coords == 0 else f"  ⚠ 缺失MNI坐标: {missing_coords} 个")
    print(f"\n  🎯 前5个最显著的节点:")
    for i, node in enumerate(nodes_data[:5], 1):
        print(f"    {i}. {node['label']}: {node['importance']:.6f}")
    
    print(f"  Saved: {node_file}")
    print(f"    - {len(nodes_data)} nodes (按重要性排序)")
    print(f"    - 标签已简化（避免可视化重叠）")
    
    # 写入mapping文件
    with open(mapping_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BrainNet Viewer 节点映射验证文件\n")
        f.write("="*70 + "\n\n")
        f.write(f"【数据来源】\n")
        f.write(f"  - 节点重要性: 来自深度学习模型的可解释性分析（GradCAM + SHAP）\n")
        f.write(f"  - 分类标签: {label}\n")
        f.write(f"  - 总节点数: {len(node_importance)}\n")
        f.write(f"  - 导出节点数: {len(top_indices)} （前{len(top_indices)}个最显著节点）\n")
        f.write(f"  - 缺失MNI坐标: {missing_coords} 个\n\n")
        f.write(f"【数据对应关系】\n")
        f.write(f"  按重要性降序排列的前K个最显著节点，每一行对应:\n")
        f.write(f"  - Rank: 显著性排名（1=最显著）\n")
        f.write(f"  - OrigIdx: 原始索引（在完整节点列表中的位置）\n")
        f.write(f"  - Importance: 节点重要性分数（显著性）\n")
        f.write(f"  - MNI_X/Y/Z: 脑空间坐标（HCP842模板）\n")
        f.write(f"  - Label: 节点名称（纤维束）\n\n")
        f.write("="*80 + "\n")
        f.write(f"{'Rank':<8}{'OrigIdx':<10}{'Importance':<15}{'MNI_X':<8}{'MNI_Y':<8}{'MNI_Z':<8}Label\n")
        f.write("="*80 + "\n")
        for node in nodes_data:
            f.write(f"{node['rank']:<8}{node['orig_idx']:<10}{node['importance']:<15.6f}"
                   f"{node['x']:<8.1f}{node['y']:<8.1f}{node['z']:<8.1f}{node['label']}\n")
    
    print(f"  Saved: {mapping_file}")
    print(f"    ✓ 可用于验证节点对应关系")
    print(f"    ✓ 重要性分数范围: [{node_importance[top_indices].min():.4f}, {node_importance[top_indices].max():.4f}]")
    
    return node_file


def map_fiber_importance_to_regions(fiber_importance, connectivity_matrix):
    """
    将纤维束importance映射到脑区
    
    Args:
        fiber_importance: (25,) 纤维束重要性分数
        connectivity_matrix: (25纤维束, 25脑区) 连接矩阵
    
    Returns:
        region_importance: (25,) 脑区重要性分数
    """
    # 归一化连接矩阵
    conn_norm = connectivity_matrix.copy()
    conn_norm[np.isnan(conn_norm)] = 0
    conn_norm = (conn_norm - conn_norm.min()) / (conn_norm.max() - conn_norm.min() + 1e-10)
    
    # 加权和：region_importance = Σ(连接强度 × fiber_importance)
    region_importance = conn_norm.T @ fiber_importance
    
    return region_importance


def export_region_importance_to_brainnet(model=None,
                                        dataset=None,
                                        device=None,
                                        model_path='best_model_fold1.pt',
                                        data_root='F:\\workspace1\\dataset',
                                        sample_dir='F:\\workspace1\\dataset\\AD\\003_S_4136_ses-2011-08-10',
                                        output_dir='results_explainability',
                                        n_samples=50,
                                        sensitivity_analysis=True):
    """
    导出脑区节点到BrainNet Viewer（支持敏感性分析）
    
    流程：
    1. 加载模型和数据集（或使用传入的），计算纤维束importance（GradCAM）
    2. 从CSV加载纤维束-脑区连接矩阵（支持多个扩散指标）
    3. 映射纤维束importance到脑区
    4. 导出脑区节点到BrainNet格式
    5. (可选) 敏感性分析：对比不同扩散指标的结果
    """
    import torch
    from explainability import GradCAM
    
    print("="*60)
    print("导出脑区节点到 BrainNet Viewer")
    if sensitivity_analysis:
        print("（多指标敏感性分析）")
    else:
        print("（基于纤维束importance映射）")
    print("="*60)
    
    # 定义要分析的扩散指标（完整的7个指标）
    if sensitivity_analysis:
        metrics = ['number_of_tracts', 'curl', 'qa', 'md', 'rd', 'gfa', 'intersect_ratio']
        print(f"\n将对以下指标进行敏感性分析: {', '.join(metrics)}")
    else:
        metrics = ['number_of_tracts']  # 默认只用number_of_tracts
    
    # 1. 计算纤维束importance（只需计算一次）
    print("\n[1] 计算纤维束重要性（GradCAM）...")
    print("    （所有指标共用相同的GradCAM结果）")
    
    # 如果没有传入model/dataset，则加载
    if model is None or dataset is None:
        from models import DualPathModel
        from dataloader import T2RDataset
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载数据
        dataset = T2RDataset(data_root)
        print(f"  ✓ 加载数据集: {len(dataset)} 样本")
        
        # 加载模型
        model = DualPathModel(
            gnn_hidden=32,
            global_hidden=32,
            global_out=16,
            num_classes=2,
            dropout=0.5
        )
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device).eval()
        print(f"  ✓ 加载模型: {model_path}")
    else:
        if device is None:
            device = next(model.parameters()).device
        print(f"  ✓ 使用传入的模型和数据集")
    
    model = model.eval()
    
    # 计算GradCAM
    cam = GradCAM(model)
    ad_samples = [d for d in dataset if d.y.item() == 0][:n_samples]
    ftd_samples = [d for d in dataset if d.y.item() == 1][:n_samples]
    
    # AD纤维束importance
    ad_nodes = []
    for d in ad_samples:
        imp, _, _ = cam.compute(d, target_class=0)
        ad_nodes.append(imp[:25])  # 只取前25个纤维束节点
    fiber_importance_ad = np.mean(ad_nodes, axis=0)
    
    # FTD纤维束importance
    ftd_nodes = []
    for d in ftd_samples:
        imp, _, _ = cam.compute(d, target_class=1)
        ftd_nodes.append(imp[:25])  # 只取前25个纤维束节点
    fiber_importance_ftd = np.mean(ftd_nodes, axis=0)
    
    print(f"  ✓ AD 纤维束 importance: {fiber_importance_ad.shape}")
    print(f"  ✓ FTD 纤维束 importance: {fiber_importance_ftd.shape}")
    
    # 存储所有指标的结果
    all_results = {}
    
    # 2-4. 对每个指标进行分析
    for metric_idx, metric in enumerate(metrics, 1):
        print(f"\n{'='*60}")
        print(f"指标 [{metric_idx}/{len(metrics)}]: {metric.upper()}")
        print(f"{'='*60}")
        
        # 2. 加载连接矩阵
        data_csv = os.path.join(sample_dir, f'{metric}.csv')
        print(f"\n[2] 加载连接矩阵: {metric}.csv")
        
        if not os.path.exists(data_csv):
            print(f"  ⚠️  文件不存在，跳过: {data_csv}")
            continue
            
        df = pd.read_csv(data_csv, index_col=0)
        fiber_names = list(df.index)
        region_names = list(df.columns)
        connectivity = df.to_numpy()
        
        print(f"  ✓ 纤维束数量: {len(fiber_names)}")
        print(f"  ✓ 脑区数量: {len(region_names)}")
        print(f"  ✓ 连接矩阵: {connectivity.shape}")
        
        # 3. 映射到脑区（保存样本级数据用于箱型图）
        print(f"\n[3] 映射纤维束importance到脑区...")
        
        # 为每个样本计算region importance
        region_importances_ad_samples = []
        for fiber_imp in ad_nodes:
            reg_imp = map_fiber_importance_to_regions(fiber_imp, connectivity)
            region_importances_ad_samples.append(reg_imp)
        region_importances_ad_samples = np.array(region_importances_ad_samples)  # (n_samples, 25)
        
        region_importances_ftd_samples = []
        for fiber_imp in ftd_nodes:
            reg_imp = map_fiber_importance_to_regions(fiber_imp, connectivity)
            region_importances_ftd_samples.append(reg_imp)
        region_importances_ftd_samples = np.array(region_importances_ftd_samples)  # (n_samples, 25)
        
        # 计算平均值（用于BrainNet导出）
        region_importance_ad = np.mean(region_importances_ad_samples, axis=0)
        region_importance_ftd = np.mean(region_importances_ftd_samples, axis=0)
        
        print(f"  ✓ AD 脑区 importance: mean={region_importance_ad.mean():.4f}, max={region_importance_ad.max():.4f}")
        print(f"  ✓ FTD 脑区 importance: mean={region_importance_ftd.mean():.4f}, max={region_importance_ftd.max():.4f}")
        
        # 计算差异
        region_diff = region_importance_ftd - region_importance_ad
        print(f"\n  脑区差异 (FTD - AD): mean={region_diff.mean():.4f}, range=[{region_diff.min():.4f}, {region_diff.max():.4f}]")
        
        # 显示最具区分性的脑区
        print(f"\n  Top-10 最具区分性的脑区 (按|FTD-AD|排序):")
        top_diff = np.argsort(np.abs(region_diff))[::-1][:10]
        for i, idx in enumerate(top_diff, 1):
            print(f"    {i}. {region_names[idx]}: AD={region_importance_ad[idx]:.4f}, FTD={region_importance_ftd[idx]:.4f}, Diff={region_diff[idx]:+.4f}")
        
        # 4. 导出到BrainNet格式
        print(f"\n[4] 导出到BrainNet Viewer格式...")
        metric_output_dir = os.path.join(output_dir, metric) if sensitivity_analysis else output_dir
        os.makedirs(metric_output_dir, exist_ok=True)
        
        # FTD > AD的脑区
        print(f"\n  导出FTD受影响更严重的脑区（Top-10）...")
        ftd_dominant = region_diff.copy()
        ftd_file = export_to_brainnet_viewer(
            ftd_dominant,
            region_names,
            output_path=os.path.join(metric_output_dir, 'FTD_dominant_regions'),
            label='FTD-dominant',
            top_k=10
        )
        
        # AD > FTD的脑区
        print(f"\n  导出AD受影响更严重的脑区（Top-10）...")
        ad_dominant = -region_diff
        ad_file = export_to_brainnet_viewer(
            ad_dominant,
            region_names,
            output_path=os.path.join(metric_output_dir, 'AD_dominant_regions'),
            label='AD-dominant',
            top_k=10
        )
        
        # 保存结果（包含样本级数据）
        save_file = os.path.join(metric_output_dir, 'region_importance.npz')
        np.savez(save_file,
                 region_importance_ad=region_importance_ad,
                 region_importance_ftd=region_importance_ftd,
                 region_diff=region_diff,
                 region_names=np.array(region_names),
                 fiber_importance_ad=fiber_importance_ad,
                 fiber_importance_ftd=fiber_importance_ftd,
                 region_importances_ad_samples=region_importances_ad_samples,
                 region_importances_ftd_samples=region_importances_ftd_samples)
        print(f"  ✓ 保存: {save_file}")
        
        # 5. 生成箱型图（合并FTD和AD到一张图）
        print(f"\n[5] 生成统计箱型图...")
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互后端
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置样式
            sns.set_style("whitegrid")
            plt.rcParams['font.sans-serif'] = ['Arial']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 简写标签函数
            def get_short_name(full_name):
                """生成简短的脑区名称"""
                abbrevs = {
                    'Anterior_Commissure': 'AC',
                    'Inferior_Fronto_Occipital_Fasciculus': 'IFOF',
                    'Inferior_Longitudinal_Fasciculus': 'ILF',
                    'Arcuate_Fasciculus': 'AF',
                    'Cortico_Spinal_Tract': 'CST',
                    'Vertical_Occipital_Fasciculus': 'VOF',
                    'Uncinate_Fasciculus': 'UF',
                    'Optic_Radiation': 'OR',
                    'Frontal_Aslant_Tract': 'FAT',
                    'Middle_Longitudinal_Fasciculus': 'MLF',
                    'Middle_Cerebellar_Peduncle': 'MCP',
                    'Acoustic_Radiation': 'AR',
                    'Cingulum': 'Cin',
                    'Fornix': 'Fx'
                }
                for full, short in abbrevs.items():
                    if full in full_name:
                        suffix = ''
                        if '_L' in full_name:
                            suffix = '-L'
                        elif '_R' in full_name:
                            suffix = '-R'
                        return short + suffix
                return full_name[:15]
            
            # 5.1 合并的箱型图：FTD-dominant（正值）和AD-dominant（负值）
            # 按BrainNet实际使用的排序：FTD取最大正值，AD取最小负值
            top_ftd_idx = np.argsort(region_diff)[-10:][::-1]  # 最大的10个正值，降序
            top_ad_idx = np.argsort(region_diff)[:10]  # 最小的10个负值
            
            df_combined = []
            
            # FTD-dominant脑区（正差异）
            for idx in top_ftd_idx:
                label = get_short_name(region_names[idx])
                for j in range(len(region_importances_ad_samples)):
                    diff = region_importances_ftd_samples[j, idx] - region_importances_ad_samples[j, idx]
                    df_combined.append({'Region': label, 'Difference': diff, 'Type': 'FTD-dominant'})
            
            # AD-dominant脑区（负差异）
            for idx in top_ad_idx:
                label = get_short_name(region_names[idx])
                for j in range(len(region_importances_ad_samples)):
                    diff = region_importances_ftd_samples[j, idx] - region_importances_ad_samples[j, idx]
                    df_combined.append({'Region': label, 'Difference': diff, 'Type': 'AD-dominant'})
            
            df_combined = pd.DataFrame(df_combined)
            
            # 按差异均值排序：FTD正值从大到小在上，AD负值从大到小在下（最负的在最底）
            region_order = []
            # FTD: 差异最大的在最上面
            for idx in top_ftd_idx:
                region_order.append(get_short_name(region_names[idx]))
            # AD: 差异最小（最负）的在最下面
            for idx in top_ad_idx[::-1]:  # 倒序，最负的在最底
                region_order.append(get_short_name(region_names[idx]))
            
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            sns.boxplot(y='Region', x='Difference', data=df_combined, ax=ax, 
                       palette={'FTD-dominant': '#e74c3c', 'AD-dominant': '#3498db'},
                       hue='Type', orient='h', order=region_order)
            
            ax.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.8)
            ax.set_title(f'{metric.upper()} - Top-10 Discriminative Regions (FTD vs AD)', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Difference (FTD - AD)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Brain Region', fontsize=13, fontweight='bold')
            
            # 添加文字说明
            ax.text(0.98, 0.98, 'Positive = FTD > AD\nNegative = AD > FTD', 
                   transform=ax.transAxes, fontsize=11, verticalalignment='top',
                   horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            plt.legend(title='Dominant Group', loc='lower right', fontsize=11, title_fontsize=12)
            plt.tight_layout()
            
            boxplot_combined = os.path.join(metric_output_dir, 'boxplot_combined_top20.png')
            plt.savefig(boxplot_combined, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ 合并箱型图: {boxplot_combined}")
            
            # 保存文件路径（用于报告）
            boxplot_ftd = boxplot_combined
            boxplot_ad = boxplot_combined
            
        except Exception as e:
            print(f"  ⚠️  箱型图生成失败: {e}")
            boxplot_ftd = None
            boxplot_ad = None
        
        # 存储结果用于敏感性分析（包含AD和FTD的Top-10）
        top_ad_dominant = np.argsort(region_diff)[:10]  # AD > FTD (差异最负/最小的值)
        
        all_results[metric] = {
            'region_importance_ad': region_importance_ad,
            'region_importance_ftd': region_importance_ftd,
            'region_diff': region_diff,
            'region_names': region_names,
            'top_ftd_dominant': top_diff[:10],  # Top-10 FTD indices
            'top_ad_dominant': top_ad_dominant,  # Top-10 AD indices
            'ftd_file': ftd_file,
            'ad_file': ad_file,
            'boxplot_ftd': boxplot_ftd,
            'boxplot_ad': boxplot_ad
        }
    
    # 5. 敏感性分析：计算不同指标的重叠度
    if sensitivity_analysis and len(all_results) > 1:
        print(f"\n{'='*60}")
        print("敏感性分析：不同指标的结果一致性")
        print(f"{'='*60}")
        
        # === FTD-dominant脑区重叠分析 ===
        print("\n【FTD-dominant脑区分析】")
        overlap_ftd = np.zeros((len(metrics), len(metrics)))
        
        for i, metric1 in enumerate(metrics):
            if metric1 not in all_results:
                continue
            for j, metric2 in enumerate(metrics):
                if metric2 not in all_results:
                    continue
                    
                top1 = set(all_results[metric1]['top_ftd_dominant'])
                top2 = set(all_results[metric2]['top_ftd_dominant'])
                overlap = len(top1 & top2)
                overlap_ftd[i, j] = overlap
        
        print("\nTop-10 FTD-dominant脑区的重叠数量:")
        print(f"{'':15}", end='')
        for metric in metrics:
            if metric in all_results:
                print(f"{metric:15}", end='')
        print()
        
        for i, metric1 in enumerate(metrics):
            if metric1 not in all_results:
                continue
            print(f"{metric1:15}", end='')
            for j, metric2 in enumerate(metrics):
                if metric2 in all_results:
                    print(f"{int(overlap_ftd[i, j]):15}", end='')
            print()
        
        # === AD-dominant脑区重叠分析 ===
        print("\n【AD-dominant脑区分析】")
        overlap_ad = np.zeros((len(metrics), len(metrics)))
        
        for i, metric1 in enumerate(metrics):
            if metric1 not in all_results:
                continue
            for j, metric2 in enumerate(metrics):
                if metric2 not in all_results:
                    continue
                    
                top1 = set(all_results[metric1]['top_ad_dominant'])
                top2 = set(all_results[metric2]['top_ad_dominant'])
                overlap = len(top1 & top2)
                overlap_ad[i, j] = overlap
        
        print("\nTop-10 AD-dominant脑区的重叠数量:")
        print(f"{'':15}", end='')
        for metric in metrics:
            if metric in all_results:
                print(f"{metric:15}", end='')
        print()
        
        for i, metric1 in enumerate(metrics):
            if metric1 not in all_results:
                continue
            print(f"{metric1:15}", end='')
            for j, metric2 in enumerate(metrics):
                if metric2 in all_results:
                    print(f"{int(overlap_ad[i, j]):15}", end='')
            print()
        
        # === FTD核心脑区 ===
        # 找出在所有指标中都出现的核心脑区
        if len(all_results) >= 2:
            core_ftd = set(all_results[list(all_results.keys())[0]]['top_ftd_dominant'])
            for metric in list(all_results.keys())[1:]:
                core_ftd &= set(all_results[metric]['top_ftd_dominant'])
            
            if core_ftd:
                print(f"\nFTD核心脑区（在所有{len(all_results)}个指标中都出现，共{len(core_ftd)}个）:")
                region_names = all_results[list(all_results.keys())[0]]['region_names']
                for idx in sorted(core_ftd):
                    print(f"  • {region_names[idx]}")
            else:
                print(f"\n没有在所有指标中都出现的FTD核心脑区")
                
            # 找出高频脑区（至少在一半指标中出现）
            threshold = len(all_results) // 2
            region_counts_ftd = {}
            for metric in all_results:
                for idx in all_results[metric]['top_ftd_dominant']:
                    region_counts_ftd[idx] = region_counts_ftd.get(idx, 0) + 1
            
            frequent_ftd = [idx for idx, count in region_counts_ftd.items() if count >= threshold]
            if frequent_ftd:
                print(f"\nFTD高频脑区（至少在{threshold}个指标中出现，共{len(frequent_ftd)}个）:")
                for idx in frequent_ftd:
                    count = region_counts_ftd[idx]
                    print(f"  • {region_names[idx]} (出现{count}/{len(all_results)}次)")
        
        # === AD核心脑区 ===
        if len(all_results) >= 2:
            core_ad = set(all_results[list(all_results.keys())[0]]['top_ad_dominant'])
            for metric in list(all_results.keys())[1:]:
                core_ad &= set(all_results[metric]['top_ad_dominant'])
            
            if core_ad:
                print(f"\nAD核心脑区（在所有{len(all_results)}个指标中都出现，共{len(core_ad)}个）:")
                region_names = all_results[list(all_results.keys())[0]]['region_names']
                for idx in sorted(core_ad):
                    print(f"  • {region_names[idx]}")
            else:
                print(f"\n没有在所有指标中都出现的AD核心脑区")
                
            # 找出高频脑区（至少在一半指标中出现）
            region_counts_ad = {}
            for metric in all_results:
                for idx in all_results[metric]['top_ad_dominant']:
                    region_counts_ad[idx] = region_counts_ad.get(idx, 0) + 1
            
            frequent_ad = [idx for idx, count in region_counts_ad.items() if count >= threshold]
            if frequent_ad:
                print(f"\nAD高频脑区（至少在{threshold}个指标中出现，共{len(frequent_ad)}个）:")
                for idx in frequent_ad:
                    count = region_counts_ad[idx]
                    print(f"  • {region_names[idx]} (出现{count}/{len(all_results)}次)")
        
        # 保存敏感性分析报告
        report_file = os.path.join(output_dir, 'sensitivity_analysis_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("BrainNet脑区导出 - 敏感性分析报告\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"分析指标: {', '.join(metrics)}\n")
            f.write(f"成功分析: {', '.join(all_results.keys())}\n")
            f.write(f"样本数量: AD={n_samples}, FTD={n_samples}\n\n")
            
            # FTD-dominant脑区重叠
            f.write("="*60 + "\n")
            f.write("【FTD-dominant脑区】\n")
            f.write("="*60 + "\n\n")
            f.write("Top-10 FTD-dominant脑区重叠矩阵:\n")
            f.write(f"{'':15}")
            for metric in metrics:
                if metric in all_results:
                    f.write(f"{metric:15}")
            f.write("\n")
            
            for i, metric1 in enumerate(metrics):
                if metric1 not in all_results:
                    continue
                f.write(f"{metric1:15}")
                for j, metric2 in enumerate(metrics):
                    if metric2 in all_results:
                        f.write(f"{int(overlap_ftd[i, j]):15}")
                f.write("\n")
            
            if len(all_results) >= 2:
                if core_ftd:
                    f.write(f"\nFTD核心脑区（在所有{len(all_results)}个指标中都出现）:\n")
                    region_names = all_results[list(all_results.keys())[0]]['region_names']
                    for idx in sorted(core_ftd):
                        f.write(f"  • {region_names[idx]}\n")
                
                if frequent_ftd:
                    f.write(f"\nFTD高频脑区（至少在{threshold}个指标中出现）:\n")
                    for idx in frequent_ftd:
                        count = region_counts_ftd[idx]
                        f.write(f"  • {region_names[idx]} ({count}/{len(all_results)})\n")
            
            f.write(f"\n各指标的Top-10 FTD-dominant脑区:\n")
            for metric in all_results:
                f.write(f"\n{metric.upper()}:\n")
                region_names = all_results[metric]['region_names']
                for i, idx in enumerate(all_results[metric]['top_ftd_dominant'], 1):
                    diff = all_results[metric]['region_diff'][idx]
                    f.write(f"  {i}. {region_names[idx]}: Diff={diff:+.4f}\n")
                
                # 添加箱型图路径
                if all_results[metric].get('boxplot_ftd'):
                    f.write(f"\n  📊 箱型图: {metric}/boxplot_top10_FTD_dominant.png\n")
            
            # AD-dominant脑区重叠
            f.write("\n" + "="*60 + "\n")
            f.write("【AD-dominant脑区】\n")
            f.write("="*60 + "\n\n")
            f.write("Top-10 AD-dominant脑区重叠矩阵:\n")
            f.write(f"{'':15}")
            for metric in metrics:
                if metric in all_results:
                    f.write(f"{metric:15}")
            f.write("\n")
            
            for i, metric1 in enumerate(metrics):
                if metric1 not in all_results:
                    continue
                f.write(f"{metric1:15}")
                for j, metric2 in enumerate(metrics):
                    if metric2 in all_results:
                        f.write(f"{int(overlap_ad[i, j]):15}")
                f.write("\n")
            
            if len(all_results) >= 2:
                if core_ad:
                    f.write(f"\nAD核心脑区（在所有{len(all_results)}个指标中都出现）:\n")
                    region_names = all_results[list(all_results.keys())[0]]['region_names']
                    for idx in sorted(core_ad):
                        f.write(f"  • {region_names[idx]}\n")
                
                if frequent_ad:
                    f.write(f"\nAD高频脑区（至少在{threshold}个指标中出现）:\n")
                    for idx in frequent_ad:
                        count = region_counts_ad[idx]
                        f.write(f"  • {region_names[idx]} ({count}/{len(all_results)})\n")
            
            f.write(f"\n各指标的Top-10 AD-dominant脑区:\n")
            for metric in all_results:
                f.write(f"\n{metric.upper()}:\n")
                region_names = all_results[metric]['region_names']
                for i, idx in enumerate(all_results[metric]['top_ad_dominant'], 1):
                    diff = -all_results[metric]['region_diff'][idx]  # AD的差异是负的FTD差异
                    f.write(f"  {i}. {region_names[idx]}: Diff={diff:+.4f}\n")
                
                # 添加箱型图路径
                if all_results[metric].get('boxplot_ad'):
                    f.write(f"\n  📊 箱型图: {metric}/boxplot_top10_AD_dominant.png\n")
        
        print(f"\n✓ 敏感性分析报告已保存: {report_file}")
    
    # 6. 总结输出
    print("\n" + "="*60)
    print("✅ 脑区节点导出完成！")
    print("="*60)
    
    if sensitivity_analysis and len(all_results) > 1:
        print(f"\n📊 敏感性分析完成，共分析{len(all_results)}个指标")
        print(f"  生成目录: {output_dir}/")
        for metric in all_results:
            print(f"    • {metric}/ - 基于{metric}的结果")
        print(f"    • sensitivity_analysis_report.txt - 敏感性分析报告")
    else:
        print("\n生成的文件:")
        metric = list(all_results.keys())[0] if all_results else 'unknown'
        result_dir = os.path.join(output_dir, metric) if sensitivity_analysis else output_dir
        print(f"  📊 {result_dir}/FTD_dominant_regions.node")
        print(f"  📊 {result_dir}/AD_dominant_regions.node")
    
    print("\n💡 使用说明:")
    print("  1. 主分析使用 number_of_tracts（纤维束密度）作为连接权重")
    if sensitivity_analysis:
        print("  2. 补充材料可展示不同指标的敏感性分析结果")
        print("  3. 核心脑区（多指标一致）具有更强的证据支持")
    print("  4. 在BrainNet Viewer中加载 .node 文件可视化")
    
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='导出脑区节点到BrainNet Viewer（支持敏感性分析）')
    parser.add_argument('--model', default='best_model_fold1.pt',
                       help='模型文件路径')
    parser.add_argument('--data', default='F:\\workspace1\\dataset',
                       help='数据集根目录')
    parser.add_argument('--sample_dir', default='F:\\workspace1\\dataset\\AD\\003_S_4136_ses-2011-08-10',
                       help='样本目录（包含各种扩散指标CSV）')
    parser.add_argument('--output', default='results_explainability',
                       help='输出目录')
    parser.add_argument('--n_samples', type=int, default=50,
                       help='每类用于计算importance的样本数')
    parser.add_argument('--no-sensitivity', action='store_true',
                       help='禁用敏感性分析（只用number_of_tracts）')
    
    args = parser.parse_args()
    export_region_importance_to_brainnet(
        model_path=args.model,
        data_root=args.data,
        sample_dir=args.sample_dir,
        output_dir=args.output,
        n_samples=args.n_samples,
        sensitivity_analysis=not args.no_sensitivity
    )

"""
从深度学习模型的可解释性结果生成弦图
基于GradCAM计算的脑区重要性构建连接权重，展示TOP-K最显著的脑区连接
"""

import os
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def load_region_importance(metric, base_dir='results_explainability'):
    """加载某个指标的脑区重要性结果"""
    npz_path = os.path.join(base_dir, metric, 'region_importance.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"未找到文件: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    return {
        'region_names': list(data['region_names']),
        'region_diff': data['region_diff'],  # FTD - AD
        'region_importance_ad': data['region_importance_ad'],
        'region_importance_ftd': data['region_importance_ftd']
    }


def build_connection_matrix(region_diff):
    """
    根据脑区重要性差异构建连接矩阵
    策略：重要性差异大的脑区之间假设有更强的功能连接差异
    """
    n = len(region_diff)
    conn = np.zeros((n, n))
    
    # 使用外积构建连接强度（重要性高的区域之间连接强）
    # 这里希望连接的符号反映总体方向：正值表示 FTD > AD，负值表示 AD > FTD
    abs_diff = np.abs(region_diff)
    for i in range(n):
        for j in range(i+1, n):
            # 连接强度 = sqrt(importance_i * importance_j)
            strength = np.sqrt(abs_diff[i] * abs_diff[j])
            # 使用两区域差值的和来决定方向：如果两者均为正(sum>0)则表示 FTD>AD，
            # 若均为负(sum<0)则表示 AD>FTD；若方向相反(sum≈0)则弱化方向性
            summed = region_diff[i] + region_diff[j]
            sign = np.sign(summed) if abs(summed) > 0 else 0.0
            conn[i, j] = strength * sign
            conn[j, i] = conn[i, j]
    
    return conn


def draw_chord_diagram(region_names, conn_matrix, top_k=20, out_file='chord_model.png', 
                       title='Brain Regions Chord Diagram (Model Results)'):
    """绘制弦图"""
    n = len(region_names)
    
    # 计算节点强度（用于扇区大小）
    strength = np.sum(np.abs(conn_matrix), axis=1)
    total = np.sum(strength)
    if total <= 0:
        sector_frac = np.ones(n) / n
    else:
        sector_frac = strength / total
    
    # 构建扇区角度范围
    gap = 2 * math.pi * 0.003
    angles = []
    cur = 0.0
    for f in sector_frac:
        span = max(f * 2 * math.pi, 0.025)
        angles.append((cur, cur + span))
        cur += span + gap
    
    label_angles = [(a + b) / 2.0 for a, b in angles]
    
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.axis('off')
    
    radius = 8
    
    # 绘制扇区弧线
    colors_sector = plt.cm.Set3(np.linspace(0, 1, n))
    for idx, (s, e) in enumerate(angles):
        theta = np.linspace(s, e, 200)
        xs = radius * np.cos(theta)
        ys = radius * np.sin(theta)
        ax.plot(xs, ys, color=colors_sector[idx], linewidth=8, 
                solid_capstyle='butt', alpha=0.8)
        
        # 标签
        ang = label_angles[idx]
        # 放到更外侧以避免被弦或色条遮挡；调整到更靠近圆环的位置
        lx = 1.12 * math.cos(ang) * radius
        ly = 1.12 * math.sin(ang) * radius
        # 使标签沿半径方向（垂直于圆周）排列，并确保文字朝上可读
        ha = 'center'
        va = 'center'
        label = region_names[idx].replace('_', ' ')
        # 计算角度并确保文字沿半径朝外且不倒置
        deg = math.degrees(ang) % 360
        # 如果位于左侧半圆（90~270），将文字翻转 180° 以朝外显示
        if 90 < deg < 270:
            rot = deg + 180
        else:
            rot = deg
        # 将角度归一到 (-180, 180] 区间，Matplotlib 处理更稳定
        if rot > 180:
            rot -= 360
        ax.text(lx, ly, label, ha=ha, va=va, fontsize=13, fontweight='bold', rotation=rot, rotation_mode='anchor',
                bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=0.3))
    
    # 选择TOP-K连接（保证包含正/负两类以便展示方向差异）
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = conn_matrix[i, j]
            if abs(w) > 1e-6:
                all_edges.append((i, j, w))

    if not all_edges:
        print("警告: 没有找到有效连接")
        return

    pos_edges = sorted([e for e in all_edges if e[2] > 0], key=lambda x: abs(x[2]), reverse=True)
    neg_edges = sorted([e for e in all_edges if e[2] < 0], key=lambda x: abs(x[2]), reverse=True)

    k_pos = top_k // 2
    k_neg = top_k - k_pos

    selected = pos_edges[:k_pos] + neg_edges[:k_neg]
    # 若某类不足则用剩余最强的边补齐
    if len(selected) < top_k:
        remaining = [e for e in all_edges if e not in selected]
        remaining = sorted(remaining, key=lambda x: abs(x[2]), reverse=True)
        selected += remaining[:top_k - len(selected)]

    edges = selected
    # 为 colorbar 使用全矩阵范围（避免只基于所选边的缩放导致视觉偏差）
    maxw = np.max(np.abs(conn_matrix)) if conn_matrix.size > 0 else max(abs(w) for _, _, w in edges)
    # 颜色映射：以0为中点，负值(AD>FTD)为蓝，正值(FTD>AD)为红
    # 使用更深的蓝红配色（seismic）的发色，并基线提高透明度，使蓝色更明显
    cmap = cm.seismic
    norm = colors.Normalize(vmin=-maxw, vmax=maxw)
    
    def point_on_sector(idx, frac, r=radius - 0.5):
        s, e = angles[idx]
        ang = s + (e - s) * frac
        return (r * math.cos(ang), r * math.sin(ang))
    
    # 绘制连接带（ribbons）
    for rank, (i, j, w) in enumerate(edges):
        # 为每条边分配不同的位置以减少重叠
        frac_offset = 0.3 + 0.4 * (rank / max(len(edges) - 1, 1))
        
        p1a = point_on_sector(i, max(0.1, frac_offset - 0.15))
        p1b = point_on_sector(i, min(0.9, frac_offset + 0.15))
        p2a = point_on_sector(j, max(0.1, frac_offset - 0.15))
        p2b = point_on_sector(j, min(0.9, frac_offset + 0.15))
        
        # 贝塞尔控制点（向圆心）
        c1 = (0.4 * (p1a[0] + p2a[0]), 0.4 * (p1a[1] + p2a[1]))
        c2 = (0.4 * (p1b[0] + p2b[0]), 0.4 * (p1b[1] + p2b[1]))
        
        verts = [p1a, c1, p2a, p2b, c2, p1b, p1a]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, 
                 Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]
        path = Path(verts, codes)
        
        # 颜色根据有符号权重映射为渐变色，透明度随权重大小变化
        color = cmap(norm(w))
        # 增加透明度基线并放大随权重增长的透明度变化，使深色更明显
        if maxw > 0:
            alpha = 0.3 + 0.7 * (abs(w) / maxw)
            alpha = min(alpha, 1.0)
        else:
            alpha = 0.6
        patch = PathPatch(path, facecolor=color, edgecolor=color, 
                         alpha=alpha, linewidth=0.5)
        ax.add_patch(patch)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 添加图例（方向性）
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cmap(norm(maxw*0.6)), alpha=0.7, label='FTD > AD (positive)'),
        Patch(facecolor=cmap(norm(-maxw*0.6)), alpha=0.7, label='AD > FTD (negative)')
    ]
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=13)
    # 加粗图例文字
    for text in legend.get_texts():
        text.set_fontweight('bold')

    # 添加数值颜色条（colorbar）用于显示权重数值映射
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(np.linspace(-maxw, maxw, 100))
    # 将颜色条放到右下角，避免遮挡扇区标签；再向右微调以防止覆盖文字
    # 使用绝对轴位置 [left, bottom, width, height]
    # 将颜色条缩小并移向右下角，避免遮挡并更靠近角落
    # axes = [left, bottom, width, height]
    cax = fig.add_axes([0.97, 0.02, 0.012, 0.16])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Connection Weight (signed)', fontsize=12, fontweight='bold')
    # 将刻度和标签放在右侧，视觉更清晰
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.tick_right()
    # 加粗和放大刻度标签
    cbar.ax.tick_params(axis='y', labelsize=11)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # 适当调整边距以确保标签显示完整
    plt.subplots_adjust(left=0.03, right=0.96, top=0.95, bottom=0.02)
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ 保存弦图: {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从深度学习结果生成脑区弦图')
    parser.add_argument('--metric', default='gfa', 
                       help='扩散指标 (gfa, qa, number_of_tracts, etc.)，用all生成所有可用指标')
    parser.add_argument('--base_dir', default='results_explainability',
                       help='可解释性结果目录')
    parser.add_argument('--top_k', type=int, default=20,
                       help='显示TOP-K条连接')
    parser.add_argument('--output', default=None,
                       help='输出文件路径（如不指定则自动生成）')
    parser.add_argument('--all', action='store_true',
                       help='生成所有可用指标的弦图')
    
    args = parser.parse_args()
    
    # 定义常用指标列表
    common_metrics = ['gfa', 'qa', 'number_of_tracts', 'md', 'rd', 'curl']
    
    # 判断是否批量生成
    if args.all or args.metric == 'all':
        metrics_to_process = []
        # 检查哪些指标实际存在
        for metric in common_metrics:
            npz_path = os.path.join(args.base_dir, metric, 'region_importance.npz')
            if os.path.exists(npz_path):
                metrics_to_process.append(metric)
        
        if not metrics_to_process:
            print(f"❌ 未找到任何可用指标的可解释性结果")
            print(f"   请先运行 run_explainability.py 生成结果")
            exit(1)
        
        print(f"\n找到 {len(metrics_to_process)} 个可用指标: {', '.join(metrics_to_process)}\n")
    else:
        metrics_to_process = [args.metric]
    
    # 依次处理每个指标
    success_count = 0
    for metric in metrics_to_process:
        try:
            print(f"\n{'='*60}")
            print(f"处理指标: {metric.upper()}")
            print(f"{'='*60}")
            
            print(f"加载 {metric} 指标的脑区重要性...")
            data = load_region_importance(metric, args.base_dir)
            
            print(f"构建连接矩阵...")
            conn_matrix = build_connection_matrix(data['region_diff'])
            
            # 确定输出路径
            if args.output and len(metrics_to_process) == 1:
                output_path = args.output
            else:
                output_path = os.path.join(args.base_dir, f'chord_{metric}_top{args.top_k}.png')
            
            print(f"绘制TOP-{args.top_k}弦图...")
            title = f'Top-{args.top_k} Brain Region Connections ({metric.upper()})'
            draw_chord_diagram(data['region_names'], conn_matrix, 
                              top_k=args.top_k, out_file=output_path, title=title)
            
            print(f"✅ 完成: {output_path}")
            success_count += 1
            
        except FileNotFoundError as e:
            print(f"❌ 跳过 {metric}: {e}")
        except Exception as e:
            print(f"❌ 处理 {metric} 时出错: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ 总计成功生成 {success_count}/{len(metrics_to_process)} 个弦图")
    print(f"{'='*60}")

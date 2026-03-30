"""
弦图可视化 - Circos风格
=========================================================
生成类似论文中circleplot风格的弦图，显示脑区之间的连接关系
参考图：黑色背景、放射状标签、彩色圆周、内部彩色弦
"""

import os
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.patches as patches
from matplotlib.patches import Arc, Wedge


def load_region_importance(metric, base_dir='results_explainability'):
    """加载某个指标的脑区重要性结果"""
    npz_path = os.path.join(base_dir, metric, 'region_importance.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"未找到文件: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    return {
        'region_names': list(data['region_names']),
        'region_diff': data['region_diff'],
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


def draw_circos_chord(region_names, conn_matrix, top_k=25, out_file='chord_circos.png',
                      title='Brain Regions Connectivity'):
    """
    绘制Circos风格弦图（ribbon连接风格）
    特点：
    - 黑色背景
    - 圆形布局，圆周分段着色
    - 放射状标签（无背景框）
    - 带状连接线（ribbons）
    - 颜色条指示连接权重（红=FTD>AD，蓝=AD>FTD）
    """
    
    n = len(region_names)
    
    # 计算节点强度（用于扇区大小）
    strength = np.sum(np.abs(conn_matrix), axis=1)
    total_strength = np.sum(strength)
    if total_strength <= 0:
        sector_frac = np.ones(n) / n
    else:
        sector_frac = strength / total_strength
    
    # 构建扇区角度范围
    gap = 2 * np.pi * 0.003
    angles = []
    cur = 0.0
    for f in sector_frac:
        span = max(f * 2 * np.pi, 0.025)
        angles.append((cur, cur + span))
        cur += span + gap
    
    label_angles = [(a + b) / 2.0 for a, b in angles]
    
    # 创建图形（黑色背景）
    fig = plt.figure(figsize=(14, 14), facecolor='black')
    ax = fig.add_subplot(111, facecolor='black')
    ax.set_xlim(-11.5, 11.5)
    ax.set_ylim(-11.5, 11.5)
    ax.axis('off')
    
    # 半径设置
    radius = 8
    
    # ===== 绘制圆周扇区 =====
    # 使用Set3配色方案
    cmap_sector = plt.cm.Set3
    sector_colors = [cmap_sector(i / n) for i in range(n)]
    
    for idx, ((s, e), color) in enumerate(zip(angles, sector_colors)):
        theta = np.linspace(s, e, 100)
        xs = radius * np.cos(theta)
        ys = radius * np.sin(theta)
        ax.plot(xs, ys, color=color, linewidth=8, solid_capstyle='butt', alpha=0.8)
    
    # ===== 绘制标签（无背景框）=====
    for idx, (ang, name) in enumerate(zip(label_angles, region_names)):
        lx = 1.12 * radius * np.cos(ang)
        ly = 1.12 * radius * np.sin(ang)
        
        # 调整文本对齐
        ha = 'center'
        va = 'center'
        
        # 旋转角度（沿径向排列）
        deg = np.degrees(ang) % 360
        if 90 < deg < 270:
            rot = deg + 180
        else:
            rot = deg
        if rot > 180:
            rot -= 360
        
        # 简化长标签
        display_name = name.replace('_', ' ')
        
        # 无背景框，直接绘制黄色文字（清晰可见）
        ax.text(lx, ly, display_name, fontsize=11, fontweight='bold',
                ha=ha, va=va, rotation=rot, rotation_mode='anchor',
                color='#FFD700')  # 金黄色，清晰可见
    
    # ===== 选择TOP-K连接 =====
    # 分别选择正负连接
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = conn_matrix[i, j]
            if abs(w) > 1e-6:
                all_edges.append((i, j, w))
    
    if not all_edges:
        print("警告: 没有找到有效连接")
        return
    
    # 分别处理正负连接
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
    
    # ===== 绘制连接线（简单曲线） =====
    # 使用seismic colormap：蓝色(AD>FTD)<->红色(FTD>AD)
    maxw = np.max(np.abs(conn_matrix)) if conn_matrix.size > 0 else max(abs(w) for _, _, w in edges)
    
    cmap_chord = plt.cm.seismic
    norm = colors.Normalize(vmin=-maxw, vmax=maxw)
    
    # 绘制连接线
    for i, j, w in edges:
        # 获取两个节点的角度
        angle_i = (angles[i][0] + angles[i][1]) / 2
        angle_j = (angles[j][0] + angles[j][1]) / 2
        
        # 圆周上的起点和终点
        x1 = radius * np.cos(angle_i)
        y1 = radius * np.sin(angle_i)
        x2 = radius * np.cos(angle_j)
        y2 = radius * np.sin(angle_j)
        
        # 绘制二次贝塞尔曲线（控制点在圆心）
        t = np.linspace(0, 1, 100)
        xs = (1-t)**2 * x1 + 2*(1-t)*t * 0 + t**2 * x2
        ys = (1-t)**2 * y1 + 2*(1-t)*t * 0 + t**2 * y2
        
        # 颜色根据权重
        color = cmap_chord(norm(w))
        
        # 线宽根据权重大小
        linewidth = 1 + 2.5 * (abs(w) / maxw) if maxw > 0 else 1.5
        
        # 透明度根据权重
        alpha = 0.4 + 0.6 * (abs(w) / maxw) if maxw > 0 else 0.7
        
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=1)
    
    # ===== 添加图例 =====
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cmap_chord(norm(maxw*0.6)), alpha=0.7, label='FTD > AD (positive)'),
        Patch(facecolor=cmap_chord(norm(-maxw*0.6)), alpha=0.7, label='AD > FTD (negative)')
    ]
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
                       framealpha=0.95, fancybox=True, edgecolor='#FFD700', 
                       frameon=True, facecolor='black')
    for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_color('#FFD700')  # 金黄色，高对比度
    
    # 颜色条
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    sm = plt.cm.ScalarMappable(cmap=cmap_chord, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Connection Weight', fontsize=11, fontweight='bold', color='#FFD700')
    cbar.ax.tick_params(colors='#FFD700', labelsize=9)
    for spine in cax.spines.values():
        spine.set_color('#FFD700')
        spine.set_linewidth(1.5)
    
    # 保存
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"✓ 保存Circos弦图: {out_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='生成Circos风格弦图')
    parser.add_argument('--metric', type=str, default='number_of_tracts',
                        help='指标名称（number_of_tracts, qa, md, rd, gfa等）')
    parser.add_argument('--base_dir', type=str, default='results_explainability',
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
            print("错误：未找到任何可用的指标数据")
            return
        
        print(f"将为以下指标生成Circos弦图: {metrics_to_process}")
        
        for metric in metrics_to_process:
            print(f"\n{'='*60}")
            print(f"处理指标: {metric}")
            print(f"{'='*60}")
            
            try:
                data = load_region_importance(metric, args.base_dir)
                conn_matrix = build_connection_matrix(data['region_diff'])
                
                out_file = os.path.join(args.base_dir, f'{metric}_chord_circos.png')
                draw_circos_chord(
                    data['region_names'],
                    conn_matrix,
                    top_k=args.top_k,
                    out_file=out_file,
                    title=f'Brain Regions Connectivity ({metric})'
                )
            except FileNotFoundError as e:
                print(f"跳过 {metric}: {e}")
    else:
        print("=" * 60)
        print(f"生成 {args.metric} 的Circos弦图")
        print("=" * 60)
        
        try:
            data = load_region_importance(args.metric, args.base_dir)
            conn_matrix = build_connection_matrix(data['region_diff'])
            
            if args.output:
                out_file = args.output
            else:
                out_file = os.path.join(args.base_dir, f'{args.metric}_chord_circos.png')
            
            draw_circos_chord(
                data['region_names'],
                conn_matrix,
                top_k=args.top_k,
                out_file=out_file,
                title=f'Brain Regions Connectivity ({args.metric})'
            )
            print(f"\n✓ Circos弦图生成完成: {out_file}")
        except FileNotFoundError as e:
            print(f"错误: {e}")


if __name__ == '__main__':
    main()

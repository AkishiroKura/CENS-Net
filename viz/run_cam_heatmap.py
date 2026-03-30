"""
快速生成 CAM 热力图可视化
=========================
运行此脚本自动生成类似附件图片的 CAM 热力图
"""

import torch
from models import DualPathModel
from dataloader import T2RDataset
from cam_heatmap_vis import plot_cam_heatmaps
import pandas as pd
import os

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "best_model_fold1.pt"
DATA_ROOT = "F:\\workspace1\\dataset"
SAMPLE_CSV = "F:\\workspace1\\dataset\\AD\\003_S_4136_ses-2011-08-10\\curl.csv"
OUTPUT_DIR = "results_explainability"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 可视化参数
N_SAMPLES = None  # 使用所有样本（设为 None 表示全部，或设为具体数字如 4）
CMAP = 'viridis'  # 色图：'viridis', 'plasma', 'inferno', 'magma', 'cividis'

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    
    # 加载数据
    print("\n[1/4] Loading dataset...")
    dataset = T2RDataset(DATA_ROOT)
    print(f"  Loaded {len(dataset)} samples")
    
    # 加载模型
    print("\n[2/4] Loading model...")
    model = DualPathModel(
        gnn_hidden=32,
        global_hidden=32,
        global_out=16,
        num_classes=2,
        dropout=0.5
    )
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"  Loaded from {MODEL_PATH}")
    
    # 读取节点名称（可选）
    node_names = None
    if os.path.exists(SAMPLE_CSV):
        try:
            df = pd.read_csv(SAMPLE_CSV, index_col=0)
            node_names = list(df.columns)  # 使用列名作为节点名
            print(f"\n[3/4] Loaded {len(node_names)} fiber tract names from CSV")
        except:
            print("\n[3/4] Could not load node names, using default")
    else:
        print("\n[3/4] No CSV file found, using default fiber tract names")
    
    # 生成 CAM 热力图
    print(f"\n[4/4] Generating CAM heatmaps...")
    print("="*60)
    
    # 生成 CAM 热力图
    print("\n📊 Generating CAM Heatmap (Fiber Tracts)")
    fig, ad_cams, ftd_cams = plot_cam_heatmaps(
        model, dataset, DEVICE,
        output_dir=OUTPUT_DIR,
        n_samples=N_SAMPLES,
        cmap=CMAP
    )
    
    print("\n" + "="*60)
    print("✅ CAM heatmap visualization completed!")
    print("="*60)
    print(f"\n生成的文件:")
    print(f"  📊 {OUTPUT_DIR}/cam_heatmap_comparison.png")
    print(f"     - AD: {ad_cams.shape[0]} subjects")
    print(f"     - FTD: {ftd_cams.shape[0]} subjects")
    print(f"     - Fiber Tracts: {ad_cams.shape[1]}")
    
    print(f"\n💡 提示:")
    print(f"  - 修改 N_SAMPLES 可以改变每类显示的样本数 (None=全部)")
    print(f"  - 修改 CMAP 可以改变配色方案")
    print(f"    可选: 'viridis', 'plasma', 'inferno', 'magma', 'cividis'")

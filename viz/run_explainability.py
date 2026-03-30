"""
运行可解释性分析
"""
import torch
from dataloader import T2RDataset
from models import DualPathModel
from explainability import run_analysis

# ============================================================
# 配置
# ============================================================
DATA_ROOT = "F:\\workspace1\\dataset"
MODEL_PATH = "best_model_fold1.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = "results_explainability"

# 用于读取节点名称的样本 CSV 路径
SAMPLE_CSV = "F:\\workspace1\\dataset\\AD\\003_S_4136_ses-2011-08-10\\curl.csv"

# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    
    # 加载数据
    dataset = T2RDataset(DATA_ROOT)
    
    # 加载模型
    sample = dataset[0]
    model = DualPathModel(
        gnn_hidden=32,
        global_hidden=32,
        global_out=16,
        num_classes=2,
        dropout=0.5
    )
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # 兼容两种保存格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded model from {MODEL_PATH}")
    
    # 运行分析（传入 CSV 路径以获取节点名称）
    results = run_analysis(model, dataset, DEVICE, OUTPUT_DIR, 
                          n_samples=50, sample_csv_path=SAMPLE_CSV)
    
    print("\n" + "="*50)
    print("✅ Explainability analysis completed!")
    print("="*50)
    print(f"\n生成的文件:")
    print(f"  📊 可视化图表:")
    print(f"    - {OUTPUT_DIR}/feature_by_category.png")
    print(f"  💾 分析结果:")
    print(f"    - {OUTPUT_DIR}/results.npz")
    
    print(f"\n💡 后续步骤:")
    print(f"  1. 生成 CAM 热力图:")
    print(f"     python run_cam_heatmap.py")
    print(f"  2. 导出脑区到 BrainNet Viewer:")
    print(f"     python export_brainnet_regions.py")
    
    print(f"\n输出目录: {OUTPUT_DIR}/")

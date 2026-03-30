"""
计算全局统计量（用于归一化）
对 number_of_tracts 先做 log 变换，然后所有指标计算 mean 和 std
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

DATA_ROOT = r"F:\workspace1\dataset"
METRICS = [
    "number_of_tracts",
    "curl",
    "qa",
    "md",
    "rd",
    "gfa",
    "intersect_ratio"
]

def compute_global_stats():
    """计算全局统计量"""
    # 存储每个 metric 的所有有效值
    stats = {m: {"count": 0, "sum": 0.0, "sum_sq": 0.0} for m in METRICS}
    
    groups = ["AD", "FTD"]
    
    print("Scanning dataset to compute global statistics...")
    print(f"Note: number_of_tracts will be log-transformed first\n")
    
    for group in groups:
        group_dir = os.path.join(DATA_ROOT, group)
        if not os.path.exists(group_dir):
            continue
            
        subjects = [s for s in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, s))]
        
        for subject in tqdm(subjects, desc=f"Processing {group}"):
            subject_dir = os.path.join(group_dir, subject)
            
            for m in METRICS:
                csv_path = os.path.join(subject_dir, f"{m}.csv")
                if not os.path.exists(csv_path):
                    continue
                
                try:
                    df = pd.read_csv(csv_path, index_col=0)
                    vals = df.values.flatten()
                    
                    # 过滤 NaN
                    vals = vals[~np.isnan(vals)]
                    
                    if len(vals) == 0:
                        continue
                    
                    # 对 number_of_tracts 先做 log 变换
                    if m == "number_of_tracts":
                        vals = np.log(vals + 1)  # log(x+1) 避免 log(0)
                    
                    stats[m]["count"] += len(vals)
                    stats[m]["sum"] += np.sum(vals)
                    stats[m]["sum_sq"] += np.sum(vals ** 2)
                        
                except Exception as e:
                    print(f"\nError reading {csv_path}: {e}")

    print("\n" + "="*70)
    print("Global Statistics (for Z-Score normalization)")
    print("="*70)
    print(f"{'Metric':<30} | {'Mean':<15} | {'Std':<15}")
    print("-"*70)
    
    results = {}
    
    for m in METRICS:
        count = stats[m]["count"]
        if count > 0:
            mean_val = stats[m]["sum"] / count
            # Variance = E[X^2] - (E[X])^2
            var_val = (stats[m]["sum_sq"] / count) - (mean_val ** 2)
            std_val = np.sqrt(max(var_val, 0))  # 防止负方差
            
            # 防止除以 0
            if std_val < 1e-8:
                std_val = 1.0
        else:
            mean_val = 0.0
            std_val = 1.0
            
        results[m] = {"mean": mean_val, "std": std_val}
        
        note = " (after log transform)" if m == "number_of_tracts" else ""
        print(f"{m:<30} | {mean_val:<15.6f} | {std_val:<15.6f}{note}")
    
    print("="*70)
    print("\nCopy these values into dataloader.py:")
    print("\nGLOBAL_STATS = {")
    for m in METRICS:
        print(f'    "{m}": {{"mean": {results[m]["mean"]:.8f}, "std": {results[m]["std"]:.8f}}},')
    print("}")
    
    return results

if __name__ == "__main__":
    compute_global_stats()

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import scipy.stats

METRICS = [
    "number_of_tracts",
    "curl",
    "qa",
    "md",
    "rd",
    "gfa",
    "intersect_ratio"
]

LABEL_MAP = {
    "AD": 0,
    "FTD": 1
}

# 全局统计量（用于 Z-Score 归一化）
# number_of_tracts 的统计量是对 log(x+1) 后的值计算的
GLOBAL_STATS = {
    "number_of_tracts": {"mean": 2.00519606, "std": 3.54350441},
    "curl": {"mean": 0.46338965, "std": 0.93852629},
    "qa": {"mean": 0.08175016, "std": 0.13772361},
    "md": {"mean": 0.24673592, "std": 0.42768002},
    "rd": {"mean": 0.19471116, "std": 0.34654813},
    "gfa": {"mean": 0.02446091, "std": 0.04060925},
    "intersect_ratio": {"mean": 0.03222963, "std": 0.09834234},
}


def extract_global_features(arrs, keep_fibers, keep_regions, arrs_normalized):
    """
    提取全局统计特征
    返回一个向量，包含：
    1. 节点级统计（14维）：每个指标×2种统计量的全图平均（mean, std）
    2. 边级统计（14维）：每个指标的全图边中位数和偏度
    3. 拓扑特征（5维）：节点数、边数、图密度等
    总共 33 维
    """
    M = len(METRICS)
    Fk = keep_fibers.size
    Rk = keep_regions.size
    
    global_feat = []
    
    # 1. 节点级统计（14维）：对归一化后的数据计算mean和std
    for m in range(M):
        metric_data = arrs_normalized[m, keep_fibers][:, keep_regions]
        # Mean of means (沿region维度)
        means = np.nanmean(metric_data, axis=1)
        global_feat.append(np.nanmean(means))
        # Std of means (沿fiber维度)
        global_feat.append(np.nanstd(means))
    
    # 2. 边级统计（14维）：原始数据，使用中位数和偏度
    for m in range(M):
        edge_vals = []
        for i in range(Fk):
            for j in range(Rk):
                val = arrs[m, keep_fibers[i], keep_regions[j]]
                if not np.isnan(val):
                    # 归一化处理
                    if METRICS[m] == "number_of_tracts":
                        val = np.log(val + 1)
                    mean_val = GLOBAL_STATS[METRICS[m]]["mean"]
                    std_val = GLOBAL_STATS[METRICS[m]]["std"]
                    val = (val - mean_val) / std_val
                    edge_vals.append(val)
        
        if len(edge_vals) > 0:
            global_feat.append(np.median(edge_vals))  # 边中位数
            global_feat.append(scipy.stats.skew(edge_vals, nan_policy='omit'))  # 边偏度
        else:
            global_feat.extend([0.0, 0.0])
    
    # 3. 拓扑特征（5维）
    num_nodes = Fk + Rk
    num_edges = Fk * Rk * 2  # 双向边
    graph_density = num_edges / (num_nodes * num_nodes) if num_nodes > 0 else 0
    avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
    fiber_ratio = Fk / num_nodes if num_nodes > 0 else 0
    
    global_feat.extend([
        num_nodes / 100.0,      # 归一化节点数
        num_edges / 10000.0,    # 归一化边数
        graph_density,
        avg_degree / 100.0,     # 归一化平均度
        fiber_ratio
    ])
    
    return np.array(global_feat, dtype=np.float32)


def load_subject_graph(subject_dir, label):
    dfs = []
    for m in METRICS:
        path = os.path.join(subject_dir, f"{m}.csv")
        df = pd.read_csv(path, index_col=0)
        dfs.append(df)

    num_fibers = dfs[0].shape[0]
    num_regions = dfs[0].shape[1]

    # 将所有 metric 数据堆成 (M, F, R) 的数组，便于处理 NaN
    arrs = np.stack([df.to_numpy() for df in dfs], axis=0)  # (M, F, R)
    nan_mask = np.isnan(arrs)

    # 判定哪些 fiber/region 是完全缺失（整行/整列全 NaN）
    fiber_allnan = np.all(nan_mask, axis=(0, 2))   # shape (F,)
    region_allnan = np.all(nan_mask, axis=(0, 1))  # shape (R,)

    keep_fibers = np.where(~fiber_allnan)[0]
    keep_regions = np.where(~region_allnan)[0]

    if keep_fibers.size == 0 or keep_regions.size == 0:
        # 该样本信息不足，返回一个空/最小图（上层可以选择跳过）
        print(f"Sample {subject_dir}: all fibers or all regions are missing -> returning minimal graph")
        x = torch.zeros((1, len(METRICS)), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(METRICS)), dtype=torch.float32)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(label, dtype=torch.long))

    # 对原始数据进行归一化处理（在聚合之前）
    # 对 number_of_tracts (index=0) 先做 log 变换，然后所有指标做 Z-Score
    M = len(METRICS)
    arrs_normalized = arrs.copy()
    for m in range(M):
        metric_name = METRICS[m]
        metric_data = arrs_normalized[m, keep_fibers][:, keep_regions]
        
        # 对 number_of_tracts 先做 log 变换
        if metric_name == "number_of_tracts":
            metric_data = np.log(metric_data + 1)
        
        # Z-Score 归一化
        mean_val = GLOBAL_STATS[metric_name]["mean"]
        std_val = GLOBAL_STATS[metric_name]["std"]
        metric_data = (metric_data - mean_val) / std_val
        
        arrs_normalized[m, keep_fibers[:, None], keep_regions] = metric_data
    
    # 计算 node 特征：对保留的维度计算 mean, std 两种统计量
    # 特征维度从 7 变为 14 (7 metrics * 2 stats)
    fiber_features = []
    region_features = []
    
    for m in range(M):
        metric_data = arrs_normalized[m, keep_fibers][:, keep_regions]
        
        # 纤维节点特征：对每条纤维在所有脑区上聚合
        fiber_mean = np.nanmean(metric_data, axis=1)  # (F_kept,)
        fiber_std = np.nanstd(metric_data, axis=1)
        fiber_features.extend([fiber_mean, fiber_std])
        
        # 脑区节点特征：对每个脑区在所有纤维上聚合
        region_mean = np.nanmean(metric_data, axis=0)  # (R_kept,)
        region_std = np.nanstd(metric_data, axis=0)
        region_features.extend([region_mean, region_std])
    
    # 转置并堆叠：每个节点现在有 14 个特征 (7 metrics * 2 stats)
    fiber_x = np.stack(fiber_features, axis=1)  # (F_kept, 14)
    region_x = np.stack(region_features, axis=1)  # (R_kept, 14)

    # 构建双向边，只在保留的节点间建立
    # 边特征也需要归一化处理
    Fk = keep_fibers.size
    Rk = keep_regions.size
    edge_index_list = []
    edge_attr_list = []
    for i in range(Fk):
        for j in range(Rk):
            vals = []
            for m in range(M):
                val = arrs[m, keep_fibers[i], keep_regions[j]]
                
                # 对 number_of_tracts 先做 log 变换
                if METRICS[m] == "number_of_tracts" and not np.isnan(val):
                    val = np.log(val + 1)
                
                # Z-Score 归一化
                if not np.isnan(val):
                    mean_val = GLOBAL_STATS[METRICS[m]]["mean"]
                    std_val = GLOBAL_STATS[METRICS[m]]["std"]
                    val = (val - mean_val) / std_val
                
                vals.append(val)
            
            edge_index_list.append([i, Fk + j])
            edge_index_list.append([Fk + j, i])
            edge_attr_list.append(vals)
            edge_attr_list.append(vals)

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).T if len(edge_index_list) > 0 else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32) if len(edge_attr_list) > 0 else torch.zeros((0, M), dtype=torch.float32)

    # 合并节点特征并填充剩余 NaN（这里用 0 填充）
    x_np = np.concatenate([fiber_x, region_x], axis=0)
    x_np = np.nan_to_num(x_np, nan=0.0)
    edge_attr = torch.nan_to_num(edge_attr, nan=0.0)

    # 打印调试信息
    removed_f = int(fiber_allnan.sum())
    removed_r = int(region_allnan.sum())
    if removed_f > 0 or removed_r > 0:
        print(f"Sample {subject_dir}: removed {removed_f} all-NaN fibers, {removed_r} all-NaN regions")

    # 提取全局特征
    global_feat = extract_global_features(arrs, keep_fibers, keep_regions, arrs_normalized)
    
    x = torch.tensor(x_np, dtype=torch.float32)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(label, dtype=torch.long),
        global_feat=torch.tensor(global_feat, dtype=torch.float32)
    )



class T2RDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir = DATA_ROOT
        DATA_ROOT/
          ├── AD/
          └── FTD/
        """
        self.samples = []

        for group in ["AD", "FTD"]:
            group_dir = os.path.join(root_dir, group)
            label = LABEL_MAP[group]

            for subject in os.listdir(group_dir):
                subject_dir = os.path.join(group_dir, subject)
                if not os.path.isdir(subject_dir):
                    continue

                self.samples.append((subject_dir, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subject_dir, label = self.samples[idx]
        return load_subject_graph(subject_dir, label)


if __name__ == "__main__":
    dataset = T2RDataset("F:\\workspace1\\dataset")
    print(len(dataset))
    print(dataset[0])
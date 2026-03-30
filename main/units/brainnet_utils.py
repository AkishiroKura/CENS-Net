"""
将图数据转换为邻接矩阵（用于 BrainNetCNN）
"""
import torch
import numpy as np
from torch_geometric.data import Data


def graph_to_adjacency_matrix(data, max_nodes=148):
    """
    将 PyG Data 对象转换为邻接矩阵
    
    Args:
        data: PyG Data 对象
        max_nodes: 最大节点数（用于填充）
    
    Returns:
        adj_matrix: (num_channels, max_nodes, max_nodes) 邻接矩阵
    """
    num_nodes = data.x.size(0)
    edge_index = data.edge_index
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
    
    # 简化版本：只使用边特征的第一个维度（通常是纤维束数量或连接强度）
    # 避免编码过多信息导致数据泄露
    if edge_attr is not None:
        # 只使用第一个边特征（纤维束数量）
        num_channels = 1
    else:
        num_channels = 1
    
    adj_matrix = torch.zeros(num_channels, max_nodes, max_nodes)
    
    # 填充邻接矩阵
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        tgt = edge_index[1, i].item()
        
        if src >= max_nodes or tgt >= max_nodes:
            continue
        
        # 只使用边特征的第一个维度（纤维束计数），归一化到 [0, 1]
        if edge_attr is not None:
            # 使用 log 归一化避免极端值
            edge_value = edge_attr[i, 0].item()
            edge_value = torch.log1p(torch.tensor(edge_value)).item()  # log(1 + x)
            adj_matrix[0, src, tgt] = edge_value
        else:
            # 如果没有边特征，使用二值化（0/1）
            adj_matrix[0, src, tgt] = 1.0
    
    return adj_matrix


def convert_dataset_to_adjacency(dataset, max_nodes=148):
    """
    将整个数据集转换为邻接矩阵格式
    
    Args:
        dataset: PyG 数据集
        max_nodes: 最大节点数
    
    Returns:
        list of (adj_matrix, label, global_feat)
    """
    converted_data = []
    for data in dataset:
        adj_matrix = graph_to_adjacency_matrix(data, max_nodes)
        label = data.y
        global_feat = data.global_feat if hasattr(data, 'global_feat') else None
        converted_data.append({
            'adj_matrix': adj_matrix,
            'label': label,
            'global_feat': global_feat
        })
    return converted_data


class BrainNetDataLoader:
    """
    自定义 DataLoader 用于 BrainNetCNN
    """
    def __init__(self, dataset, batch_size=8, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_adj = []
            batch_labels = []
            
            for idx in batch_indices:
                item = self.dataset[idx]
                batch_adj.append(item['adj_matrix'])
                batch_labels.append(item['label'])
            
            # Stack
            batch_adj = torch.stack(batch_adj, dim=0)  # (batch, channels, nodes, nodes)
            batch_labels = torch.stack(batch_labels, dim=0)  # (batch,)
            
            yield batch_adj, batch_labels
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

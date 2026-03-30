"""
ContrastPool 辅助模块
实现 Contrastive Graph Pooling for Explainable Classification of Brain Networks
论文: https://arxiv.org/abs/2307.11133
原始实现: https://github.com/AngusMonroe/ContrastPool

适配 PyTorch Geometric 的实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import SAGEConv, global_mean_pool, global_add_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch


class MultiHeadAttention(nn.Module):
    """多头自注意力层"""
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        
        assert hid_dim % n_heads == 0
        
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)
        
    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = self.dropout(torch.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        
        return x, attention.squeeze()


class ContrastiveEncoderLayer(nn.Module):
    """对比编码层：学习类别差异性表示"""
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device, feat_dim, learnable_q=False):
        super().__init__()
        self.learnable_q = learnable_q
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.dropout = nn.Dropout(dropout)
        
        if self.learnable_q:
            self.q = nn.Parameter(torch.ones([pf_dim, feat_dim, hid_dim]))
            nn.init.xavier_uniform_(self.q)
    
    def forward(self, src, src_mask=None):
        if self.learnable_q:
            _src, _ = self.self_attention(self.q, src, src, src_mask)
        else:
            _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        return src


class DenseGraphSageLayer(nn.Module):
    """稠密版本的 GraphSage 层（用于池化后的稠密邻接矩阵）"""
    def __init__(self, in_feat, out_feat, residual=False, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        self.residual = residual and (in_feat == out_feat)
        
        self.W = nn.Linear(in_feat, out_feat, bias=True)
        nn.init.xavier_uniform_(self.W.weight, gain=nn.init.calculate_gain('relu'))
        
        if self.use_bn:
            self.bn = nn.BatchNorm1d(out_feat)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, num_nodes, in_feat)
            adj: (batch, num_nodes, num_nodes)
        Returns:
            (batch, num_nodes, out_feat)
        """
        h_in = x
        
        # Aggregate neighborhood features
        h_k_N = torch.bmm(adj, x)  # (batch, num_nodes, in_feat)
        h_k = self.W(h_k_N)
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        
        if self.residual:
            h_k = h_in + h_k
        
        if self.use_bn:
            # BatchNorm expects (batch, features, seq_len)
            h_k = self.bn(h_k.permute(0, 2, 1)).permute(0, 2, 1)
        
        return h_k


class DiffPoolLayer(nn.Module):
    """
    可微分图池化层 (DiffPool)
    用于将图粗化（coarsening）到更小的节点集合
    """
    def __init__(self, in_feat, assign_dim, out_feat, link_pred=True):
        super().__init__()
        self.link_pred = link_pred
        
        # 特征变换
        self.feat_gc = DenseGraphSageLayer(in_feat, out_feat)
        
        # 分配矩阵生成
        self.pool_gc = DenseGraphSageLayer(in_feat, assign_dim)
        
        self.log = {}
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, num_nodes, in_feat)
            adj: (batch, num_nodes, num_nodes)
        Returns:
            adj_new: (batch, assign_dim, assign_dim)
            x_new: (batch, assign_dim, out_feat)
        """
        # 计算软分配矩阵 S
        s = self.pool_gc(x, adj)  # (batch, num_nodes, assign_dim)
        s = F.softmax(s, dim=-1)  # 沿着分配维度 softmax
        
        # 新的特征
        x_feat = self.feat_gc(x, adj)  # (batch, num_nodes, out_feat)
        
        # 池化：X_new = S^T * X
        x_new = torch.bmm(s.transpose(1, 2), x_feat)  # (batch, assign_dim, out_feat)
        
        # 新的邻接矩阵：A_new = S^T * A * S
        adj_new = torch.bmm(torch.bmm(s.transpose(1, 2), adj), s)  # (batch, assign_dim, assign_dim)
        
        # 链接预测损失
        if self.link_pred:
            link_pred_loss = torch.norm(adj - torch.bmm(s, s.transpose(1, 2)), dim=(1, 2))
            link_pred_loss = link_pred_loss / (adj.size(1) * adj.size(2))
            self.log['LinkPredLoss'] = link_pred_loss.mean()
        
        # 熵损失（鼓励稀疏分配）
        entropy_loss = (-s * torch.log(s + 1e-10)).sum(dim=-1).mean()
        self.log['EntropyLoss'] = entropy_loss
        
        return adj_new, x_new


class ContrastPoolLayer(nn.Module):
    """
    ContrastPool 核心层
    结合对比学习和可微分池化
    """
    def __init__(self, in_feat, assign_dim, out_feat, max_node_num, dropout=0.1):
        super().__init__()
        
        self.assign_dim = assign_dim
        self.max_node_num = max_node_num
        
        # 特征变换
        self.feat_gc = DenseGraphSageLayer(in_feat, out_feat)
        
        # 分配矩阵参数
        self.weight = nn.Parameter(torch.Tensor(max_node_num, assign_dim))
        self.bias = nn.Parameter(torch.Tensor(1, assign_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
        self.log = {}
    
    def forward(self, x, adj, contrast_adj=None, diff_h=None):
        """
        Args:
            x: (batch, num_nodes, in_feat)
            adj: (batch, num_nodes, num_nodes)
            contrast_adj: (num_nodes, num_nodes) 对比邻接矩阵（可选）
            diff_h: (num_nodes, in_feat) 对比节点特征（可选）
        Returns:
            adj_new, x_new
        """
        batch_size, num_nodes, _ = x.shape
        device = x.device
        
        # 应用对比特征调制（如果有）
        if diff_h is not None:
            # 使用对比节点特征增强
            diff_h_expanded = diff_h[:num_nodes].unsqueeze(0).expand(batch_size, -1, -1)
            x = x + diff_h_expanded
        
        # 计算分配矩阵
        # S = softmax(X @ W + b)
        # 使用对比邻接矩阵调制（如果有）
        weight_active = self.weight[:num_nodes, :]
        if contrast_adj is not None:
            # 使用对比邻接矩阵加权
            contrast_adj_active = contrast_adj[:num_nodes, :]
            weight_modulated = weight_active * (1 + contrast_adj_active)
        else:
            weight_modulated = weight_active
        
        s = torch.matmul(x.mean(dim=-1, keepdim=True).squeeze(-1).unsqueeze(-1), 
                        weight_modulated.unsqueeze(0).unsqueeze(0))
        s = s.squeeze(2) + self.bias
        s = F.softmax(s, dim=-1)  # (batch, num_nodes, assign_dim)
        
        # 特征变换
        x_feat = self.feat_gc(x, adj)
        
        # 池化
        x_new = torch.bmm(s.transpose(1, 2), x_feat)
        adj_new = torch.bmm(torch.bmm(s.transpose(1, 2), adj), s)
        
        # 正则化损失
        link_pred_loss = torch.norm(adj - torch.bmm(s, s.transpose(1, 2)), dim=(1, 2))
        link_pred_loss = link_pred_loss / (adj.size(1) * adj.size(2))
        self.log['LinkPredLoss'] = link_pred_loss.mean()
        
        entropy_loss = (-s * torch.log(s + 1e-10)).sum(dim=-1).mean()
        self.log['EntropyLoss'] = entropy_loss
        
        return adj_new, x_new


def get_contrast_features(dataset, labels, device, num_classes=2):
    """
    计算类别对比特征
    通过统计不同类别的平均邻接矩阵和节点特征的差异
    
    Args:
        dataset: list of PyG Data objects
        labels: numpy array of labels
        device: torch device
        num_classes: number of classes
    
    Returns:
        adj_dict: dict of mean adjacency matrices per class
        nodes_dict: dict of mean node features per class
    """
    adj_dict = {}
    nodes_dict = {}
    
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        
        # 收集该类的邻接矩阵和节点特征
        adj_list = []
        nodes_list = []
        
        for idx in class_indices:
            data = dataset[idx]
            num_nodes = data.x.size(0)
            
            # 转换为稠密邻接矩阵
            adj = to_dense_adj(data.edge_index, max_num_nodes=num_nodes).squeeze(0)
            adj_list.append(adj)
            nodes_list.append(data.x)
        
        if len(adj_list) > 0:
            # 对于不同大小的图，需要填充到相同大小
            max_nodes = max([a.size(0) for a in adj_list])
            
            padded_adjs = []
            padded_nodes = []
            
            for adj, nodes in zip(adj_list, nodes_list):
                n = adj.size(0)
                # 填充邻接矩阵
                padded_adj = F.pad(adj, (0, max_nodes - n, 0, max_nodes - n))
                padded_adjs.append(padded_adj)
                # 填充节点特征
                padded_node = F.pad(nodes, (0, 0, 0, max_nodes - n))
                padded_nodes.append(padded_node)
            
            adj_dict[c] = torch.stack(padded_adjs).to(device)  # (num_samples, max_nodes, max_nodes)
            nodes_dict[c] = torch.stack(padded_nodes).to(device)  # (num_samples, max_nodes, feat_dim)
    
    return adj_dict, nodes_dict


def compute_contrast_summary(adj_dict, nodes_dict, device):
    """
    计算类别之间的对比信息
    
    Returns:
        contrast_adj: 邻接矩阵的类间方差（高方差=高区分性）
        diff_h: 节点特征的类间方差
    """
    # 计算每个类的均值
    adj_means = []
    nodes_means = []
    
    for c in sorted(adj_dict.keys()):
        adj_means.append(adj_dict[c].mean(dim=0))
        nodes_means.append(nodes_dict[c].mean(dim=0))
    
    # 计算类间标准差（作为对比信号）
    adj_stack = torch.stack(adj_means)
    nodes_stack = torch.stack(nodes_means)
    
    contrast_adj = torch.std(adj_stack, dim=0)  # (max_nodes, max_nodes)
    diff_h = torch.std(nodes_stack, dim=0)  # (max_nodes, feat_dim)
    
    return contrast_adj, diff_h

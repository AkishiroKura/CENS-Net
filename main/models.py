"""
模型定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GlobalAttention, GATConv, global_mean_pool, GCNConv, GINConv, SAGEConv
import numpy as np


class PureGINEBaseline(nn.Module):
    """
    纯净的 GINE Baseline - 三层，无任何创新点
    用于消融实验的基准模型
    """
    def __init__(self, in_dim=14, edge_dim=7, hidden_dim=32, num_classes=2, dropout=0.5):
        super().__init__()
        
        self.dropout = dropout

        # edge feature → hidden
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GINE layers: 标准三层，无残差
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=hidden_dim,
            train_eps=True
        )

        self.conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=hidden_dim,
            train_eps=True
        )

        self.conv3 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=hidden_dim,
            train_eps=True
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch
        )

        # 边特征编码
        edge_attr = self.edge_mlp(edge_attr)
        
        # 第一层卷积（无残差）
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层卷积（无残差）
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第三层卷积（无残差）
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 简单的全局平均池化（非注意力）
        x = global_mean_pool(x, batch)

        # 分类
        return self.classifier(x)


class GINEBaseline(nn.Module):
    """
    GINE + 创新点（三层残差连接 + 注意力池化）
    这是你的改进版本
    """
    def __init__(self, in_dim=14, edge_dim=7, hidden_dim=32, num_classes=2, dropout=0.5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # edge feature → hidden
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GINE layers: 三层堆叠 with residual connections
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=hidden_dim,
            train_eps=True
        )
        
        # 第一层的输入维度适配（用于残差连接）
        self.adapt1 = nn.Linear(in_dim, hidden_dim)

        self.conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=hidden_dim,
            train_eps=True
        )

        self.conv3 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=hidden_dim,
            train_eps=True
        )

        # Attention Pooling (可解释性关键)
        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch
        )

        # 边特征编码
        edge_attr = self.edge_mlp(edge_attr)
        
        # 第一层卷积 + 残差连接
        identity = self.adapt1(x)  # 适配维度
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x + identity)  # 残差连接
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层卷积 + 残差连接
        identity = x
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x + identity)  # 残差连接
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第三层卷积 + 残差连接
        identity = x
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x + identity)  # 残差连接
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 注意力池化 (graph-level pooling with attention)
        x = self.pool(x, batch)

        # 分类
        return self.classifier(x)


class FeatureGraphEncoder(nn.Module):
    """
    特征图编码器：将全局特征构建成一个图，用GNN处理特征间的依赖关系
    
    特征分组（8个节点）：
    0: 节点统计-均值 (7维)
    1: 节点统计-标准差 (7维)
    2: 边统计-中位数 (7维)
    3: 边统计-偏度 (7维)
    4: 拓扑特征-节点数 (1维)
    5: 拓扑特征-边数 (1维)
    6: 拓扑特征-密度 (2维)
    7: 拓扑特征-比例 (1维)
    """
    def __init__(self, global_feat_dim=33, hidden_dim=32, out_dim=16, dropout=0.3):
        super().__init__()
        
        # 特征分组投影（将原始特征分成8组，每组投影到统一维度）
        self.node_mean_proj = nn.Linear(7, hidden_dim)
        self.node_std_proj = nn.Linear(7, hidden_dim)
        self.edge_median_proj = nn.Linear(7, hidden_dim)
        self.edge_skew_proj = nn.Linear(7, hidden_dim)
        self.topo1_proj = nn.Linear(1, hidden_dim)  # 节点数
        self.topo2_proj = nn.Linear(1, hidden_dim)  # 边数
        self.topo3_proj = nn.Linear(2, hidden_dim)  # 密度+平均度
        self.topo4_proj = nn.Linear(1, hidden_dim)  # fiber比例
        
        # 定义特征图的边（哪些特征组应该交互）
        # 完全图：所有特征组都相互连接
        self.edge_index = self._build_feature_graph_edges()
        
        # GAT卷积层（使用注意力机制学习特征间的重要性）
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def _build_feature_graph_edges(self):
        """
        构建特征图的边结构
        策略：构建一个语义相关的图
        - 节点统计组内全连接 (0-1)
        - 边统计组内全连接 (2-3)
        - 拓扑特征组内全连接 (4-5-6-7)
        - 跨组连接：节点统计 ↔ 边统计 ↔ 拓扑特征
        """
        edges = []
        
        # 节点统计组内连接 (0-1)
        edges.append([0, 1])
        edges.append([1, 0])
        
        # 边统计组内连接 (2-3)
        edges.append([2, 3])
        edges.append([3, 2])
        
        # 拓扑特征组内连接 (4-5-6-7)
        for i in range(4, 8):
            for j in range(i+1, 8):
                edges.append([i, j])
                edges.append([j, i])
        
        # 跨组连接：节点统计 ↔ 边统计
        for i in range(2):
            for j in range(2, 4):
                edges.append([i, j])
                edges.append([j, i])
        
        # 跨组连接：边统计 ↔ 拓扑特征
        for i in range(2, 4):
            for j in range(4, 8):
                edges.append([i, j])
                edges.append([j, i])
        
        # 跨组连接：节点统计 ↔ 拓扑特征
        for i in range(2):
            for j in range(4, 8):
                edges.append([i, j])
                edges.append([j, i])
        
        return torch.tensor(edges, dtype=torch.long).T  # (2, num_edges)
    
    def forward(self, global_feat, batch_size):
        """
        Args:
            global_feat: (batch_size, 33) 全局特征向量
            batch_size: batch大小
        Returns:
            (batch_size, out_dim) 编码后的全局特征表示
        """
        device = global_feat.device
        
        # 将特征分组并投影（14+14+5=33维）
        node_mean = self.node_mean_proj(global_feat[:, 0:7])      # (bs, hidden)
        node_std = self.node_std_proj(global_feat[:, 7:14])
        edge_median = self.edge_median_proj(global_feat[:, 14:21])
        edge_skew = self.edge_skew_proj(global_feat[:, 21:28])
        topo1 = self.topo1_proj(global_feat[:, 28:29])
        topo2 = self.topo2_proj(global_feat[:, 29:30])
        topo3 = self.topo3_proj(global_feat[:, 30:32])
        topo4 = self.topo4_proj(global_feat[:, 32:33])
        
        # 堆叠成特征图节点 (batch_size, 8, hidden_dim)
        x = torch.stack([node_mean, node_std, edge_median, edge_skew,
                        topo1, topo2, topo3, topo4], dim=1)
        
        # 展平成 (batch_size * 8, hidden_dim) 用于GNN
        x = x.view(-1, x.size(-1))
        
        # 构建batch索引（每个样本有8个节点）
        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(8)
        
        # 扩展edge_index以适配batch
        edge_index = self.edge_index.to(device)
        edge_index_batch = []
        for i in range(batch_size):
            edge_index_batch.append(edge_index + i * 8)
        edge_index_batch = torch.cat(edge_index_batch, dim=1)
        
        # GAT卷积处理特征图
        x = self.gat1(x, edge_index_batch)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.gat2(x, edge_index_batch)
        x = F.relu(x)
        
        # 池化：对每个样本的8个节点取平均
        x = global_mean_pool(x, batch_indices)  # (batch_size, hidden_dim)
        
        # 输出层
        return self.fc(x)


class DualPathModel(nn.Module):
    """
    双路径模型：
    - GNN路径：处理脑网络图结构（局部）
    - 特征图路径：处理全局统计特征
    """
    def __init__(self, gnn_hidden=32, global_hidden=32, global_out=16, num_classes=2, dropout=0.5):
        super().__init__()
        
        # 路径1：GNN处理脑网络
        self.gnn_path = GINEBaseline(
            in_dim=14, 
            edge_dim=7, 
            hidden_dim=gnn_hidden, 
            num_classes=num_classes,
            dropout=dropout
        )
        # 修改GNN的分类器为恒等映射（只输出特征）
        self.gnn_path.classifier = nn.Identity()
        
        # 路径2：特征图编码器处理全局特征
        self.global_path = FeatureGraphEncoder(
            global_feat_dim=33,
            hidden_dim=global_hidden,
            out_dim=global_out,
            dropout=dropout * 0.6  # 全局路径用较小的dropout
        )
        
        # 融合分类器
        self.fusion_classifier = nn.Sequential(
            nn.Linear(gnn_hidden + global_out, gnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_hidden, num_classes)
        )
        
    def forward(self, data):
        """
        Args:
            data: PyG Data对象，必须包含 global_feat 属性
        """
        # 路径1：GNN处理图
        gnn_out = self.gnn_path(data)  # (batch_size, gnn_hidden)
        
        # 路径2：特征图编码器处理全局特征
        # 注意：DataLoader 会把 global_feat 拼接成 (batch_size * 40,)
        # 需要 reshape 回 (batch_size, 40)
        batch_size = data.y.size(0)
        global_feat = data.global_feat.view(batch_size, -1)  # reshape to (batch_size, 40)
        global_out = self.global_path(global_feat, batch_size)  # (batch_size, global_out)
        
        # 融合两路特征
        combined = torch.cat([gnn_out, global_out], dim=1)
        
        # 最终分类
        return self.fusion_classifier(combined)


class GCNBaseline(nn.Module):
    """
    Baseline: Graph Convolutional Network (GCN)
    Standard GCN implementation without edge features or global features.
    """
    def __init__(self, in_dim=14, hidden_dim=32, num_classes=2, dropout=0.5):
        super().__init__()
        
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.dropout = dropout
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global Mean Pooling
        x = global_mean_pool(x, batch)
        
        return self.classifier(x)


class GINBaseline(nn.Module):
    """
    Baseline: Graph Isomorphism Network (GIN)
    Standard GIN implementation - no edge features, no residual, no attention pooling.
    """
    def __init__(self, in_dim=14, hidden_dim=32, num_classes=2, dropout=0.5):
        super().__init__()
        
        self.dropout = dropout

        # GIN layers: 标准 GIN，使用 MLP 作为节点更新函数
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            train_eps=True
        )

        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            train_eps=True
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 第一层卷积（无残差）
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层卷积（无残差）
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 简单的全局平均池化（非注意力池化）
        x = global_mean_pool(x, batch)

        # 分类
        return self.classifier(x)


class ContrastPoolNet(nn.Module):
    """
    ContrastPool: Contrastive Graph Pooling for Explainable Classification of Brain Networks
    论文: Xu et al., IEEE TMI 2024 (https://arxiv.org/abs/2307.11133)
    
    核心思想:
    1. 使用对比注意力机制学习类别间差异性特征
    2. 可微分图池化 (DiffPool) 进行图粗化
    3. GraphSage 作为消息传递骨干
    
    适配 PyTorch Geometric 的实现
    """
    def __init__(self, in_dim=14, hidden_dim=32, num_classes=2, 
                 max_num_nodes=200, pool_ratio=0.5, n_layers=2, dropout=0.3,
                 lambda1=0.01):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_classes = num_classes
        self.max_num_nodes = max_num_nodes
        self.pool_ratio = pool_ratio
        self.lambda1 = lambda1  # 注意力熵正则化系数
        self.dropout = dropout
        
        # 输入嵌入
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        
        # GraphSage 层（池化前）
        self.gc_before_pool = nn.ModuleList()
        self.gc_before_pool.append(self._make_sage_layer(hidden_dim, hidden_dim, dropout))
        for _ in range(n_layers - 2):
            self.gc_before_pool.append(self._make_sage_layer(hidden_dim, hidden_dim, dropout))
        self.gc_before_pool.append(self._make_sage_layer(hidden_dim, hidden_dim, dropout))
        
        # 池化层
        self.assign_dim = int(max_num_nodes * pool_ratio)
        
        # 分配矩阵生成层：将节点特征投影到分配空间
        # 输入: (batch, num_nodes, hidden_dim), 输出: (batch, num_nodes, assign_dim)
        self.assign_layer = nn.Linear(hidden_dim, self.assign_dim)
        
        # 池化后的 GraphSage 层
        self.gc_after_pool = nn.ModuleList()
        for _ in range(n_layers):
            self.gc_after_pool.append(self._make_dense_sage_layer(hidden_dim, hidden_dim))
        
        # 分类器
        self.pred_layer = nn.Linear(hidden_dim, num_classes)
        
        # 对比学习相关
        self.contrast_adj = None
        self.diff_h = None
        self.attn_loss = None
        
        # 对比编码器
        self.encoder1 = None
        self.encoder2 = None
        
    def _make_sage_layer(self, in_dim, out_dim, dropout):
        """创建 GraphSage 层（使用 PyG 的 SAGEConv）"""
        from torch_geometric.nn import SAGEConv
        return nn.ModuleDict({
            'conv': SAGEConv(in_dim, out_dim),
            'bn': nn.BatchNorm1d(out_dim),
            'dropout': nn.Dropout(dropout)
        })
    
    def _make_dense_sage_layer(self, in_dim, out_dim):
        """创建稠密版本的 GraphSage 层"""
        return nn.ModuleDict({
            'linear': nn.Linear(in_dim, out_dim),
            'bn': nn.BatchNorm1d(out_dim)
        })
    
    def init_contrast_encoders(self, node_num, feat_dim, device):
        """初始化对比编码器"""
        self.node_num_contrast = node_num
        self.feat_dim_contrast = feat_dim
        
        # 简化版对比编码器：使用线性变换代替复杂的注意力机制
        # 邻接矩阵编码器: (node_num, node_num) -> (node_num, node_num)
        self.adj_encoder = nn.Sequential(
            nn.Linear(node_num, node_num),
            nn.ReLU(),
            nn.Linear(node_num, node_num)
        ).to(device)
        
        # 节点特征编码器: (node_num, feat_dim) -> (node_num, feat_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        ).to(device)
    
    def cal_contrast(self, dataset, labels, device, num_classes=2):
        """计算类别对比特征"""
        from contrast_utils import get_contrast_features
        
        self.adj_dict, self.nodes_dict = get_contrast_features(
            dataset, labels, device, num_classes
        )
        
        if len(self.adj_dict) > 0:
            # 获取最大节点数和特征维度
            max_nodes = max([self.adj_dict[c].size(1) for c in self.adj_dict.keys()])
            feat_dim = list(self.nodes_dict.values())[0].size(-1)
            
            # 初始化对比编码器
            self.init_contrast_encoders(max_nodes, feat_dim, device)
    
    def cal_contrast_adj(self, device):
        """计算对比邻接矩阵和对比节点特征"""
        if self.adj_dict is None or len(self.adj_dict) == 0:
            return
        
        adj_list = []
        nodes_list = []
        
        for c in sorted(self.adj_dict.keys()):
            # 编码邻接矩阵: (max_nodes, max_nodes)
            adj = self.adj_dict[c].mean(dim=0)  # (max_nodes, max_nodes)
            adj_encoded = self.adj_encoder(adj)  # (max_nodes, max_nodes)
            adj_list.append(adj_encoded)
            
            # 编码节点特征: (max_nodes, feat_dim)
            nodes = self.nodes_dict[c].mean(dim=0)  # (max_nodes, feat_dim)
            nodes_encoded = self.node_encoder(nodes)  # (max_nodes, feat_dim)
            nodes_list.append(nodes_encoded)
        
        # 计算类间标准差作为对比信号
        self.contrast_adj = torch.std(torch.stack(adj_list), dim=0)  # (max_nodes, max_nodes)
        self.diff_h = torch.std(torch.stack(nodes_list), dim=0)  # (max_nodes, feat_dim)
        
        # 计算注意力熵损失（归一化后计算）
        contrast_adj_norm = F.softmax(self.contrast_adj.view(-1), dim=0)
        self.attn_loss = self._cal_attn_loss(contrast_adj_norm)
    
    def _cal_attn_loss(self, attn):
        """计算注意力熵损失"""
        # 确保数值稳定性
        attn_clamped = torch.clamp(attn, min=1e-10, max=1.0)
        # 确保 attn 已归一化
        attn_sum = attn_clamped.sum()
        if attn_sum > 0:
            attn_normalized = attn_clamped / attn_sum
        else:
            attn_normalized = attn_clamped
        entropy = (-attn_normalized * torch.log(attn_normalized + 1e-10)).sum()
        return entropy
    
    def forward(self, data):
        """
        Args:
            data: PyG Data/Batch 对象
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        device = x.device
        
        # 嵌入
        h = self.embedding_h(x)
        
        # 池化前的 GNN 层
        for gc_layer in self.gc_before_pool:
            h = gc_layer['conv'](h, edge_index)
            h = gc_layer['bn'](h)
            h = F.relu(h)
            h = gc_layer['dropout'](h)
        
        # 转换为稠密格式进行池化
        from torch_geometric.utils import to_dense_batch, to_dense_adj
        
        h_dense, mask = to_dense_batch(h, batch)  # (batch_size, max_nodes, hidden)
        adj_dense = to_dense_adj(edge_index, batch)  # (batch_size, max_nodes, max_nodes)
        
        batch_size, num_nodes, _ = h_dense.shape
        
        # 对比池化
        # 如果有对比信息，使用它来调制分配
        if self.contrast_adj is not None:
            # 截取到当前图大小
            contrast_adj_size = min(num_nodes, self.contrast_adj.size(0))
            assign_size = min(self.assign_dim, self.contrast_adj.size(1) if self.contrast_adj.dim() > 1 else self.assign_dim)
            
            if self.contrast_adj.dim() == 2:
                contrast_adj_active = self.contrast_adj[:contrast_adj_size, :assign_size]
            else:
                contrast_adj_active = None
        else:
            contrast_adj_active = None
        
        # 计算分配矩阵 S
        # 使用 assign_layer 将节点特征投影到分配空间
        # h_dense: (batch, num_nodes, hidden) -> s: (batch, num_nodes, assign_dim)
        s = self.assign_layer(h_dense)  # (batch, num_nodes, assign_dim)
        
        # 应用 mask：将填充节点的分配分数设为负无穷
        s = s.masked_fill(~mask.unsqueeze(-1).expand_as(s), float('-inf'))
        
        # Softmax 沿着节点维度，使得每个聚类的分配权重和为1
        s = F.softmax(s, dim=1)  # (batch, num_nodes, assign_dim)
        
        # 应用对比调制（如果维度匹配）
        if contrast_adj_active is not None and contrast_adj_active.size(0) == num_nodes and contrast_adj_active.size(1) == s.size(2):
            # 使用对比邻接矩阵调制分配权重
            s = s * (1 + 0.1 * contrast_adj_active.unsqueeze(0).expand(batch_size, -1, -1))
            # 重新归一化
            s = s / (s.sum(dim=1, keepdim=True) + 1e-10)
        
        # 池化
        h_pooled = torch.bmm(s.transpose(1, 2), h_dense)  # (batch, assign_dim, hidden)
        adj_pooled = torch.bmm(torch.bmm(s.transpose(1, 2), adj_dense), s)  # (batch, assign_dim, assign_dim)
        
        # 池化后的 GNN 层（稠密版本）
        h_out = h_pooled
        for gc_layer in self.gc_after_pool:
            # 邻接矩阵聚合
            h_agg = torch.bmm(adj_pooled, h_out)
            h_out = gc_layer['linear'](h_agg)
            # BatchNorm (需要转换维度)
            bs, n, d = h_out.shape
            h_out = gc_layer['bn'](h_out.view(-1, d)).view(bs, n, d)
            h_out = F.relu(h_out)
            h_out = F.dropout(h_out, p=self.dropout, training=self.training)
        
        # 图级 readout (sum pooling)
        graph_repr = h_out.sum(dim=1)  # (batch, hidden)
        
        # 分类
        return self.pred_layer(graph_repr)
    
    def loss(self, pred, label):
        """自定义损失函数（包含正则化项）"""
        criterion = nn.CrossEntropyLoss()
        ce_loss = criterion(pred, label)
        
        # 添加注意力熵正则化
        if self.attn_loss is not None:
            total_loss = ce_loss + self.lambda1 * self.attn_loss
        else:
            total_loss = ce_loss
        
        return total_loss


class BrainNetCNN(nn.Module):
    """
    BrainNetCNN: Kawahara et al., 2017
    CNN架构专门用于脑连接矩阵分析
    包含 E2E (Edge-to-Edge), E2N (Edge-to-Node), N2G (Node-to-Graph) 层
    """
    def __init__(self, num_nodes=148, num_edge_features=1, num_classes=2, dropout=0.5):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.dropout = dropout
        
        # E2E Layer: Edge-to-Edge 卷积（在邻接矩阵上）
        # 输入: (batch, channels, num_nodes, num_nodes)
        self.e2e_conv1 = nn.Conv2d(num_edge_features, 32, kernel_size=1)
        self.e2e_conv2 = nn.Conv2d(32, 64, kernel_size=1)
        
        # E2N Layer: Edge-to-Node（行列聚合）
        # 通过全局池化实现
        
        # N2G Layer: Node-to-Graph
        self.n2g_fc1 = nn.Linear(num_nodes * 64, 128)
        self.n2g_fc2 = nn.Linear(128, 64)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, adj_matrix):
        """
        Args:
            adj_matrix: (batch, channels, num_nodes, num_nodes) 邻接矩阵
        """
        # E2E: Edge-to-Edge convolution
        x = self.e2e_conv1(adj_matrix)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.e2e_conv2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x: (batch, 64, num_nodes, num_nodes)
        
        # E2N: Edge-to-Node (行列求和)
        # 沿着最后一个维度求和，得到节点级表示
        x = torch.sum(x, dim=-1)  # (batch, 64, num_nodes)
        
        # Flatten for N2G
        x = x.view(x.size(0), -1)  # (batch, 64 * num_nodes)
        
        # N2G: Node-to-Graph
        x = self.n2g_fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.n2g_fc2(x)
        x = F.relu(x)
        
        # Classifier
        return self.classifier(x)


class GATBaseline(nn.Module):
    """
    Graph Attention Network (GAT) Baseline
    三层 GAT 网络，用于图分类基准对比
    """
    def __init__(self, in_dim=14, hidden_dim=32, num_classes=2, dropout=0.5, heads=4):
        super().__init__()
        
        self.dropout = dropout
        self.heads = heads

        # GAT layers: 三层堆叠
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)  # 最后一层 heads=1

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 第一层卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层卷积
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第三层卷积
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 全局平均池化
        x = global_mean_pool(x, batch)

        # 分类
        return self.classifier(x)

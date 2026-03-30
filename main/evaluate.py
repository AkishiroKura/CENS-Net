"""
验证函数
"""
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate(model, loader, device, criterion):
    """
    验证函数：计算损失、精度、F1、精确率、召回率
    并返回预测值、真实值和预测概率
    
    Args:
        model: 模型
        loader: 验证数据加载器
        device: 设备
        criterion: 损失函数
        
    Returns:
        avg_loss, acc, f1, precision, recall, all_preds, all_labels, all_probs
    """
    if loader is None:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), [], [], []
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_graphs = 0
    all_preds = []
    all_labels = []
    all_probs = []  # 预测概率（用于ROC曲线）
    
    with torch.no_grad():
        for batch in loader:
            if not hasattr(batch, "y") or batch.y is None:
                continue
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            
            # 跳过异常 loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            total_loss += loss.item() * batch.num_graphs
            preds = out.argmax(dim=1)
            total_correct += (preds == batch.y).sum().item()
            total_graphs += batch.num_graphs
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch.y.cpu().numpy().tolist())
            
            # 获取预测概率（Class 1的概率）
            probs = torch.softmax(out, dim=1)[:, 1]  # 取Class 1的概率
            all_probs.extend(probs.cpu().numpy().tolist())
    
    avg_loss = total_loss / total_graphs if total_graphs > 0 else float("nan")
    acc = total_correct / total_graphs if total_graphs > 0 else float("nan")
    
    # 计算 F1、精确率、召回率（用加权平均处理数据不均衡）
    if len(all_preds) > 0 and len(all_labels) > 0:
        try:
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        except:
            f1 = precision = recall = float("nan")
    else:
        f1 = precision = recall = float("nan")
    
    model.train()
    return avg_loss, acc, f1, precision, recall, all_preds, all_labels, all_probs


def extract_features(model, loader, device):
    """
    提取模型的特征表示（从池化层后的特征）
    
    Args:
        model: 模型
        loader: 数据加载器
        device: 设备
        
    Returns:
        features: (N, hidden_dim) 的特征矩阵
        labels: (N,) 的标签数组
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            if not hasattr(batch, "y") or batch.y is None:
                continue
            batch = batch.to(device)
            
            # 根据模型类型提取特征
            if hasattr(model, 'conv') and not hasattr(model, 'conv1'):
                # GINEBaseline 模型（老版本，有 conv 但没有 conv1）
                x, edge_index, edge_attr, batch_indices = (
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch
                )
                
                edge_attr = model.edge_mlp(edge_attr)
                x = model.conv(x, edge_index, edge_attr)
                x = torch.relu(x)
                x = model.conv2(x, edge_index, edge_attr)
                x = torch.relu(x)
                
                from torch_geometric.nn import global_mean_pool
                features = global_mean_pool(x, batch_indices)
                
            elif hasattr(model, 'gnn_path'):
                # DualPathModel：双路径模型，使用 GNN 路径的输出
                gnn_out = model.gnn_path(batch)
                features = gnn_out  # GNN路径已经是池化后的特征
                
            elif hasattr(model, 'pool'):
                # GINELightModel, GINEGPSModel 或其他带 pool 属性的模型
                x, edge_index, edge_attr, batch_indices = (
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch
                )
                
                num_nodes = x.size(0)
                edge_attr_encoded = model.edge_mlp(edge_attr)
                
                # Layer 1
                if hasattr(model, 'adapt1'):
                    identity = model.adapt1(x)
                    x_local = model.conv1(x, edge_index, edge_attr_encoded)
                    x_local = torch.relu(x_local + identity)
                    
                    # 根据不同模型类型使用不同的归一化
                    if hasattr(model, 'norm1_local'):  # GINEGPSModel
                        x_local = model.norm1_local(x_local)
                    elif hasattr(model, 'norm1'):  # GINELightModel
                        x_local = model.norm1(x_local)
                    
                    # 添加结构编码（仅 GINEGPSModel）
                    if hasattr(model, 'compute_structural_encoding'):
                        struct_encoding = model.compute_structural_encoding(edge_index, num_nodes)
                        x_local = x_local + struct_encoding
                    
                    # Transformer（仅 GINEGPSModel）
                    if hasattr(model, 'graph_transformer_layer'):
                        x_global = model.graph_transformer_layer(x_local, batch_indices, model.transformer1, model.norm1_global)
                        x1 = x_local + x_global
                    else:
                        x1 = x_local
                    
                    # Layer 2
                    x_local = model.conv2(x1, edge_index, edge_attr_encoded)
                    
                    # 根据模型类型应用残差和归一化
                    if hasattr(model, 'norm2_local'):  # GINEGPSModel
                        x_local = torch.relu(x_local)
                        x_local = model.norm2_local(x_local)
                    elif hasattr(model, 'norm2'):  # GINELightModel
                        x_local = torch.relu(x_local + x1)  # 残差
                        x_local = model.norm2(x_local)
                    else:
                        x_local = torch.relu(x_local)
                    
                    # Transformer（仅 GINEGPSModel）
                    if hasattr(model, 'graph_transformer_layer'):
                        x_global = model.graph_transformer_layer(x_local, batch_indices, model.transformer2, model.norm2_global)
                        x2 = x_local + x_global
                    else:
                        x2 = x_local
                    
                    # JK
                    if hasattr(model, 'jk_proj'):
                        x_jk = torch.cat([x1, x2], dim=-1)
                        x_jk = model.jk_proj(x_jk)
                        features = model.pool(x_jk, batch_indices)
                    else:
                        features = model.pool(x2, batch_indices)
                else:
                    # 简单情况：直接用 pool
                    x = model.conv1(x, edge_index, edge_attr_encoded)
                    x = torch.relu(x)
                    x = model.conv2(x, edge_index, edge_attr_encoded)
                    x = torch.relu(x)
                    features = model.pool(x, batch_indices)
            else:
                # 默认：直接前向传播到分类器前
                raise AttributeError("Model type not recognized for feature extraction")
            
            all_features.extend(features.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy().tolist())
    
    import numpy as np
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    return all_features, all_labels

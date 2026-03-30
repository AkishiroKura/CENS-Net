"""
ContrastPool 专用训练函数
"""
import torch
import torch.nn as nn
import numpy as np
import csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def train_contrastpool(model, train_loader, val_loader, device, criterion, optimizer,
                       save_path='best_model.pt', log_path='training_log.csv',
                       results_path='best_results.txt', fold_num=1, preds_save_path=None,
                       epochs=100, patience=20):
    """
    ContrastPool 模型的训练函数
    
    Args:
        model: ContrastPoolNet 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 计算设备
        criterion: 基础损失函数（如 CrossEntropyLoss with weights）
        optimizer: 优化器
        save_path: 模型保存路径
        log_path: 训练日志路径
        results_path: 最佳结果保存路径
        fold_num: 当前折编号
        preds_save_path: 预测结果保存路径
        epochs: 训练轮数
        patience: 早停耐心值
    """
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 训练日志
    log_data = []
    
    print(f"Training ContrastPool for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 更新对比邻接矩阵
            if hasattr(model, 'adj_dict') and model.adj_dict is not None:
                model.cal_contrast_adj(device)
            
            # 前向传播
            out = model(batch)
            
            # 使用模型的自定义损失（包含正则化项）
            base_loss = criterion(out, batch.y)
            
            # 添加 ContrastPool 的正则化损失
            if model.attn_loss is not None:
                loss = base_loss + model.lambda1 * model.attn_loss
            else:
                loss = base_loss
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # 记录预测
            preds = out.argmax(dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
        
        # 计算训练指标
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # 验证
        val_loss, val_acc, val_f1, val_probs, val_preds, val_labels = evaluate_contrastpool(
            model, val_loader, device, criterion
        )
        
        # 记录日志
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1
        }
        log_data.append(log_entry)
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # 基于 F1 分数保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存模型
            torch.save(model.state_dict(), save_path)
            
            # 保存预测结果
            if preds_save_path is not None:
                np.savez(preds_save_path,
                        preds=val_preds,
                        labels=val_labels,
                        probs=val_probs)
            
            # 保存结果
            with open(results_path, 'w') as f:
                f.write(f"Fold {fold_num} Best Results\n")
                f.write("="*40 + "\n")
                f.write(f"Best Epoch: {epoch + 1}\n")
                f.write(f"Val Loss: {val_loss:.4f}\n")
                f.write(f"Val Accuracy: {val_acc:.4f}\n")
                f.write(f"Val F1 Score: {val_f1:.4f}\n")
                
                # 额外指标
                val_prec = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
                val_rec = recall_score(val_labels, val_preds, average='weighted')
                f.write(f"Val Precision: {val_prec:.4f}\n")
                f.write(f"Val Recall: {val_rec:.4f}\n")
                
                try:
                    val_auc = roc_auc_score(val_labels, val_probs)
                    f.write(f"Val AUC: {val_auc:.4f}\n")
                except:
                    f.write("Val AUC: N/A\n")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # 保存训练日志
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_data[0].keys())
        writer.writeheader()
        writer.writerows(log_data)
    
    print(f"Training complete. Best Val F1: {best_val_f1:.4f}")
    return best_val_f1, best_val_loss


def evaluate_contrastpool(model, loader, device, criterion):
    """
    评估 ContrastPool 模型
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            out = model(batch)
            loss = criterion(out, batch.y)
            
            total_loss += loss.item()
            
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, acc, f1, np.array(all_probs), np.array(all_preds), np.array(all_labels)

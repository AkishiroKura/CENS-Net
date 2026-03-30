"""
BrainNetCNN 专用训练函数
"""
import torch
import csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np


def train_brainnetcnn(model, train_loader, val_loader, device, criterion, optimizer, 
                      num_epochs=100, save_path="best_model.pt", log_path="training_log.csv", 
                      results_path="best_results.txt", fold_num=None, preds_save_path=None):
    """
    训练 BrainNetCNN 模型
    """
    best_val_loss = float('inf')
    best_epoch = 0
    best_metrics = {}
    best_preds = None
    best_labels = None
    best_probs = None
    
    # 创建 CSV 日志
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_f1', 'val_precision', 'val_recall'])
    
    for epoch in range(num_epochs):
        try:
            # Training
            model.train()
            train_loss_accum = 0.0
            train_count = 0
            
            for adj_matrix, labels in train_loader:
                adj_matrix = adj_matrix.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                out = model(adj_matrix)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                
                train_loss_accum += loss.item() * labels.size(0)
                train_count += labels.size(0)
            
            train_loss = train_loss_accum / train_count if train_count > 0 else 0.0
            
            # Validation
            if val_loader is not None:
                model.eval()
                val_loss_accum = 0.0
                val_count = 0
                all_preds = []
                all_labels = []
                all_probs = []
                
                with torch.no_grad():
                    for adj_matrix, labels in val_loader:
                        adj_matrix = adj_matrix.to(device)
                        labels = labels.to(device)
                        
                        out = model(adj_matrix)
                        loss = criterion(out, labels)
                        
                        val_loss_accum += loss.item() * labels.size(0)
                        val_count += labels.size(0)
                        
                        probs = torch.softmax(out, dim=1)
                        preds = out.argmax(dim=1)
                        
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs[:, 1].cpu().numpy())
                
                val_loss = val_loss_accum / val_count if val_count > 0 else 0.0
                
                # 计算指标
                val_acc = accuracy_score(all_labels, all_preds)
                val_f1 = f1_score(all_labels, all_preds, average='weighted')
                val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
                val_recall = recall_score(all_labels, all_preds, average='weighted')
                
                # 记录日志
                with open(log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch+1, train_loss, val_loss, val_acc, val_f1, val_precision, val_recall])
                
                print(f"Epoch {epoch+1:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    best_metrics = {
                        'epoch': best_epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_f1': val_f1,
                        'val_precision': val_precision,
                        'val_recall': val_recall
                    }
                    best_preds = all_preds
                    best_labels = all_labels
                    best_probs = all_probs
                    torch.save(model.state_dict(), save_path)
                    print(f"  Saved best model (epoch {epoch+1}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f})")
                    
                    # 保存预测结果
                    try:
                        if fold_num is not None and best_preds is not None and best_labels is not None:
                            save_name = preds_save_path if preds_save_path else f"best_preds_fold{fold_num}.npz"
                            np.savez_compressed(save_name, preds=np.array(best_preds), 
                                              labels=np.array(best_labels), 
                                              probs=np.array(best_probs) if best_probs is not None else None)
                            print(f"  Saved predictions to {save_name}")
                    except Exception as e:
                        print(f"  Warning: failed to save preds npz: {e}")
        
        except Exception as e:
            print(f"Epoch {epoch:02d} interrupted by error: {e}")
            break
    
    # 保存最佳结果
    with open(results_path, "w") as f:
        f.write("=== Best Model Results ===\n")
        f.write(f"Epoch: {best_metrics.get('epoch', 'N/A')}\n")
        train_loss = best_metrics.get('train_loss', 'N/A')
        f.write(f"Train Loss: {train_loss if isinstance(train_loss, str) else f'{train_loss:.4f}'}\n")
        val_loss = best_metrics.get('val_loss', 'N/A')
        f.write(f"Val Loss: {val_loss if isinstance(val_loss, str) else f'{val_loss:.4f}'}\n")
        val_acc = best_metrics.get('val_acc', 'N/A')
        f.write(f"Val Accuracy: {val_acc if isinstance(val_acc, str) else f'{val_acc:.4f}'}\n")
        val_f1 = best_metrics.get('val_f1', 'N/A')
        f.write(f"Val F1 Score: {val_f1 if isinstance(val_f1, str) else f'{val_f1:.4f}'}\n")
        val_precision = best_metrics.get('val_precision', 'N/A')
        f.write(f"Val Precision: {val_precision if isinstance(val_precision, str) else f'{val_precision:.4f}'}\n")
        val_recall = best_metrics.get('val_recall', 'N/A')
        f.write(f"Val Recall: {val_recall if isinstance(val_recall, str) else f'{val_recall:.4f}'}\n")
    
    print(f"Training finished. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Training log saved to {log_path}")
    print(f"Best results saved to {results_path}")

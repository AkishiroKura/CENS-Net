"""
训练函数
"""
import torch
import csv
from evaluate import evaluate, extract_features
from plot import plot_training_curves, plot_predictions, plot_tsne


def train_model(model, train_loader, val_loader, device, criterion, optimizer, 
                num_epochs=100, save_path="best_model.pt", log_path="training_log.csv", results_path="best_results.txt", fold_num=None, preds_save_path=None, save_dir="."):
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        save_path: 最佳模型保存路径
        log_path: 训练日志保存路径
        results_path: 最佳结果保存路径
    """
    best_val_f1 = 0.0  # 改为追踪最佳F1分数（越大越好）
    best_epoch = -1
    best_metrics = {}
    best_preds = None
    best_labels = None
    best_probs = None
    
    # 初始化日志文件
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_f1", "val_precision", "val_recall"])
    
    # 评估初始状态（训练前）
    print("Evaluating initial model state (before training)...")
    initial_val_loss, initial_val_acc, initial_val_f1, initial_val_precision, initial_val_recall, _, _, _ = evaluate(model, val_loader, device, criterion)
    print(f"Initial State | val_loss = {initial_val_loss:.4f} | val_acc = {initial_val_acc:.4f} | val_f1 = {initial_val_f1:.4f}")
    
    # 记录初始状态（epoch = -1 表示训练前）
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([-1, "N/A", f"{initial_val_loss:.4f}", f"{initial_val_acc:.4f}", f"{initial_val_f1:.4f}", f"{initial_val_precision:.4f}", f"{initial_val_recall:.4f}"])
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        batch_count = 0
        try:
            for batch in train_loader:
                # 简单检查并跳过坏 batch
                if torch.isnan(batch.x).any() or (hasattr(batch, "edge_attr") and batch.edge_attr is not None and torch.isnan(batch.edge_attr).any()):
                    print(f"Skipping train batch: NaN in inputs/edge_attr")
                    continue
                if torch.isnan(batch.y).any():
                    print("Skipping train batch: NaN in labels")
                    continue

                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  Epoch {epoch:02d} Batch {batch_count}: Invalid loss (NaN/Inf)")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            train_avg = total_loss / batch_count if batch_count > 0 else float("nan")
            val_loss, val_acc, val_f1, val_precision, val_recall, val_preds, val_labels, val_probs = evaluate(model, val_loader, device, criterion)
            print(f"Epoch {epoch:02d} | train_loss = {train_avg:.4f} | val_loss = {val_loss:.4f} | val_acc = {val_acc:.4f} | val_f1 = {val_f1:.4f} | val_prec = {val_precision:.4f} | val_recall = {val_recall:.4f}")

            # 保存日志
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{train_avg:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", f"{val_f1:.4f}", f"{val_precision:.4f}", f"{val_recall:.4f}"])

            # 保存最优模型（按 val_f1）
            if not (val_f1 != val_f1):  # val_f1 不是 NaN
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch = epoch
                    best_metrics = {
                        "epoch": epoch,
                        "train_loss": train_avg,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_f1": val_f1,
                        "val_precision": val_precision,
                        "val_recall": val_recall
                    }
                    best_preds = val_preds
                    best_labels = val_labels
                    best_probs = val_probs
                    torch.save(model.state_dict(), save_path)
                    print(f"  Saved best model (epoch {epoch}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f})")
                    # 如果传入折号，则保存预测结果以便后续汇总（npz）
                    try:
                        import numpy as _np
                        if fold_num is not None and best_preds is not None and best_labels is not None:
                            save_name = preds_save_path if preds_save_path else f"best_preds_fold{fold_num}.npz"
                            _np.savez_compressed(save_name, preds=_np.array(best_preds), labels=_np.array(best_labels), probs=_np.array(best_probs) if best_probs is not None else None)
                            print(f"  Saved predictions to {save_name}")
                    except Exception as _e:
                        print(f"  Warning: failed to save preds npz: {_e}")
        except Exception as e:
            print(f"Epoch {epoch:02d} interrupted by error: {e}")
            break

    # 保存最佳结果到文件
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

    print(f"Training finished. Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Training log saved to {log_path}")
    print(f"Best results saved to {results_path}")
    
    # 绘制训练曲线
    plot_training_curves(log_path, save_dir=save_dir, fold_num=fold_num)
    
    # 绘制最佳模型的预测vs真实值
    if best_preds is not None and best_labels is not None:
        plot_predictions(best_preds, best_labels, best_probs, save_dir=save_dir, title="Best Model: Predictions vs Ground Truth", fold_num=fold_num)
    
    # 提取特征并绘制 t-SNE（需要先加载最佳模型）
    print("Extracting features for t-SNE visualization...")
    try:
        # 加载最佳模型权重
        model.load_state_dict(torch.load(save_path))
        print(f"  Loaded best model from {save_path}")
        
        features, labels = extract_features(model, val_loader, device)
        if len(features) > 0:
            # 自动调整 perplexity（必须小于样本数）
            n_samples = len(features)
            perplexity = min(30, max(5, n_samples // 3))
            print(f"  Using perplexity={perplexity} for {n_samples} samples")
            plot_tsne(features, labels, save_dir=save_dir, perplexity=perplexity, fold_num=fold_num)
        else:
            print("  Warning: No features extracted, skipping t-SNE")
    except Exception as e:
        print(f"  Warning: Failed to generate t-SNE: {e}")

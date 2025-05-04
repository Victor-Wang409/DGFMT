"""
训练执行模块
负责整个训练流程的执行
"""

import os
import logging
import torch
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from early_stopping import EarlyStopping
from lr_scheduler import LRSchedulerFactory
from data_processor import DataProcessor
from loss_functions import LossFactory
from model import VADConfig, VADModelWithGating
from trainer import TrainingManager

class TrainerExecutor:
    """
    训练执行器，负责整个训练流程的执行
    """
    @staticmethod
    def train_model(args, fold, fold_dir, dataset, train_idx, eval_idx, test_idx, device):
        """
        训练单个fold的模型
        
        参数:
            args: 参数配置
            fold: 当前fold索引
            fold_dir: fold目录
            dataset: 数据集
            train_idx: 训练集索引
            eval_idx: 验证集索引
            test_idx: 测试集索引
            device: 设备
            
        返回:
            测试结果
        """
        # 创建数据加载器
        train_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(train_idx),
            collate_fn=DataProcessor.collate_fn
        )
        eval_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(eval_idx),
            collate_fn=DataProcessor.collate_fn
        )
        test_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(test_idx),
            collate_fn=DataProcessor.collate_fn
        )
        
        # 配置模型
        config = VADConfig(
            emotion2vec_dim=1024,
            hubert_dim=1024,
            hidden_dim=1024,
            num_hidden_layers=4,
            num_groups=8,
            # 新增特征维度，默认为0表示不使用
            wavlm_dim=0,
            whisper_dim=0
        )

        # 创建模型
        model = VADModelWithGating(config).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        num_training_steps = len(train_loader)
        
        # 学习率调度器
        scheduler = LRSchedulerFactory.create_scheduler(optimizer, args, num_training_steps)
        criterion = LossFactory.CCCLoss()
        contrast_criterion = LossFactory.SupervisedContrastiveLoss(temperature=0.07)
        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
        
        best_val_ccc = -float('inf')
        best_model = None
        
        # 创建性能跟踪文件
        metrics_file = os.path.join(fold_dir, 'metrics.csv')
        with open(metrics_file, 'w') as f:
            f.write('epoch,train_loss,val_ccc_v,val_ccc_a,val_ccc_d,val_ccc_avg\n')
        
        for epoch in range(args.epochs):
            # 训练一个epoch
            metrics = TrainingManager.train_one_epoch(
                model, 
                optimizer, 
                criterion, 
                contrast_criterion, 
                train_loader, 
                device, 
                args.gradient_accumulation_steps
            )
            train_loss = metrics['loss']
            
            # 更新学习率
            if args.lr_scheduler == 'step':
                scheduler.step()
            else:
                for _ in range(num_training_steps):
                    scheduler.step()
                    
            # 记录当前学习率
            current_lr = scheduler.get_last_lr()[0]
            logging.info(f"Epoch {epoch+1}")
            logging.info(f"Current learning rate: {current_lr:.2e}")
            logging.info(f"Average emotion2vec weight: {metrics['e2v_weight']:.3f}")
            logging.info(f"Average hubert weight: {metrics['hub_weight']:.3f}")
            
            # 验证
            val_v, val_a, val_d = TrainingManager.validate_and_test(model, eval_loader, device)
            val_ccc_avg = (val_v + val_a + val_d) / 3
            
            # 保存每个epoch的模型
            epoch_dir = os.path.join(fold_dir, f'epoch{epoch+1}')
            os.makedirs(epoch_dir, exist_ok=True)
            
            # 保存模型和配置
            model.save_pretrained(epoch_dir, safe_serialization=False)
            
            # 保存优化器状态
            torch.save(optimizer.state_dict(), os.path.join(epoch_dir, 'optimizer.pt'))
            
            # 保存训练指标
            with open(metrics_file, 'a') as f:
                f.write(f'{epoch+1},{train_loss:.4f},{val_v:.4f},{val_a:.4f},{val_d:.4f},{val_ccc_avg:.4f}\n')
            
            logging.info(
                f"Fold {fold+1}, Epoch {epoch+1:3d} | "
                f"Loss: {train_loss:.4f} | "
                f"Val CCC: V={val_v:.3f}, A={val_a:.3f}, D={val_d:.3f} | "
                f"Avg={val_ccc_avg:.3f}"
            )

            # 测试
            test_v, test_a, test_d = TrainingManager.validate_and_test(model, test_loader, device)
            logging.info(f"Test: CCC_v={test_v:.3f}, CCC_a={test_a:.3f}, CCC_d={test_d:.3f}")
            
            # 更新最佳模型
            if val_ccc_avg > best_val_ccc:
                best_val_ccc = val_ccc_avg
                best_model = model.state_dict()
                # 创建并保存最佳模型
                best_model_dir = os.path.join(fold_dir, 'best_model')
                os.makedirs(best_model_dir, exist_ok=True)
                model.save_pretrained(best_model_dir, safe_serialization=False)
                logging.info(f"Saved new best model with val_ccc={val_ccc_avg:.3f}")
            
            # 保存checkpoint以便恢复训练
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_ccc': best_val_ccc,
                'config': config.to_dict()
            }
            torch.save(checkpoint, os.path.join(fold_dir, 'checkpoint.pt'))
            
            # 早停检查
            early_stopping(1 - val_ccc_avg)
            if early_stopping.early_stop:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # 加载最佳模型进行测试
        model.load_state_dict(best_model)
        test_v, test_a, test_d = TrainingManager.validate_and_test(model, test_loader, device)
        test_ccc_avg = (test_v + test_a + test_d) / 3
        
        # 保存测试结果
        with open(os.path.join(fold_dir, 'test_results.txt'), 'w') as f:
            f.write(f"Test CCC:\nValence: {test_v:.3f}\nArousal: {test_a:.3f}\n"
                   f"Dominance: {test_d:.3f}\nAverage: {test_ccc_avg:.3f}")
        
        return test_v, test_a, test_d
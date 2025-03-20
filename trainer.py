"""
训练和评估模块
负责模型的训练和评估
"""

import torch
from tqdm import tqdm
from loss_functions import LossFactory

class TrainingManager:
    """
    训练管理器，封装训练和评估相关的函数
    """
    @staticmethod
    def train_one_epoch(model, optimizer, criterion, contrast_criterion, train_loader, device, gradient_accumulation_steps):
        """
        训练一个epoch
        
        参数:
            model: 模型
            optimizer: 优化器
            criterion: 主损失函数
            contrast_criterion: 对比损失函数
            train_loader: 训练数据加载器
            device: 设备
            gradient_accumulation_steps: 梯度累积步数
            
        返回:
            训练指标字典
        """
        model.train()
        total_loss = 0.0
        total_batches = 0
        total_e2v_weight = 0.0
        total_hub_weight = 0.0
        total_vad_loss = 0.0
        total_contrast_loss = 0.0
        
        optimizer.zero_grad()
        
        progress_bar = tqdm(total=len(train_loader), desc='Training', leave=False)
        
        for batch_idx, batch in enumerate(train_loader):
            emotion2vec_features = batch["emotion2vec_features"].to(device)
            hubert_features = batch["hubert_features"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            vad_labels = batch["labels"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)

            # 获取标签索引,用于对比学习
            emotion_indices = torch.argmax(emotion_labels, dim=1)
            
            # 前向传播,增加对比学习特征
            outputs, (e2v_weight, hub_weight), contrast_features = model(
                emotion2vec_features,
                hubert_features,
                padding_mask
            )
            vad_loss = criterion(outputs, vad_labels)
            # 对比损失
            contrast_loss = contrast_criterion(contrast_features, emotion_indices)

            # 总损失
            loss = vad_loss + 0.1 * contrast_loss  # 权重系数可调
            
            # 记录门控权重
            total_e2v_weight += e2v_weight.mean().item()
            total_hub_weight += hub_weight.mean().item()
            
            # 梯度累积
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * gradient_accumulation_steps
            total_vad_loss += vad_loss.item()
            total_contrast_loss += contrast_loss.item()
            total_batches += 1
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()  # 清理显存
                
                avg_e2v_weight = total_e2v_weight / total_batches
                avg_hub_weight = total_hub_weight / total_batches
                
                progress_bar.set_postfix({
                    'loss': f'{(total_loss/(batch_idx+1)):.4f}',
                    'vad_loss': f'{(total_vad_loss/(batch_idx+1)):.4f}',
                    'contrast_loss': f'{(total_contrast_loss/(batch_idx+1)):.4f}',
                    'e2v_w': f'{avg_e2v_weight:.3f}',
                    'hub_w': f'{avg_hub_weight:.3f}'
                })
                progress_bar.update(gradient_accumulation_steps)
            
        progress_bar.close()
        
        # 返回平均损失和平均门控权重
        return {
            'loss': total_loss / len(train_loader),
            'vad_loss': total_vad_loss / len(train_loader),
            'contrast_loss': total_contrast_loss / len(train_loader),
            'e2v_weight': total_e2v_weight / total_batches,
            'hub_weight': total_hub_weight / total_batches
        }

    @staticmethod
    def validate_and_test(model, data_loader, device):
        """
        验证和测试模型
        
        参数:
            model: 模型
            data_loader: 数据加载器
            device: 设备
            
        返回:
            验证/测试指标
        """
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating', leave=False):
                # 获取分离的特征
                emotion2vec_features = batch["emotion2vec_features"].to(device)
                hubert_features = batch["hubert_features"].to(device)
                padding_mask = batch["padding_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # 前向传播 (注意接收门控权重但不使用)
                outputs, _, _ = model(emotion2vec_features, hubert_features, padding_mask)
                
                all_preds.append(outputs)
                all_labels.append(labels)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        ccc_v = LossFactory._compute_dimension_ccc(all_preds[:, 0], all_labels[:, 0]).item()
        ccc_a = LossFactory._compute_dimension_ccc(all_preds[:, 1], all_labels[:, 1]).item()
        ccc_d = LossFactory._compute_dimension_ccc(all_preds[:, 2], all_labels[:, 2]).item()
        
        return ccc_v, ccc_a, ccc_d
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
    def train_one_epoch(model, optimizer, vad_criterion, contrast_criterion, train_loader, device):
        """
        训练一个epoch
        
        参数:
            model: 模型
            optimizer: 优化器
            vad_criterion: 主损失函数
            contrast_criterion: 对比损失函数
            train_loader: 训练数据加载器
            device: 设备
            gradient_accumulation_steps: 梯度累积步数

        参数修改:
            criterion -> vad_criterion (CCC Loss)
            
        返回:
            训练指标字典
        """
        model.train()
        total_loss = 0.0
        total_batches = 0
        total_weights = {}  # 存储各特征的权重
        total_vad_loss = 0.0
        total_contrast_loss = 0.0
        
        optimizer.zero_grad()
        
        progress_bar = tqdm(total=len(train_loader), desc='Training', leave=False)
        
        for batch_idx, batch in enumerate(train_loader):

            optimizer.zero_grad()

            # 准备特征字典
            features = {
                "emotion2vec": batch["emotion2vec_features"].to(device),
                "hubert": batch["hubert_features"].to(device)
            }
            
            # 添加其他特征（如果存在）
            if "wav2vec_features" in batch:
                features["wav2vec"] = batch["wav2vec_features"].to(device)
            if "data2vec_features" in batch:
                features["data2vec"] = batch["data2vec_features"].to(device)

            padding_mask = batch["padding_mask"].to(device)
            vad_labels = batch["labels"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)
            emotion_indices = torch.argmax(emotion_labels, dim=1)
            
            vad_preds, feature_weights, contrast_features, current_temp = model(
                features,
                padding_mask
            )
            
            vad_loss = vad_criterion(vad_preds, vad_labels)
            contrast_loss = contrast_criterion(contrast_features, emotion_indices)
            loss = 1.0 * vad_loss + 0.6 * contrast_loss

            # 反向传播与更新
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            # 记录数据
            total_loss += loss.item()
            total_vad_loss += vad_loss.item()
            total_contrast_loss += contrast_loss.item()
            total_batches += 1
            
            # 记录门控权重
            for feat_type, weight in feature_weights.items():
                if feat_type not in total_weights:
                    total_weights[feat_type] = 0.0
                total_weights[feat_type] += weight.mean().item()

            # 更新进度条
            avg_weights = {f"{k}_w": f"{total_weights[k] / total_batches:.3f}" for k in total_weights}
            postfix_info = {'loss': f'{(total_loss/(batch_idx+1)):.4f}'}
            postfix_info.update(avg_weights)
            progress_bar.set_postfix(postfix_info)
            progress_bar.update(1) # 改为每步更新1
        
        progress_bar.close()
        
        # 返回结果
        result = {
            'loss': total_loss / len(train_loader),
            'vad_loss': total_vad_loss / len(train_loader),
            'contrast_loss': total_contrast_loss / len(train_loader),
        }
    
        # 添加各特征的平均权重
        for feat_type in total_weights:
            result[f'{feat_type}_weight'] = total_weights[feat_type] / total_batches
        
        return result

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
        all_vad_preds = []
        all_vad_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating', leave=False):
                # 准备特征字典
                features = {
                    "emotion2vec": batch["emotion2vec_features"].to(device),
                    "hubert": batch["hubert_features"].to(device)
                }
                
                # 添加其他特征（如果存在）
                if "wav2vec_features" in batch:
                    features["wav2vec"] = batch["wav2vec_features"].to(device)
                if "data2vec_features" in batch:
                    features["data2vec"] = batch["data2vec_features"].to(device)
                    
                padding_mask = batch["padding_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # [修改 7] 获取多任务输出
                vad_preds, discrete_logits, _, _ = model(features, padding_mask)
                
                all_vad_preds.append(vad_preds)
                all_vad_labels.append(batch["labels"].to(device))
                
        
        # 拼接结果
        all_vad_preds = torch.cat(all_vad_preds, dim=0)
        all_vad_labels = torch.cat(all_vad_labels, dim=0)
        
        # 计算指标
        ccc_v = LossFactory._compute_dimension_ccc(all_vad_preds[:, 0], all_vad_labels[:, 0]).item()
        ccc_a = LossFactory._compute_dimension_ccc(all_vad_preds[:, 1], all_vad_labels[:, 1]).item()
        ccc_d = LossFactory._compute_dimension_ccc(all_vad_preds[:, 2], all_vad_labels[:, 2]).item()
        
        return ccc_v, ccc_a, ccc_d
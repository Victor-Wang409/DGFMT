"""
损失函数模块
包含用于训练的各种损失函数实现
"""

import torch
from torch import nn
import torch.nn.functional as F

class LossFactory:
    """
    损失函数工厂类，用于创建和管理各种损失函数
    """
    class SupervisedContrastiveLoss(nn.Module):
        """
        监督对比损失，使同类样本在特征空间中更接近
        """
        def __init__(self, temperature=0.07):
            """
            初始化对比损失
            
            参数:
                temperature: 温度参数，控制分布平滑程度
            """
            super().__init__()
            self.temperature = temperature
            
        def forward(self, features, labels):
            """
            计算对比损失
            
            参数:
                features: 特征表示
                labels: 标签
                
            返回:
                对比损失值
            """
            # 添加数值稳定性检查
            if torch.isnan(features).any() or torch.isinf(features).any():
                return torch.tensor(0.0, device=features.device, requires_grad=True)
                
            batch_size = features.shape[0]
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float()
            mask = mask.fill_diagonal_(0)
            
            # 特征归一化,添加eps避免除零
            features = F.normalize(features, dim=1, eps=1e-8)
            
            # 计算相似度矩阵时添加数值稳定性
            similarity_matrix = torch.clamp(
                torch.matmul(features, features.T),
                min=-1.0,
                max=1.0
            )
            
            logits = similarity_matrix / self.temperature
            
            # 使用log_softmax提高数值稳定性
            log_prob = F.log_softmax(logits, dim=1)
            
            # 计算正样本的loss
            positives = mask * log_prob
            
            # 平均每个样本的loss
            loss = -positives.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            
            return loss.mean()

    class CCCLoss(nn.Module):
        """
        一致性相关系数损失，用于回归问题
        """
        def __init__(self):
            """
            初始化CCC损失
            """
            super().__init__()
            
        def forward(self, preds, labels):
            """
            计算CCC损失
            
            参数:
                preds: 预测值
                labels: 真实值
                
            返回:
                CCC损失值
            """
            ccc_v = LossFactory._compute_dimension_ccc(preds[:, 0], labels[:, 0])
            ccc_a = LossFactory._compute_dimension_ccc(preds[:, 1], labels[:, 1])
            ccc_d = LossFactory._compute_dimension_ccc(preds[:, 2], labels[:, 2])
            
            mean_ccc = (ccc_v + ccc_a + ccc_d) / 3.0
            return torch.tensor(1.0, device=preds.device) - mean_ccc

    @staticmethod
    def _compute_dimension_ccc(preds, labels):
        """
        计算单个维度的一致性相关系数
        
        参数:
            preds: 预测值
            labels: 真实值
            
        返回:
            CCC值
        """
        preds_mean = torch.mean(preds)
        labels_mean = torch.mean(labels)
        
        preds_var = torch.mean((preds - preds_mean) ** 2)
        labels_var = torch.mean((labels - labels_mean) ** 2)
        
        covar = torch.mean((preds - preds_mean) * (labels - labels_mean))
        
        ccc = 2 * covar / (preds_var + labels_var + (preds_mean - labels_mean) ** 2 + 1e-8)
        return ccc
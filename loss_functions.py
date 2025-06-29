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
        监督对比损失
        """
        def __init__(self, temperature=0.1):
            """
            初始化对比损失
            
            参数:
                temperature: 温度参数，控制分布平滑程度
            """
            super().__init__()
            self.temperature = temperature
            
        def forward(self, features, labels):
            """
            计算对比损失，使用论文中的算法实现
            
            参数:
                features: 特征表示 [batch_size, feature_dim]
                labels: 标签索引 [batch_size]
                
            返回:
                对比损失值
            """
            # 添加数值稳定性检查
            if torch.isnan(features).any() or torch.isinf(features).any():
                return torch.tensor(0.0, device=features.device, requires_grad=True)
            
            batch_size = features.shape[0]
            
            # 特征归一化
            features = F.normalize(features, dim=1, eps=1e-8)
            
            # 计算内积得到相似度矩阵 (类似于伪代码中的einsum操作)
            similarity_matrix = torch.matmul(features, features.T)
            similarity_matrix = torch.clamp(similarity_matrix, min=-1.0, max=1.0)
            
            # 应用温度缩放并计算log_softmax (对应伪代码中的LogSoftmax(l_pn/τ))
            logits = similarity_matrix / self.temperature
            log_probs = F.log_softmax(logits, dim=1)
            
            # 构建索引矩阵，用于gather操作
            # 为每个样本创建同类样本的索引列表
            L_cl = []
            for i in range(batch_size):
                # 找出与当前样本i同类的所有样本索引
                same_class_indices = torch.where(labels == labels[i])[0]
                # 排除自身
                same_class_indices = same_class_indices[same_class_indices != i]
                L_cl.append(same_class_indices)
            
            # 计算损失
            loss = torch.tensor(0.0, device=features.device)
            for i in range(batch_size):
                if len(L_cl[i]) > 0:  # 确保有同类样本
                    # 从log_probs中收集同类样本的概率值 (对应伪代码中的gather操作)
                    pos_logits = log_probs[i, L_cl[i]]
                    # 计算平均损失 (对应伪代码中的除以T)
                    sample_loss = -torch.mean(pos_logits)
                    loss += sample_loss
            
            # 对batch中的所有样本取平均
            return loss / batch_size if batch_size > 0 else loss

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
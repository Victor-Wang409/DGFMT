"""
模型组件模块
包含各种模型组件的实现，增强了门控融合机制
"""

import torch
from torch import nn
import torch.nn.functional as F

class ModelComponents:
    """
    模型组件类，包含各种模型组件的实现
    """
    class AttentionPooling(nn.Module):
        """
        注意力池化，用于提取序列特征的全局表示
        """
        def __init__(self, hidden_dim):
            """
            初始化注意力池化
            
            参数:
                hidden_dim: 隐藏层维度
            """
            super().__init__()
            self.attention = nn.Linear(hidden_dim, 1)

        def forward(self, x, padding_mask=None):
            """
            前向传播
            
            参数:
                x: 输入特征 [batch_size, seq_len, hidden_dim]
                padding_mask: 填充掩码
                
            返回:
                池化后的表示
            """
            # 计算注意力分数
            attn_weights = self.attention(x)
            attn_weights = attn_weights.squeeze(-1)

            if padding_mask is not None:
                attn_weights = attn_weights.masked_fill(padding_mask, float('-inf'))
            attn_weights = torch.softmax(attn_weights, dim=1)
            weights_sum = torch.bmm(attn_weights.unsqueeze(1), x)
            weights_sum = weights_sum.squeeze(1)

            return weights_sum

    class MultiHeadAttention(nn.Module):
        """
        多头注意力机制
        """
        def __init__(self, config):
            """
            初始化多头注意力
            
            参数:
                config: 配置对象
            """
            super().__init__()
            self.num_heads = config.num_attention_heads
            self.hidden_dim = config.hidden_dim
            self.head_dim = config.hidden_dim // config.num_attention_heads
            self.scaling = self.head_dim ** -0.5

            self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            
        def forward(self, x, key_padding_mask=None):
            """
            前向传播
            
            参数:
                x: 输入特征 [batch_size, seq_len, embed_dim]
                key_padding_mask: 键填充掩码
                
            返回:
                注意力计算结果
            """
            batch_size, seq_len, embed_dim = x.shape
            
            # 投影得到query、key、value
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # 计算注意力分数
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                )

            attn_weights = torch.softmax(attn_weights, dim=-1)
            
            # 注意力加权
            attn = torch.matmul(attn_weights, v)
            
            # 重塑并投影
            attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
            attn = self.out_proj(attn)
            
            return attn

    class TransformerEncoderLayer(nn.Module):
        """
        Transformer编码器层
        """
        def __init__(self, config):
            """
            初始化Transformer编码器层
            
            参数:
                config: 配置对象
            """
            super().__init__()
            self.self_attn = ModelComponents.MultiHeadAttention(config)
            
            self.linear1 = nn.Linear(config.hidden_dim, config.intermediate_dim)
            self.linear2 = nn.Linear(config.intermediate_dim, config.hidden_dim)
            
            self.norm1 = nn.LayerNorm(config.hidden_dim)
            self.norm2 = nn.LayerNorm(config.hidden_dim)
            
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.activation = nn.GELU()
            
        def forward(self, x, padding_mask=None):
            """
            前向传播
            
            参数:
                x: 输入特征
                padding_mask: 填充掩码
                
            返回:
                编码后的特征
            """
            # 自注意力层
            residual = x
            x = self.norm1(x)
            x = self.self_attn(x, padding_mask)
            x = self.dropout(x)
            x = residual + x
            
            # 前馈网络
            residual = x
            x = self.norm2(x)
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            x = self.dropout(x)
            x = residual + x
            
            return x

    class GatedFeatureFusion(nn.Module):
        """
        门控特征融合机制
        将多粒度融合和时序敏感处理改为串行关系
        支持2-4种不同类型的输入特征
        """
        def __init__(self, feature_dims, num_groups=8, dropout_rate=0.1):
            """
            初始化门控特征融合
            
            参数:
                feature_dims: 字典，包含各特征类型及其维度，如{'emotion2vec': 1024, 'hubert': 1024}
                num_groups: 多粒度门控的分组数量
                dropout_rate: dropout比例
            """
            super().__init__()
            self.feature_types = list(feature_dims.keys())
            self.feature_dims = feature_dims
            self.num_features = len(self.feature_types)
            self.dropout_rate = dropout_rate
            
            # 验证特征数量满足要求
            assert 2 <= self.num_features <= 4, f"特征数量必须在2-4之间，当前为{self.num_features}"
            
            # 特征变换和归一化
            self.feature_transforms = nn.ModuleDict()
            self.feature_norms = nn.ModuleDict()
            
            # 使用第一个特征的维度作为标准维度
            self.standard_dim = list(feature_dims.values())[0]
            
            # 为每种特征创建转换层和归一化层
            for feat_type, dim in feature_dims.items():
                # 添加dropout提高泛化能力
                self.feature_transforms[feat_type] = nn.Sequential(
                    nn.Linear(dim, self.standard_dim),
                    nn.Dropout(dropout_rate)
                )
                self.feature_norms[feat_type] = nn.LayerNorm(dim)
            
            # 多粒度门控
            self.num_groups = num_groups
            self.group_size = self.standard_dim // self.num_groups
            
            # 为每组特征创建独立的门控网络，添加残差连接和改进激活
            self.group_gate_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.group_size * self.num_features, self.group_size),
                    nn.LayerNorm(self.group_size),  # 添加层归一化
                    nn.GELU(),  # 使用GELU替代ReLU
                    nn.Dropout(dropout_rate),
                    nn.Linear(self.group_size, self.num_features),
                ) for _ in range(self.num_groups)
            ])
            
            # 时序处理LSTM，添加dropout和更好的初始化
            self.temporal_lstm = nn.LSTM(
                input_size=self.standard_dim * self.num_features,
                hidden_size=self.standard_dim,
                batch_first=True,
                bidirectional=True,
                dropout=dropout_rate if dropout_rate > 0 else 0,
                num_layers=3
            )
            
            # 添加残差连接的投影层
            self.residual_proj = nn.Linear(self.standard_dim * self.num_features, self.standard_dim * 2)
            
            # 最终的融合层，添加层归一化
            self.final_norm = nn.LayerNorm(self.standard_dim * 2)
            self.final_projection = nn.Sequential(
                nn.Linear(self.standard_dim * 2, self.standard_dim * self.num_features),
                nn.Dropout(dropout_rate)
            )
            
            # 初始化权重
            self._init_weights()
            
        def _init_weights(self):
            """改进的权重初始化"""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    # 使用Xavier初始化
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LSTM):
                    # LSTM权重初始化
                    for name, param in module.named_parameters():
                        if 'weight_ih' in name:
                            nn.init.xavier_uniform_(param.data)
                        elif 'weight_hh' in name:
                            nn.init.orthogonal_(param.data)
                        elif 'bias' in name:
                            nn.init.zeros_(param.data)
                            # 设置遗忘门偏置为1
                            n = param.size(0)
                            param.data[n//4:n//2].fill_(1.)
            
        def forward(self, features):
            """
            前向传播
            
            参数:
                features: 字典，包含各类型特征，如{'emotion2vec': tensor, 'hubert': tensor}
                
            返回:
                融合后的特征和门控权重
            """
            batch_size, seq_len, _ = features[self.feature_types[0]].shape
            
            # 1. 特征归一化和转换
            transformed_features = {}
            for feat_type in self.feature_types:
                normalized = self.feature_norms[feat_type](features[feat_type])
                transformed = self.feature_transforms[feat_type](normalized)
                transformed_features[feat_type] = transformed
            
            # 2. 多粒度门控融合
            multi_grained_features = []
            gate_weights = {feat_type: [] for feat_type in self.feature_types}
            
            for i in range(self.num_groups):
                # 提取当前组的所有特征
                start_idx = i * self.group_size
                end_idx = (i + 1) * self.group_size
                
                group_feats = []
                for feat_type in self.feature_types:
                    group_feats.append(transformed_features[feat_type][..., start_idx:end_idx])
                
                # 拼接特征并计算门控权重
                group_concat = torch.cat(group_feats, dim=-1)
                group_logits = self.group_gate_nets[i](group_concat)
                
                # 使用温度缩放的softmax提高数值稳定性
                temperature = 0.1
                group_gates = F.softmax(group_logits / temperature, dim=-1)
                
                # 添加噪声防止过度自信
                if self.training:
                    noise = torch.randn_like(group_gates) * 0.01
                    group_gates = F.softmax((group_logits + noise) / temperature, dim=-1)
                
                # 加权特征
                weighted_feats = []
                for j, feat_type in enumerate(self.feature_types):
                    weight = group_gates[..., j:j+1]
                    gate_weights[feat_type].append(weight)
                    weighted_feat = group_feats[j] * weight
                    weighted_feats.append(weighted_feat)
                
                # 拼接加权后的特征
                group_feature = torch.cat(weighted_feats, dim=-1)
                multi_grained_features.append(group_feature)
            
            # 拼接所有组的特征
            multi_grained_fusion = torch.cat(multi_grained_features, dim=-1)
            
            # 计算各特征的平均权重（用于监控）
            avg_weights = {}
            for feat_type in self.feature_types:
                avg_weights[feat_type] = torch.cat(gate_weights[feat_type], dim=-1).mean(dim=-1, keepdim=True)
            
            # 3. 时序处理 - 添加残差连接
            # 为LSTM输入创建残差连接
            residual_input = self.residual_proj(multi_grained_fusion)
            
            # 梯度裁剪
            if self.training:
                torch.nn.utils.clip_grad_norm_(self.temporal_lstm.parameters(), max_norm=1.0)
            
            temporal_features, _ = self.temporal_lstm(multi_grained_fusion)
            
            # 添加残差连接
            temporal_features = temporal_features + residual_input
            
            # 4. 最终投影
            temporal_features = self.final_norm(temporal_features)
            fused_features = self.final_projection(temporal_features)
            
            return fused_features, avg_weights
            
        def get_fusion_weights(self):
            """
            获取当前门控融合机制使用的策略权重
            
            返回:
                包含权重信息的字典
            """
            # 这里可以返回多粒度和时序处理的相关权重
            # 实际中可能需要通过注册hook等方式获取
            return {
                "grain_weight": 0.5,  # 示例值
                "temporal_weight": 0.5
            }
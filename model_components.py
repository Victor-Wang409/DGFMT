"""
模型组件模块
包含各种模型组件的实现，增强了门控融合机制
"""

import torch
from torch import nn

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
        门控特征融合机制 - 增强版
        集成了多粒度和时序敏感特性
        """
        def __init__(self, emotion2vec_dim, hubert_dim):
            """
            初始化门控特征融合
            
            参数:
                emotion2vec_dim: emotion2vec特征维度
                hubert_dim: hubert特征维度
            """
            super().__init__()
            self.emotion2vec_dim = emotion2vec_dim  # 1024
            self.hubert_dim = hubert_dim  # 1024
            
            # 特征转换层
            self.emotion2vec_transform = nn.Linear(emotion2vec_dim, emotion2vec_dim)
            self.hubert_transform = nn.Linear(hubert_dim, emotion2vec_dim)
            
            # LayerNorm层
            self.norm_emotion2vec = nn.LayerNorm(emotion2vec_dim)
            self.norm_hubert = nn.LayerNorm(hubert_dim)
            
            # 1. 多粒度门控
            self.num_groups = 4  # 将1024维特征分为4组，每组256维
            self.group_size = emotion2vec_dim // self.num_groups
            
            # 为每组特征创建独立的门控网络
            self.group_gate_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.group_size * 2, self.group_size),
                    nn.ReLU(),
                    nn.Linear(self.group_size, 2),
                    nn.Softmax(dim=-1)
                ) for _ in range(self.num_groups)
            ])
            
            # 2. 时序敏感门控
            self.temporal_lstm = nn.LSTM(
                input_size=emotion2vec_dim * 2,
                hidden_size=emotion2vec_dim,
                batch_first=True,
                bidirectional=True
            )
            
            # 时序门控网络
            self.temporal_gate_net = nn.Sequential(
                nn.Linear(emotion2vec_dim * 2, emotion2vec_dim),
                nn.ReLU(),
                nn.Linear(emotion2vec_dim, 2),
                nn.Softmax(dim=-1)
            )
            
            # 原始全局门控网络
            self.gate_net = nn.Sequential(
                nn.Linear(emotion2vec_dim * 2, emotion2vec_dim),
                nn.ReLU(),
                nn.Linear(emotion2vec_dim, 2),
                nn.Softmax(dim=-1)
            )
            
            # 融合权重参数 - 控制不同门控策略的权重
            self.grain_weight = nn.Parameter(torch.tensor(0.5))
            self.temporal_weight = nn.Parameter(torch.tensor(0.5))
            
        def forward(self, emotion2vec_feat, hubert_feat):
            """
            前向传播
            
            参数:
                emotion2vec_feat: emotion2vec特征
                hubert_feat: hubert特征
                
            返回:
                融合后的特征和门控权重
            """
            batch_size, seq_len, _ = emotion2vec_feat.shape
            
            # 1. 特征归一化
            emotion2vec_feat = self.norm_emotion2vec(emotion2vec_feat)  # [B, T, 1024]
            hubert_feat = self.norm_hubert(hubert_feat)  # [B, T, 1024]
            
            # 2. 特征转换
            emotion2vec_transformed = self.emotion2vec_transform(emotion2vec_feat)
            hubert_transformed = self.hubert_transform(hubert_feat)
            
            # 3. 多粒度门控融合
            multi_grained_features = []
            multi_grained_e_weights = []
            multi_grained_h_weights = []
            
            for i in range(self.num_groups):
                # 提取当前组的特征
                start_idx = i * self.group_size
                end_idx = (i + 1) * self.group_size
                
                e_group = emotion2vec_transformed[..., start_idx:end_idx]
                h_group = hubert_transformed[..., start_idx:end_idx]
                
                # 拼接特征并计算门控权重
                group_concat = torch.cat([e_group, h_group], dim=-1)
                group_gates = self.group_gate_nets[i](group_concat)
                
                # 权重分离并应用
                e_weight = group_gates[..., 0:1]
                h_weight = group_gates[..., 1:2]
                
                multi_grained_e_weights.append(e_weight)
                multi_grained_h_weights.append(h_weight)
                
                weighted_e = e_group * e_weight
                weighted_h = h_group * h_weight
                
                # 拼接加权后的特征
                group_feature = torch.cat([weighted_e, weighted_h], dim=-1)
                multi_grained_features.append(group_feature)
            
            # 拼接所有组的特征
            multi_grained_fusion = torch.cat(multi_grained_features, dim=-1)
            
            # 计算多粒度门控的平均权重
            multi_grained_e_weight = torch.cat(multi_grained_e_weights, dim=-1).mean(dim=-1, keepdim=True)
            multi_grained_h_weight = torch.cat(multi_grained_h_weights, dim=-1).mean(dim=-1, keepdim=True)
            
            # 4. 时序敏感门控
            # 拼接原始特征用于LSTM处理
            concat_features = torch.cat([emotion2vec_transformed, hubert_transformed], dim=-1)
            
            # 使用LSTM提取时序信息
            temporal_features, _ = self.temporal_lstm(concat_features)
            
            # 基于时序特征计算门控权重
            temporal_gates = self.temporal_gate_net(temporal_features)
            temporal_e_weight = temporal_gates[..., 0:1]
            temporal_h_weight = temporal_gates[..., 1:2]
            
            # 加权并拼接
            temporal_weighted_e = emotion2vec_transformed * temporal_e_weight
            temporal_weighted_h = hubert_transformed * temporal_h_weight
            
            temporal_fusion = torch.cat([temporal_weighted_e, temporal_weighted_h], dim=-1)
            
            # 5. 原始全局门控(保持兼容性)
            # 计算全局门控权重
            concat_feat = torch.cat([emotion2vec_transformed, hubert_transformed], dim=-1)
            gates = self.gate_net(concat_feat)
            
            # 分离权重
            global_e_weight = gates[..., 0:1]
            global_h_weight = gates[..., 1:2]
            
            # 6. 融合不同策略的结果
            # 归一化融合权重
            fusion_weights = torch.softmax(torch.stack([self.grain_weight, self.temporal_weight]), dim=0)
            grain_factor = fusion_weights[0]
            temporal_factor = fusion_weights[1]
            
            # 加权组合多粒度和时序敏感的结果
            fused_features = grain_factor * multi_grained_fusion + temporal_factor * temporal_fusion
            
            # 计算最终平均门控权重(用于监控)
            e_weight = grain_factor * multi_grained_e_weight + temporal_factor * temporal_e_weight
            h_weight = grain_factor * multi_grained_h_weight + temporal_factor * temporal_h_weight
            
            return fused_features, (e_weight, h_weight)
            
        def get_fusion_weights(self):
            """
            获取当前门控融合的权重
            用于可视化和分析
            
            返回:
                门控融合的粒度和时序权重
            """
            weights = torch.softmax(torch.stack([self.grain_weight, self.temporal_weight]), dim=0)
            return {
                'grain_weight': weights[0].item(),
                'temporal_weight': weights[1].item()
            }
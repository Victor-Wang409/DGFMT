"""
模型组件模块
包含各种模型组件的实现
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
        门控特征融合机制
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
            self.hubert_transform = nn.Linear(emotion2vec_dim, emotion2vec_dim)
            
            # LayerNorm层
            self.norm_emotion2vec = nn.LayerNorm(emotion2vec_dim)
            self.norm_hubert = nn.LayerNorm(hubert_dim)
            
            # 门控网络
            gate_dim = emotion2vec_dim * 2  # 2048
            self.gate_net = nn.Sequential(
                nn.Linear(gate_dim, gate_dim // 2),
                nn.ReLU(),
                nn.Linear(gate_dim // 2, 2),
                nn.Softmax(dim=-1)
            )
        
        def forward(self, emotion2vec_feat, hubert_feat):
            """
            前向传播
            
            参数:
                emotion2vec_feat: emotion2vec特征
                hubert_feat: hubert特征
                
            返回:
                融合后的特征和门控权重
            """
            # 特征归一化
            emotion2vec_feat = self.norm_emotion2vec(emotion2vec_feat)  # [B, T, 1024]
            hubert_feat = self.norm_hubert(hubert_feat)  # [B, T, 1024]
            
            # 特征转换
            emotion2vec_transformed = self.emotion2vec_transform(emotion2vec_feat)
            hubert_transformed = self.hubert_transform(hubert_feat)
            
            # 计算门控权重
            concat_feat = torch.cat([emotion2vec_transformed, hubert_transformed], dim=-1)
            gates = self.gate_net(concat_feat)
            
            # 分离权重和加权融合
            emotion2vec_weight = gates[..., 0:1]
            hubert_weight = gates[..., 1:2]
            
            weighted_emotion2vec = emotion2vec_transformed * emotion2vec_weight
            weighted_hubert = hubert_transformed * hubert_weight
            
            fused_features = torch.cat([weighted_emotion2vec, weighted_hubert], dim=-1)
            
            return fused_features, (emotion2vec_weight, hubert_weight)
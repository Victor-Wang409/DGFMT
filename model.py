"""
模型定义模块
定义VAD模型及其配置，更新支持多粒度和时序敏感的门控机制
"""

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from model_components import ModelComponents

class VADConfig(PretrainedConfig):
    """
    VAD模型配置类
    """
    def __init__(
        self,
        emotion2vec_dim=1024,
        hubert_dim=1024,
        hidden_dim=1024,
        intermediate_dim=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        hidden_dropout_prob=0.1,
        use_multi_grained_gating=True,  # 是否使用多粒度门控
        use_temporal_gating=True,       # 是否使用时序敏感门控
        num_groups=4,                   # 多粒度门控的分组数量
        **kwargs
    ):
        """
        初始化VAD配置
        
        参数:
            emotion2vec_dim: emotion2vec特征维度
            hubert_dim: hubert特征维度
            hidden_dim: 隐藏层维度
            intermediate_dim: 中间层维度
            num_hidden_layers: 隐藏层数量
            num_attention_heads: 注意力头数量
            hidden_dropout_prob: 隐藏层dropout概率
            use_multi_grained_gating: 是否使用多粒度门控
            use_temporal_gating: 是否使用时序敏感门控
            num_groups: 多粒度门控的分组数量
        """
        super().__init__(**kwargs)
        self.emotion2vec_dim = emotion2vec_dim
        self.hubert_dim = hubert_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_multi_grained_gating = use_multi_grained_gating
        self.use_temporal_gating = use_temporal_gating
        self.num_groups = num_groups

class VADModelWithGating(PreTrainedModel):
    """
    带门控机制的VAD模型
    支持多粒度和时序敏感的门控机制
    """
    def __init__(self, config):
        """
        初始化VAD模型
        
        参数:
            config: 模型配置
        """
        super().__init__(config)
        self.config = config
        
        self.emotion2vec_dim = config.emotion2vec_dim  # 1024
        self.hubert_dim = config.hubert_dim  # 1024
        
        # 使用增强版的门控特征融合
        self.feature_fusion = ModelComponents.GatedFeatureFusion(
            emotion2vec_dim=self.emotion2vec_dim,
            hubert_dim=self.hubert_dim
        )
        
        # 修改输入投影层维度，因为融合后的特征维度是 emotion2vec_dim * 2
        self.input_proj = nn.Linear(self.emotion2vec_dim * 2, config.hidden_dim)
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            ModelComponents.TransformerEncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # 输出层
        self.pooler = ModelComponents.AttentionPooling(config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, 3)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, emotion2vec_features, hubert_features, padding_mask=None):
        """
        前向传播
        
        参数:
            emotion2vec_features: emotion2vec特征
            hubert_features: hubert特征
            padding_mask: 填充掩码
            
        返回:
            预测结果、门控权重和池化特征
        """
        # 特征融合
        x, gate_weights = self.feature_fusion(emotion2vec_features, hubert_features)
        
        # 将融合后的特征映射到hidden_dim
        x = self.input_proj(x)
        x = self.dropout(x)
        
        # Transformer编码
        for layer in self.encoder_layers:
            x = layer(x, padding_mask)
            
        # 池化和输出
        pooled_features = self.pooler(x, padding_mask)
        x = torch.sigmoid(self.output_proj(pooled_features))
        
        return x, gate_weights, pooled_features
        
    def get_fusion_weights(self):
        """
        获取当前门控融合机制使用的策略权重
        用于分析模型如何权衡多粒度和时序信息
        
        返回:
            包含权重信息的字典
        """
        if hasattr(self.feature_fusion, 'get_fusion_weights'):
            return self.feature_fusion.get_fusion_weights()
        return None
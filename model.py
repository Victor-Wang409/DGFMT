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
        use_multi_grained_gating=True,
        use_temporal_gating=True,
        num_groups=8,
        wav2vec_dim=0,         # 新增：wav2vec特征维度，0表示不使用
        data2vec_dim=0,       # 新增：data2vec特征维度，0表示不使用
        num_emotions=8,  # [修改 1] 新增：离散情感类别数，默认为8 (根据dataset.py)
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
            wav2vec_dim: wav2vec特征维度，0表示不使用
            data2vec_dim: data2vec特征维度，0表示不使用
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
        self.wav2vec_dim = wav2vec_dim
        self.data2vec_dim = data2vec_dim
        self.num_emotions = num_emotions # [修改 1] 赋值

class VADModelWithGating(PreTrainedModel):
    """
    带门控机制的VAD模型
    支持多种特征输入的门控机制
    """
    def __init__(self, config):
        """
        初始化VAD模型
        
        参数:
            config: 模型配置
        """
        super().__init__(config)
        self.config = config
        
        # 确定特征类型和维度
        feature_dims = {'emotion2vec': config.emotion2vec_dim, 'hubert': config.hubert_dim}
        
        # 加入额外特征（如果配置中有）
        if hasattr(config, 'wav2vec_dim') and config.wav2vec_dim > 0:
            feature_dims['wav2vec'] = config.wav2vec_dim
        if hasattr(config, 'data2vec_dim') and config.data2vec_dim > 0:
            feature_dims['data2vec'] = config.data2vec_dim
            
        self.feature_types = list(feature_dims.keys())
        self.num_features = len(self.feature_types)
        
        # 使用增强版的门控特征融合
        self.feature_fusion = ModelComponents.GatedFeatureFusion(
            feature_dims=feature_dims,
            num_groups=config.num_groups
        )
        
        # 计算融合后的特征维度
        fusion_output_dim = config.emotion2vec_dim * 2
        
        # 修改输入投影层维度
        self.input_proj = nn.Linear(fusion_output_dim, config.hidden_dim)
        
        # Transformer编码器层
        # self.encoder_layers = nn.ModuleList([
        #     ModelComponents.TransformerEncoderLayer(config)
        #     for _ in range(config.num_hidden_layers)
        # ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_dim,
            dropout=config.hidden_dropout_prob,
            activation='gelu',
            batch_first=True,
            norm_first=True # 现代 Transformer 通常使用 Pre-Norm
        )
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        
        # 输出层
        self.pooler = ModelComponents.AttentionPooling(config.hidden_dim)

        self.output_proj_vad = nn.Linear(config.hidden_dim, 3)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, features, padding_mask=None):
        """
        前向传播
        
        参数:
            features: 字典，包含各特征类型
            padding_mask: 填充掩码
            
        返回:
            预测结果、门控权重和池化特征
        """
        # 特征融合
        x, gate_weights, current_temp = self.feature_fusion(features)
        
        # 将融合后的特征映射到hidden_dim
        x = self.input_proj(x)
        x = self.dropout(x)
        
        # Transformer编码
        # for layer in self.encoder_layers:
        #     x = layer(x, padding_mask)
        x = self.encoder_layers(x, src_key_padding_mask=padding_mask)
            
        # 池化和输出
        pooled_features = self.pooler(x, padding_mask)
        # VAD任务: 使用Sigmoid将输出限制在[0,1] (假设VAD标签已归一化)
        vad_output = torch.sigmoid(self.output_proj_vad(pooled_features))
        
        return vad_output, gate_weights, pooled_features, current_temp
        
    def get_fusion_weights(self):
        """
        获取当前门控融合机制使用的策略权重
        
        返回:
            包含权重信息的字典
        """
        if hasattr(self.feature_fusion, 'get_fusion_weights'):
            return self.feature_fusion.get_fusion_weights()
        return None
"""
门控权重分析工具
用于分析和可视化多粒度时序敏感门控机制的权重
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import VADModelWithGating

def analyze_gate_weights(model, data_loader, device, output_dir, emotion_map=None):
    """
    分析门控权重分布，按情感类别进行分组
    
    参数:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 设备
        output_dir: 输出目录
        emotion_map: 情感标签映射
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model.eval()
    all_e_weights = []
    all_h_weights = []
    all_emotions = []
    
    fusion_weights = model.get_fusion_weights()
    grain_weight = fusion_weights['grain_weight'] if fusion_weights else 0
    temporal_weight = fusion_weights['temporal_weight'] if fusion_weights else 0
    
    print(f"模型融合权重: 多粒度={grain_weight:.3f}, 时序敏感={temporal_weight:.3f}")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='收集门控权重'):
            emotion2vec_features = batch["emotion2vec_features"].to(device)
            hubert_features = batch["hubert_features"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            emotion_labels = batch["emotion_labels"].cpu()
            
            _, (e_weight, h_weight), _ = model(
                emotion2vec_features,
                hubert_features,
                padding_mask
            )
            
            # 收集平均门控权重
            all_e_weights.extend(e_weight.mean(dim=1).cpu().numpy())
            all_h_weights.extend(h_weight.mean(dim=1).cpu().numpy())
            
            # 获取情感标签
            emotions = torch.argmax(emotion_labels, dim=1).numpy()
            all_emotions.extend(emotions)
    
    # 创建数据框
    df = pd.DataFrame({
        'emotion': all_emotions,
        'e2v_weight': all_e_weights,
        'hub_weight': all_h_weights
    })
    
    # 按情感类别分组并计算平均权重
    emotion_weights = df.groupby('emotion').mean().reset_index()
    
    # 将数字标签映射为情感名称
    if emotion_map:
        emotion_weights['emotion_name'] = emotion_weights['emotion'].map(
            {v: k for k, v in emotion_map.items()}
        )
    else:
        emotion_weights['emotion_name'] = emotion_weights['emotion'].astype(str)
    
    # 按情感类别绘制门控权重
    plt.figure(figsize=(10, 6))
    x = np.arange(len(emotion_weights))
    width = 0.35
    
    plt.bar(x - width/2, emotion_weights['e2v_weight'], width, label='Emotion2Vec权重')
    plt.bar(x + width/2, emotion_weights['hub_weight'], width, label='HuBERT权重')
    
    plt.xlabel('情感类别')
    plt.ylabel('平均门控权重')
    plt.title('不同情感类别的门控权重分布')
    plt.xticks(x, emotion_weights['emotion_name'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_gate_weights.png'), dpi=300)
    plt.close()
    
    print(f"门控权重可视化已保存到: {output_dir}")
    
    # 保存详细数据
    emotion_weights.to_csv(os.path.join(output_dir, 'emotion_gate_weights.csv'), index=False)
    
    return emotion_weights
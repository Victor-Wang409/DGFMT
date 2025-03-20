"""
数据集模块
用于加载和处理情感数据
"""

import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

class EmotionDataset(torch.utils.data.Dataset):
    """
    情感数据集类，用于加载和处理情感数据
    """
    def __init__(self, emotion2vec_dir, hubert_dir, csv_path):
        """
        初始化数据集
        
        参数:
            emotion2vec_dir: emotion2vec特征目录
            hubert_dir: hubert特征目录
            csv_path: 标注CSV文件路径
        """
        self.df = pd.read_csv(csv_path)
        self.emotion2vec_dir = emotion2vec_dir
        self.hubert_dir = hubert_dir
 
        # 将情感标签映射到数值
        self.emotion_map = {
            'neu': 0, 
            'hap': 1,
            'ang': 2,
            'sad': 3,
            'sur': 4,
            'fea': 5,
            'dis': 6,
            'con': 7
        }

        self.vad_labels = []
        for vad_str in self.df['VAD_normalized']:
            vad = eval(vad_str)
            self.vad_labels.append(torch.tensor(vad, dtype=torch.float))
        
        self.emotion_labels = []
        for label in self.df['Label']:
            # 转换为one-hot向量
            label_idx = self.emotion_map[label]
            one_hot = torch.zeros(len(self.emotion_map))
            one_hot[label_idx] = 1
            self.emotion_labels.append(one_hot)

    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.df)
        
    def __getitem__(self, idx):
        """
        获取单个样本
        
        参数:
            idx: 样本索引
            
        返回:
            样本数据字典
        """
        row = self.df.iloc[idx]
        base_filename = os.path.splitext(row['FileName'])[0]
        
        # 加载两种特征
        emotion2vec_path = os.path.join(self.emotion2vec_dir, f"{base_filename}.npy")
        hubert_path = os.path.join(self.hubert_dir, f"{base_filename}.npy")
        
        emotion2vec_features = torch.from_numpy(np.load(emotion2vec_path)).float()
        hubert_features = torch.from_numpy(np.load(hubert_path)).float()
        
        # 使用插值来对齐特征长度
        target_len = max(emotion2vec_features.size(0), hubert_features.size(0))
        
        if emotion2vec_features.size(0) != target_len:
            emotion2vec_features = F.interpolate(
                emotion2vec_features.unsqueeze(0).unsqueeze(0),
                size=target_len,
                mode='linear',
                align_corners=False
            ).squeeze(0).squeeze(0)
            
        if hubert_features.size(0) != target_len:
            hubert_features = F.interpolate(
                hubert_features.unsqueeze(0).unsqueeze(0),
                size=target_len,
                mode='linear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        
        return {
            "id": row['FileName'],
            "emotion2vec_features": emotion2vec_features,
            "hubert_features": hubert_features,
            "labels": self.vad_labels[idx],
            "emotion_labels": self.emotion_labels[idx]
        }
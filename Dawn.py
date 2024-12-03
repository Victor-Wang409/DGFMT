import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import soundfile as sf
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from tqdm import tqdm
import os

class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits

def compute_ccc(preds, labels):
    """计算CCC值"""
    mean_pred = np.mean(preds)
    mean_label = np.mean(labels)
    
    var_pred = np.var(preds)
    var_label = np.var(labels)
    
    covariance = np.mean((preds - mean_pred) * (labels - mean_label))
    
    ccc = (2 * covariance) / (var_pred + var_label + (mean_pred - mean_label)**2)
    return ccc

def process_audio(audio_path, processor, model, device):
    """处理单个音频文件"""
    audio, sampling_rate = sf.read(audio_path)
    
    # 确保音频是单声道
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # 处理音频
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
    inputs = inputs.input_values.to(device)
    
    with torch.no_grad():
        _, logits = model(inputs)
    
    return logits.cpu().numpy()[0]

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型和处理器
    model_name = 'facebook/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name).to(device)
    model.eval()
    
    # 读取CSV文件
    df = pd.read_csv('./csv_files/IEMOCAP.csv')
    
    # 存储预测结果和真实标签
    predictions = []
    true_labels = []
    
    # 音频文件目录
    audio_dir = "/home/wangchenhao/Github/Dataset/IEMOCAP"  # 替换为实际音频文件目录
    
    # 批量处理音频文件
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = os.path.join(audio_dir, row['FileName'])
        if not os.path.exists(audio_path):
            continue
            
        # 获取预测值
        pred = process_audio(audio_path, processor, model, device)
        # 重新排列为 valence, arousal, dominance 顺序
        pred = pred[[2, 0, 1]]
        predictions.append(pred)
        
        # 获取真实标签
        true_label = eval(row['VAD_normalized'])
        true_labels.append(true_label)
    
    # 转换为numpy数组
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # 计算每个维度的CCC
    dimensions = ['Valence', 'Arousal', 'Dominance']
    for i, dim in enumerate(dimensions):
        ccc = compute_ccc(predictions[:, i], true_labels[:, i])
        print(f"{dim} CCC: {ccc:.4f}")
        
    # 计算平均CCC
    mean_ccc = np.mean([compute_ccc(predictions[:, i], true_labels[:, i]) 
                       for i in range(3)])
    print(f"Average CCC: {mean_ccc:.4f}")

if __name__ == "__main__":
    main()
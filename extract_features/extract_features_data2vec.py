"""
Data2Vec音频特征提取器
用于批量提取音频文件的data2vec特征并保存到本地
"""

import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from transformers import Data2VecAudioModel, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import argparse
import logging

class Data2VecFeatureExtractor:
    """
    Data2Vec音频特征提取器类
    """
    def __init__(self, model_name="facebook/data2vec-audio-large-960h", device=None):
        """
        初始化Data2Vec特征提取器
        
        参数:
            model_name: 模型名称或路径
            device: 计算设备
        """
        self.model_name = model_name
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logging.info(f"加载Data2Vec模型: {model_name}")
        logging.info(f"使用设备: {self.device}")
        
        # 加载模型和处理器
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Data2VecAudioModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # 设置特征维度
        self.feature_dim = self.model.config.hidden_size
        logging.info(f"特征维度: {self.feature_dim}")
    
    def extract_features(self, audio_file, layer_index=-1, return_all_hiddens=False):
        """
        从音频文件提取特征
        
        参数:
            audio_file: 音频文件路径
            layer_index: 提取特征的层索引，默认最后一层
            return_all_hiddens: 是否返回所有隐藏层特征
            
        返回:
            音频特征
        """
        try:
            # 加载音频
            audio, sample_rate = sf.read(audio_file)
            
            # 确保音频是单声道的
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # 处理音频
            inputs = self.processor(
                audio, 
                sampling_rate=sample_rate, 
                return_tensors="pt",
                padding=True
            )
            
            input_values = inputs.input_values.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device) if hasattr(inputs, "attention_mask") else None
            
            # 提取特征
            with torch.no_grad():
                outputs = self.model(
                    input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=return_all_hiddens
                )
            
            if return_all_hiddens:
                # 返回所有层的特征
                all_features = []
                for hidden_state in outputs.hidden_states:
                    features = hidden_state.squeeze(0).cpu().numpy()
                    all_features.append(features)
                return all_features
            else:
                # 只返回指定层的特征
                if layer_index == -1:
                    features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                else:
                    features = outputs.hidden_states[layer_index].squeeze(0).cpu().numpy()
                return features
                
        except Exception as e:
            logging.error(f"处理文件 {audio_file} 时出错: {str(e)}")
            return None

def process_batch(extractor, csv_file, audio_dir, output_dir, file_column='FileName'):
    """
    批量处理音频文件
    
    参数:
        extractor: Data2Vec特征提取器
        csv_file: CSV文件路径
        audio_dir: 音频文件目录
        output_dir: 输出目录
        file_column: CSV中文件名列
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    logging.info(f"已加载CSV文件: {csv_file}, 共有{len(df)}条记录")
    
    # 统计处理成功和失败的文件
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # 处理所有音频文件
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理音频文件"):
        filename = row[file_column]
        # 获取文件名（不含扩展名）
        base_filename = os.path.splitext(filename)[0]
        audio_file = os.path.join(audio_dir, filename)
        output_file = os.path.join(output_dir, f"{base_filename}.npy")
        
        # 如果输出文件已存在，则跳过
        if os.path.exists(output_file):
            logging.debug(f"跳过已处理的文件: {base_filename}")
            skipped_count += 1
            continue
        
        # 检查音频文件是否存在
        if not os.path.exists(audio_file):
            logging.warning(f"音频文件不存在: {audio_file}")
            failed_count += 1
            continue
        
        try:
            # 提取特征
            features = extractor.extract_features(audio_file)
            
            if features is not None:
                # 保存特征到本地
                np.save(output_file, features)
                success_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            logging.error(f"处理文件 {audio_file} 时出错: {str(e)}")
            failed_count += 1
    
    logging.info(f"处理完成: 成功={success_count}, 失败={failed_count}, 跳过={skipped_count}")
    return success_count, failed_count, skipped_count

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用Data2Vec模型提取音频特征')
    parser.add_argument('--model', type=str, default='facebook/data2vec-audio-large-960h', 
                        help='Data2Vec模型名称或路径')
    parser.add_argument('--csv_file', type=str, required=True, 
                        help='CSV文件路径，包含音频文件列表')
    parser.add_argument('--audio_dir', type=str, required=True, 
                        help='音频文件目录')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='特征输出目录')
    parser.add_argument('--file_column', type=str, default='FileName', 
                        help='CSV中的文件名列')
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, 'extraction.log'), 'w')
        ]
    )
    
    # 创建特征提取器
    extractor = Data2VecFeatureExtractor(model_name=args.model)
    
    # 批量处理
    success_count, failed_count, skipped_count = process_batch(
        extractor, 
        args.csv_file, 
        args.audio_dir, 
        args.output_dir,
        args.file_column
    )
    
    print(f"\n处理结果总结:")
    print(f"成功处理: {success_count}")
    print(f"处理失败: {failed_count}")
    print(f"已跳过: {skipped_count}")

if __name__ == '__main__':
    main()
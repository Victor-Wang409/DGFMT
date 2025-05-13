import torch
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import pandas as pd
import os
from tqdm import tqdm
import argparse
from pathlib import Path
import logging

class Wav2Vec2Extractor:
    def __init__(self, model_path):
        """初始化wav2vec2特征提取器"""
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {self.device}")
        
        # 加载模型和处理器
        self.processor = Wav2Vec2Processor.from_pretrained(model_path, local_files_only=True)
        self.model = Wav2Vec2Model.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.model.eval()
        
        # 打印模型特征维度
        self.feature_dim = self.model.config.hidden_size
        logging.info(f"特征维度: {self.feature_dim}")

    def extract_features(self, audio_path):
        """
        从音频文件提取wav2vec2特征
        Args:
            audio_path: 音频文件路径
        Returns:
            numpy.ndarray: wav2vec2特征
        """
        try:
            # 读取音频文件
            wav, sr = sf.read(audio_path)
            
            # 确保音频是单声道的
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)
                
            # 预处理音频
            inputs = self.processor(wav, sampling_rate=16000, return_tensors="pt")
            
            # 将输入移到设备，逐个处理避免全部进入GPU
            input_values = inputs['input_values'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device) if 'attention_mask' in inputs else None

            # 提取特征
            with torch.no_grad():
                if attention_mask is not None:
                    outputs = self.model(input_values, attention_mask=attention_mask)
                else:
                    outputs = self.model(input_values)
                
                # 立即将特征转移到CPU并转换为numpy数组，释放GPU内存
                features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                
                # 显式清理GPU缓存
                torch.cuda.empty_cache()
                
                return features
                
        except Exception as e:
            logging.error(f"处理文件 {audio_path} 时出错: {str(e)}")
            # 确保错误发生时也释放GPU内存
            torch.cuda.empty_cache()
            return None

def extract_and_save_features(args):
    """
    提取所有音频的特征并保存
    """
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, 'extraction.log'), 'w')
        ]
    )
    
    # 创建特征提取器
    extractor = Wav2Vec2Extractor(args.model_path)
    
    # 读取数据集信息
    df = pd.read_csv(args.csv_path)
    logging.info(f"已加载CSV文件: {args.csv_path}, 共有{len(df)}条记录")
    
    # 创建保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 记录特征信息
    feature_info = []
    
    # 统计处理成功和失败的文件
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # 提取并保存特征
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取特征中"):
        try:
            # 构建文件路径
            audio_path = os.path.join(args.audio_dir, row['FileName'])
            output_path = os.path.join(args.output_dir, f"{Path(row['FileName']).stem}.npy")
            
            # 检查是否已经处理过
            if os.path.exists(output_path) and not args.force:
                logging.debug(f"跳过已处理的文件: {Path(row['FileName']).stem}")
                skipped_count += 1
                continue
            
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                logging.warning(f"音频文件不存在: {audio_path}")
                failed_count += 1
                continue
            
            # 提取特征
            features = extractor.extract_features(audio_path)
            
            if features is not None:
                # 保存特征
                np.save(output_path, features)
                
                # 记录信息
                feature_info.append({
                    'file_name': row['FileName'],
                    'feature_path': output_path,
                    'feature_shape': features.shape
                })
                
                success_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            logging.error(f"处理文件 {row['FileName']} 时出错: {e}")
            failed_count += 1
        
        # 每处理10个样本清理一次GPU内存，防止内存累积
        if idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # 保存特征信息到CSV
    info_df = pd.DataFrame(feature_info)
    info_df.to_csv(os.path.join(args.output_dir, 'feature_info.csv'), index=False)
    
    logging.info(f"处理完成: 成功={success_count}, 失败={failed_count}, 跳过={skipped_count}")
    print(f"\n处理结果总结:")
    print(f"成功处理: {success_count}")
    print(f"处理失败: {failed_count}")
    print(f"已跳过: {skipped_count}")

def main():
    parser = argparse.ArgumentParser(description='Extract wav2vec2 features from audio files')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to the annotation CSV file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to local wav2vec2-base model directory')
    parser.add_argument('--output_dir', type=str, default='wav2vec2_features',
                        help='Directory to save extracted features')
    parser.add_argument('--force', action='store_true',
                        help='Force re-extract features even if they already exist')
    
    args = parser.parse_args()
    extract_and_save_features(args)

if __name__ == '__main__':
    main()
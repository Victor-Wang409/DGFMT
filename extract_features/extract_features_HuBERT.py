import torch
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor, HubertModel
import os
from tqdm import tqdm
import argparse
from pathlib import Path
import librosa

class HubertExtractor:
    def __init__(self, hubert_path):
        """
        初始化HuBERT特征提取器
        Args:
            hubert_path: HuBERT模型本地路径
        """
        # 加载本地HuBERT处理器和模型
        self.processor = Wav2Vec2Processor.from_pretrained(hubert_path, local_files_only=True)
        self.model = HubertModel.from_pretrained(hubert_path, local_files_only=True).cuda()
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, audio_path):
        """
        从音频文件提取HuBERT特征
        Args:
            audio_path: 音频文件路径
        Returns:
            numpy.ndarray: HuBERT特征
        """
        # 读取音频文件
        wav, sr = sf.read(audio_path)
        
        # 重采样到16kHz(如果需要)
        if sr != 16000:
            print(f"Warning: Resampling audio from {sr}Hz to 16000Hz")
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            
        # 确保音频是单声道
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)

        # 预处理音频
        inputs = self.processor(wav, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        # 提取特征
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state.squeeze(0).cpu().numpy()

        return features

def extract_and_save_features(args):
    """
    提取所有音频的特征并保存
    """
    # 创建特征提取器
    extractor = HubertExtractor(args.hubert_path)
    
    # 获取所有.wav文件
    wav_files = list(Path(args.audio_dir).rglob("*.wav"))
    
    # 创建保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 记录特征信息
    feature_info = []
    
    # 提取并保存特征
    for wav_path in tqdm(wav_files, desc="Extracting features"):
        try:
            output_path = os.path.join(args.output_dir, f"{wav_path.stem}.npy")
            
            # 检查是否已经处理过
            if os.path.exists(output_path) and not args.force:
                print(f"Skipping {wav_path.name}: features already exist")
                continue
            
            # 提取特征
            features = extractor.extract_features(str(wav_path))
            
            # 保存特征
            np.save(output_path, features)
            
            # 记录信息
            feature_info.append({
                'file_name': wav_path.name,
                'feature_path': output_path,
                'feature_shape': features.shape
            })
            
        except Exception as e:
            print(f"Error processing {wav_path.name}: {e}")
    
    # 保存特征信息到CSV
    import pandas as pd
    info_df = pd.DataFrame(feature_info)
    info_df.to_csv('feature_info.csv', index=False)
    
    print(f"\nFeature extraction completed!")
    print(f"Total processed files: {len(feature_info)}")
    print(f"Features saved to: {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Extract HuBERT features from audio files')
    parser.add_argument('--audio_dir', type=str, default='/home/wangchenhao/Github/Dataset/EMOVO',
                        help='Directory containing .wav files')
    parser.add_argument('--hubert_path', type=str, default='./facebook/hubert-large-ls960-ft',
                        help='Path to local HuBERT model directory')
    parser.add_argument('--output_dir', type=str, default='./hubert_large_features',
                        help='Directory to save extracted features')
    parser.add_argument('--force', action='store_true',
                        help='Force re-extract features even if they already exist')
    
    args = parser.parse_args()
    extract_and_save_features(args)

if __name__ == '__main__':
    main()
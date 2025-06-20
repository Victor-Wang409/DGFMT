import torch
import numpy as np
from transformers import Wav2Vec2Processor, HubertModel
import os
from tqdm import tqdm
import argparse
from pathlib import Path
import librosa

class HubertExtractor:
    def __init__(self, hubert_path):
        """初始化HuBERT特征提取器"""
        self.processor = Wav2Vec2Processor.from_pretrained(hubert_path, local_files_only=True)
        self.model = HubertModel.from_pretrained(hubert_path, local_files_only=True).cuda()
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, audio_path):
        """从音频文件提取HuBERT特征"""
        # 直接使用librosa加载音频，自动处理多种采样率
        wav, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # 调整采样率以适应模型需求
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            
        # 预处理音频并提取特征
        inputs = self.processor(wav, sampling_rate=16000, return_tensors="pt").to('cuda')
        outputs = self.model(**inputs)
        
        return outputs.last_hidden_state.squeeze(0).cpu().numpy()

def extract_and_save_features(args):
    """提取所有音频的特征并保存"""
    extractor = HubertExtractor(args.hubert_path)
    wav_files = list(Path(args.audio_dir).rglob("*.wav"))
    os.makedirs(args.output_dir, exist_ok=True)
    
    feature_info = []
    
    for wav_path in tqdm(wav_files, desc="Extracting features"):
        try:
            output_path = os.path.join(args.output_dir, f"{wav_path.stem}.npy")
            
            # 跳过已处理文件
            if os.path.exists(output_path) and not args.force:
                continue
            
            # 提取并保存特征
            features = extractor.extract_features(str(wav_path))
            np.save(output_path, features)
            
            # 记录信息
            feature_info.append({
                'file_name': wav_path.name,
                'feature_shape': features.shape
            })
            
        except Exception as e:
            print(f"Error processing {wav_path.name}: {e}")
    
    # 保存特征信息
    if feature_info:
        import pandas as pd
        pd.DataFrame(feature_info).to_csv(os.path.join(args.output_dir, 'feature_info.csv'), index=False)
    
    print(f"Feature extraction completed! Processed {len(feature_info)} files.")

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
    
    extract_and_save_features(parser.parse_args())

if __name__ == '__main__':
    main()
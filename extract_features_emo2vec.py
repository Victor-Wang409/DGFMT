import torch
import torch.nn.functional as F
import fairseq
import soundfile as sf
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse
from pathlib import Path

class Wav2vecExtractor:
    def __init__(self, model_path, checkpoint_path):
        """
        初始化Wav2vec特征提取器
        Args:
            model_path: emotion2vec模型路径
            checkpoint_path: 检查点路径
        """
        # 初始化wav2vec2
        class UserDirModule:
            def __init__(self, user_dir):
                self.user_dir = user_dir
                
        model_path = UserDirModule(model_path)
        fairseq.utils.import_user_module(model_path)
        
        # 加载模型和配置
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path],
        )
        self.model = models[0].eval().cuda()
        self.task = task

    @torch.no_grad()
    def extract_features(self, audio_path):
        """
        从音频文件提取wav2vec特征
        Args:
            audio_path: 音频文件路径
        Returns:
            numpy.ndarray: wav2vec特征
        """
        # 读取音频文件
        wav, sr = sf.read(audio_path)
        assert sr == 16000, f"Sample rate must be 16kHz, got {sr}Hz"
        assert len(wav.shape) == 1, f"Audio must be mono channel, got shape {wav.shape}"
            
        # 预处理音频
        source = torch.from_numpy(wav).float().cuda()
        if self.task.cfg.normalize:
            source = F.layer_norm(source, source.shape)
        source = source.view(1, -1)
            
        # 提取特征
        features = self.model.extract_features(
            source=source,
            padding_mask=None,
            mask=False
        )
            
        # 获取最后一层的特征
        x = features['x']
        return x.squeeze(0).cpu().numpy()

def extract_and_save_features(args):
    """提取所有音频的特征并保存"""
    # 创建特征提取器
    extractor = Wav2vecExtractor(args.model_path, args.checkpoint_path)
    
    # 读取数据集信息
    df = pd.read_csv(args.csv_path)
    
    # 创建保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 记录特征信息
    feature_info = []
    
    # 提取并保存特征
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:
            # 构建文件路径
            audio_path = os.path.join(args.audio_dir, row['FileName'])
            output_path = os.path.join(args.output_dir, f"{Path(row['FileName']).stem}.npy")
            
            # 检查是否已经处理过
            if os.path.exists(output_path) and not args.force:
                print(f"Skipping {row['FileName']}: features already exist")
                continue
            
            # 提取特征
            features = extractor.extract_features(audio_path)
            
            # 保存特征
            np.save(output_path, features)
            
            # 记录信息
            feature_info.append({
                'file_name': row['FileName'],
                'feature_path': output_path,
                'feature_shape': features.shape,
                'original_label': row['VAD_normalized']
            })
            
        except Exception as e:
            print(f"Error processing {row['FileName']}: {str(e)}")
    
    # 保存特征信息到CSV
    info_df = pd.DataFrame(feature_info)
    info_df.to_csv(os.path.join(args.output_dir, 'feature_info.csv'), index=False)
    
    print(f"\nFeature extraction completed!")
    print(f"Total processed files: {len(feature_info)}")
    print(f"Features saved to: {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Extract wav2vec features from audio files')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to the annotation CSV file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to wav2vec model directory')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to wav2vec checkpoint file')
    parser.add_argument('--output_dir', type=str, default='wav2vec_features',
                        help='Directory to save extracted features')
    parser.add_argument('--force', action='store_true',
                        help='Force re-extract features even if they already exist')
    
    args = parser.parse_args()
    extract_and_save_features(args)

if __name__ == '__main__':
    main()
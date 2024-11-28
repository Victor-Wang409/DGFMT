import torch
import soundfile as sf
import torch.nn.functional as F
import fairseq
import argparse
import os
import numpy as np
from torch import nn
import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedModel, PretrainedConfig

# VAD模型配置类
class VADConfig(PretrainedConfig):
   def __init__(self, input_dim=768, hidden_dim=256, **kwargs):
       super().__init__(**kwargs)
       self.input_dim = input_dim
       self.hidden_dim = hidden_dim

# 特征提取器类
class Emotion2vecExtractor:
   def __init__(self, model_path, checkpoint_path):
       class UserDirModule:
           def __init__(self, user_dir):
               self.user_dir = user_dir
               
       model_path = UserDirModule(model_path)
       fairseq.utils.import_user_module(model_path)
       
       model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
           [checkpoint_path],
       )
       self.model = model[0].eval().cuda()
       self.task = task

   def extract_features(self, audio_path):
       wav, sr = sf.read(audio_path)
       assert sr == 16000 and len(wav.shape) == 1
       
       with torch.no_grad():
           source = torch.from_numpy(wav).float().cuda()
           if self.task.cfg.normalize:
               source = F.layer_norm(source, source.shape)
           source = source.view(1, -1)
           
           res = self.model.extract_features(source, padding_mask=None, remove_extra_tokens=True)
           return res['x'].squeeze(0).cpu()

class VADModel(PreTrainedModel):
    config_class = VADConfig
    base_model_prefix = "vad"
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.pre_net = nn.Linear(config.input_dim, config.hidden_dim)
        self.post_net = nn.Linear(config.hidden_dim, 3)
        self.activate = nn.ReLU()
    
    def forward(self, x, padding_mask=None):
        x = self.activate(self.pre_net(x))
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).float())
            x = x.sum(dim=1) / (1 - padding_mask.float()).sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        x = torch.sigmoid(self.post_net(x))
        return x
    
    def save_pretrained(self, save_directory):
        # 确保目录存在
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存配置
        self.config.save_pretrained(save_directory)
        
        # 保存模型权重
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        
    @classmethod
    def from_pretrained(cls, pretrained_model_path, *model_args, **kwargs):
        config = VADConfig.from_pretrained(pretrained_model_path)
        model = cls(config)
        state_dict = torch.load(os.path.join(pretrained_model_path, "pytorch_model.bin"))
        model.load_state_dict(state_dict)
        return model

def compute_dimension_ccc(preds, labels):
   """计算单个维度的CCC值"""
   preds_mean = np.mean(preds)
   labels_mean = np.mean(labels)
   
   preds_var = np.mean((preds - preds_mean) ** 2)
   labels_var = np.mean((labels - labels_mean) ** 2)
   
   covar = np.mean((preds - preds_mean) * (labels - labels_mean))
   
   ccc = 2 * covar / (preds_var + labels_var + (preds_mean - labels_mean) ** 2 + 1e-8)
   return ccc

def predict_vad(audio_path, model, feature_extractor, device):
   """对单个音频文件进行VAD预测"""
   features = feature_extractor.extract_features(audio_path)
   features = features.unsqueeze(0).to(device)
   padding_mask = torch.zeros(1, features.size(1)).bool().to(device)
   
   model.eval()
   with torch.no_grad():
       predictions = model(features, padding_mask)
   
   return predictions[0].cpu().numpy()

def main():
   parser = argparse.ArgumentParser(description='VAD Prediction with CCC Calculation')
   parser.add_argument('--audio_dir', type=str, required=True, help='Directory containing audio files')
   parser.add_argument('--model_path', type=str, required=True, help='Path to emotion2vec model')
   parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to emotion2vec checkpoint')
   parser.add_argument('--vad_model_path', type=str, required=True, help='Path to trained VAD model')
   parser.add_argument('--csv_path', type=str, required=True, help='Path to annotation CSV file')
   parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
   
   args = parser.parse_args()
   
   # 创建输出目录
   os.makedirs(args.output_dir, exist_ok=True)
   
   # 设置设备
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # 初始化特征提取器
   feature_extractor = Emotion2vecExtractor(args.model_path, args.checkpoint_path)
   # 加载VAD模型
   model = VADModel.from_pretrained(args.vad_model_path)
   model.to(device)
   model.eval()
   
   # 读取标注文件
   df = pd.read_csv(args.csv_path)
   
   # 准备结果存储
   results = []
   all_preds = []
   all_labels = []
   
   # 处理每个音频文件
   print("Processing audio files...")
   for _, row in tqdm(df.iterrows(), total=len(df)):
       audio_file = os.path.join(args.audio_dir, f"{row['FileName']}")
       
       try:
           # 获取预测值
           pred_vad = predict_vad(audio_file, model, feature_extractor, device)
           
           # 获取真实标签
           true_vad = eval(row['VAD_normalized'])
           
           results.append({
               'file_name': row['FileName'],
               'pred_valence': pred_vad[0],
               'pred_arousal': pred_vad[1],
               'pred_dominance': pred_vad[2],
               'true_valence': true_vad[0],
               'true_arousal': true_vad[1],
               'true_dominance': true_vad[2]
           })
           
           all_preds.append(pred_vad)
           all_labels.append(true_vad)
           
       except Exception as e:
           print(f"Error processing {audio_file}: {str(e)}")
   
   # 转换为numpy数组
   all_preds = np.array(all_preds)
   all_labels = np.array(all_labels)
   
   # 计算每个维度的CCC
   ccc_v = compute_dimension_ccc(all_preds[:, 0], all_labels[:, 0])
   ccc_a = compute_dimension_ccc(all_preds[:, 1], all_labels[:, 1])
   ccc_d = compute_dimension_ccc(all_preds[:, 2], all_labels[:, 2])
   avg_ccc = (ccc_v + ccc_a + ccc_d) / 3
   
   # 保存详细预测结果
   results_df = pd.DataFrame(results)
   results_df.to_csv(os.path.join(args.output_dir, 'detailed_predictions.csv'), index=False)
   
   # 保存CCC结果
   with open(os.path.join(args.output_dir, 'ccc_results.txt'), 'w') as f:
       f.write("CCC Results:\n")
       f.write(f"Valence CCC: {ccc_v:.4f}\n")
       f.write(f"Arousal CCC: {ccc_a:.4f}\n")
       f.write(f"Dominance CCC: {ccc_d:.4f}\n")
       f.write(f"Average CCC: {avg_ccc:.4f}\n")
   
   # 打印结果
   print("\nCCC Results:")
   print(f"Valence CCC: {ccc_v:.4f}")
   print(f"Arousal CCC: {ccc_a:.4f}")
   print(f"Dominance CCC: {ccc_d:.4f}")
   print(f"Average CCC: {avg_ccc:.4f}")
   print(f"\nResults saved in {args.output_dir}")

if __name__ == '__main__':
   main()
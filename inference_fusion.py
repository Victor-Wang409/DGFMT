import torch
import numpy as np
import pandas as pd
import argparse
import os
import json
from tqdm import tqdm
from pathlib import Path
from model_fusion import VADConfig, EmotionModel, FeatureFusion, MultiGatedFusion

def predict_vad(emotion2vec_features, hubert_features, model, device, padding_mask=None):
    """对单个样本进行VAD预测"""
    model.eval()
    with torch.no_grad():
        emotion2vec_features = torch.from_numpy(emotion2vec_features).float().unsqueeze(0).to(device)
        hubert_features = torch.from_numpy(hubert_features).float().unsqueeze(0).to(device)
        if padding_mask is not None:
            padding_mask = torch.from_numpy(padding_mask).bool().unsqueeze(0).to(device)
            
        outputs = model(
            emotion2vec_features=emotion2vec_features,
            hubert_features=hubert_features,
            padding_mask=padding_mask,
            mode="eval"
        )
    return outputs['vad'][0].cpu().numpy(), outputs.get('emotion', None)

def compute_dimension_ccc(preds, labels):
    """计算单个维度的CCC值"""
    preds_mean = np.mean(preds)
    labels_mean = np.mean(labels)
    
    preds_var = np.mean((preds - preds_mean) ** 2)
    labels_var = np.mean((labels - labels_mean) ** 2)
    
    covar = np.mean((preds - preds_mean) * (labels - labels_mean))
    
    ccc = 2 * covar / (preds_var + labels_var + (preds_mean - labels_mean) ** 2 + 1e-8)
    return ccc

def evaluate_model(model_path, emotion2vec_dir, hubert_dir, df, device):
    """评估单个模型的性能"""
    # 加载模型配置和训练参数
    try:
        training_state = torch.load(os.path.join(model_path, 'training_state.bin'), map_location=device)
        training_args = training_state['training_args']
        
        # 从保存的训练参数中获取配置
        config = VADConfig(
            use_hdgf=training_args.get('use_hdgf', False),
            use_cl=training_args.get('use_cl', False),
            temperature=training_args.get('temperature', 0.07)
        )
    except:
        # 如果无法加载训练状态，使用默认配置
        config = VADConfig(use_hdgf=True, use_cl=True)
    
    # 初始化模型
    model = EmotionModel(config)
    
    # 加载模型权重
    state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_emotion_preds = []
    all_emotion_labels = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['FileName']
        emotion2vec_path = os.path.join(emotion2vec_dir, f"{Path(filename).stem}.npy")
        hubert_path = os.path.join(hubert_dir, f"{Path(filename).stem}.npy")
        
        try:
            emotion2vec_features = np.load(emotion2vec_path)
            hubert_features = np.load(hubert_path)
            
            pred_vad, pred_emotion = predict_vad(emotion2vec_features, hubert_features, model, device)
            true_vad = eval(row['VAD_normalized'])
            
            all_preds.append(pred_vad)
            all_labels.append(true_vad)
            
            if pred_emotion is not None:
                emotion_map = {'N': 0, 'H': 1, 'S': 2, 'A': 3}
                true_emotion = emotion_map.get(row['Label'], 0)
                pred_emotion = pred_emotion.argmax(dim=1)[0].cpu().numpy()
                all_emotion_preds.append(pred_emotion)
                all_emotion_labels.append(true_emotion)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    ccc_v = compute_dimension_ccc(all_preds[:, 0], all_labels[:, 0])
    ccc_a = compute_dimension_ccc(all_preds[:, 1], all_labels[:, 1])
    ccc_d = compute_dimension_ccc(all_preds[:, 2], all_labels[:, 2])
    avg_ccc = (ccc_v + ccc_a + ccc_d) / 3
    
    metrics = {
        'ccc_v': ccc_v,
        'ccc_a': ccc_a,
        'ccc_d': ccc_d,
        'avg_ccc': avg_ccc  # 确保添加 avg_ccc
    }
    
    if len(all_emotion_preds) > 0:
        emotion_acc = np.mean(np.array(all_emotion_preds) == np.array(all_emotion_labels))
        metrics['emotion_acc'] = emotion_acc
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='VAD and Emotion Prediction with CCC Calculation')
    parser.add_argument('--emotion2vec_dir', type=str, required=True,
                        help='Directory containing emotion2vec features')
    parser.add_argument('--hubert_dir', type=str, required=True,
                        help='Directory containing hubert features')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to annotation CSV file')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save results')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Directory to model file')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df = pd.read_csv(args.csv_path)
    
    try:
        metrics = evaluate_model(args.model_path, args.emotion2vec_dir, args.hubert_dir, df, device)
        
        print(f"Valence CCC: {metrics['ccc_v']:.4f}")
        print(f"Arousal CCC: {metrics['ccc_a']:.4f}")
        print(f"Dominance CCC: {metrics['ccc_d']:.4f}")
        print(f"Average CCC: {metrics['avg_ccc']:.4f}")
        if 'emotion_acc' in metrics:
            print(f"Emotion Accuracy: {metrics['emotion_acc']:.4f}")
        
    except Exception as e:
        print(f"Error evaluating {model_dir}: {str(e)}")


if __name__ == '__main__':
    main()
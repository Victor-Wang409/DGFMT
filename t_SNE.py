import torch
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from model import (
    VADConfig,
    VADModelWithGating
)

def interpolate_features(features, target_len):
    if features.shape[0] == target_len:
        return features
        
    features = torch.from_numpy(features).float().transpose(0, 1).unsqueeze(0)
    features = torch.nn.functional.interpolate(
        features,
        size=target_len,
        mode='linear',
        align_corners=False
    )
    features = features.squeeze(0).transpose(0, 1).numpy()
    
    return features

def extract_pooled_features(emotion2vec_features, hubert_features, model, device):
    model.eval()
    with torch.no_grad():
        emotion2vec_features = torch.from_numpy(emotion2vec_features).float().unsqueeze(0).to(device)
        hubert_features = torch.from_numpy(hubert_features).float().unsqueeze(0).to(device)
        padding_mask = torch.zeros(1, emotion2vec_features.size(1)).bool().to(device)
        
        _, _, pooled_features = model(emotion2vec_features, hubert_features, padding_mask)
        
    return pooled_features[0].cpu().numpy()

def preprocess_features(features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled

def plot_tsne(features, labels, label_names, output_dir, perplexity=30, n_iter=1000):
    print("Preprocessing features...")
    features_preprocessed = preprocess_features(features)
    
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        learning_rate='auto',
        init='pca',
        metric='cosine',
        random_state=42
    )
    tsne_results = tsne.fit_transform(features_preprocessed)
    
    print("Creating visualization...")
    plt.figure(figsize=(12, 8))
    
    # 设置图表样式
    plt.rcParams['figure.facecolor'] = 'white'
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 为每种情感定义固定的颜色和标记
    emotion_styles = {
        'neu': {'color': '#FFCCD4', 'marker': 'v', 'label': 'Neutral'},    # 更鲜艳的粉色
        'hap': {'color': '#FF3333', 'marker': '^', 'label': 'Happy'},      # 更鲜艳的红色
        'ang': {'color': '#0066CC', 'marker': '>', 'label': 'Angry'},      # 更鲜艳的蓝色
        'sad': {'color': '#7AB5FF', 'marker': '<', 'label': 'Sad'},        # 更鲜艳的浅蓝色
    }
    
    # 绘制散点图
    print(f"Unique labels in data: {set(labels)}")
    
    for label in set(labels):
        if label not in emotion_styles:
            print(f"Warning: Unknown emotion label: {label}")
            continue
            
        style = emotion_styles[label]
        mask = np.array(labels) == label
        points_count = np.sum(mask)
        print(f"Label {label}: {points_count} points")
        
        if points_count > 0:
            plt.scatter(
                tsne_results[mask, 0],
                tsne_results[mask, 1],
                c=style['color'],
                label=style['label'],
                alpha=0.7,
                s=30,
                marker=style['marker'],
                edgecolors='black',
                linewidth=0.5
            )
    
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.10),
        ncol=4,
        fontsize=12,
        frameon=False,
        fancybox=True,
        shadow=True
    )
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'tsne_visualization.png'),
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    
    print(f"Visualization saved to: {os.path.join(output_dir, 'tsne_visualization.png')}")
    plt.close()

def extract_features(model_path, emotion2vec_dir, hubert_dir, df, device):
    config = VADConfig(
        emotion2vec_dim=1024,
        hubert_dim=1024,
        hidden_dim=1024,
        num_hidden_layers=4
    )
    model = VADModelWithGating(config).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    
    all_features = []
    all_emotions = []
    
    print("Extracting features...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['FileName']
        base_filename = Path(filename).stem
        
        try:
            emotion2vec_path = os.path.join(emotion2vec_dir, f"{base_filename}.npy")
            hubert_path = os.path.join(hubert_dir, f"{base_filename}.npy")
            
            emotion2vec_features = np.load(emotion2vec_path)
            hubert_features = np.load(hubert_path)
            
            target_len = max(emotion2vec_features.shape[0], hubert_features.shape[0])
            emotion2vec_features = interpolate_features(emotion2vec_features, target_len)
            hubert_features = interpolate_features(hubert_features, target_len)
            
            pooled_features = extract_pooled_features(emotion2vec_features, hubert_features, model, device)
            
            all_features.append(pooled_features)
            all_emotions.append(row['Label'])
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    return np.array(all_features), all_emotions

def main():
    # Set warnings to ignore
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    parser = argparse.ArgumentParser(description='Feature Visualization using t-SNE')
    parser.add_argument('--emotion2vec_dir', type=str, required=True,
                        help='Directory containing emotion2vec features')
    parser.add_argument('--hubert_dir', type=str, required=True,
                        help='Directory containing hubert features')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint file (.pt)')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to annotation CSV file')
    parser.add_argument('--output_dir', type=str, default='visualization_results',
                        help='Directory to save visualization')
    parser.add_argument('--perplexity', type=float, default=40,
                        help='Perplexity parameter for t-SNE (default: 40)')
    parser.add_argument('--n_iter', type=int, default=3000,
                        help='Number of iterations for t-SNE (default: 3000)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} samples from annotation file")
    
    features, emotions = extract_features(
        args.model_path,
        args.emotion2vec_dir,
        args.hubert_dir,
        df,
        device
    )
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Unique emotions in dataset: {set(emotions)}")
    
    # English label names - ensure these match your actual labels
    label_names = {
        'N': 'Neutral',
        'F': 'Happy',
        'W': 'Angry', 
        'T': 'Sad',
        'L': "Boredom",
        'E': "Disgust",
        'A': "Anxiety/Fear"
    }
    
    plot_tsne(
        features,
        emotions,
        label_names,
        args.output_dir,
        args.perplexity,
        args.n_iter
    )

if __name__ == '__main__':
    main()
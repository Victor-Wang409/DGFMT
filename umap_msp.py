import torch
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap  # 导入UMAP库
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
        try:
            # 限制序列长度以减少内存使用
            max_seq_len = 500  # 根据实际情况调整
            if emotion2vec_features.shape[0] > max_seq_len:
                emotion2vec_features = emotion2vec_features[:max_seq_len]
                hubert_features = hubert_features[:max_seq_len]
            
            emotion2vec_features = torch.from_numpy(emotion2vec_features).float().unsqueeze(0).to(device)
            hubert_features = torch.from_numpy(hubert_features).float().unsqueeze(0).to(device)
            padding_mask = torch.zeros(1, emotion2vec_features.size(1)).bool().to(device)
            
            _, _, pooled_features = model(emotion2vec_features, hubert_features, padding_mask)
            result = pooled_features[0].cpu().numpy()
            
            # 立即释放GPU内存
            del emotion2vec_features, hubert_features, padding_mask, pooled_features
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
            return result
        except RuntimeError as e:
            print(f"运行时错误，尝试在CPU上进行计算: {e}")
            # 如果GPU内存不足，回退到CPU
            device_cpu = torch.device("cpu")
            model = model.to(device_cpu)
            
            emotion2vec_features = torch.from_numpy(emotion2vec_features).float().unsqueeze(0)
            hubert_features = torch.from_numpy(hubert_features).float().unsqueeze(0)
            padding_mask = torch.zeros(1, emotion2vec_features.size(1)).bool()
            
            _, _, pooled_features = model(emotion2vec_features, hubert_features, padding_mask)
            return pooled_features[0].numpy()
            
            # 注意: 这里不将模型移回GPU，因为这会在下一个样本时自动处理

def preprocess_features(features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled

def plot_umap(features, labels, label_names, output_dir, n_neighbors=15, min_dist=0.1, n_components=2):
    """
    使用UMAP降维并可视化特征
    
    参数:
        features: 需要降维的特征
        labels: 类别标签
        label_names: 标签名称映射
        output_dir: 输出目录
        n_neighbors: UMAP参数，控制局部邻域大小
        min_dist: UMAP参数，控制嵌入点的最小距离
        n_components: 降维后的维度数
    """
    print("预处理特征...")
    features_preprocessed = preprocess_features(features)
    
    print("执行UMAP降维...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric='cosine',
        random_state=42
    )
    umap_results = reducer.fit_transform(features_preprocessed)
    
    print("创建可视化图像...")
    plt.figure(figsize=(12, 8))
    
    # 设置图表样式
    plt.rcParams['figure.facecolor'] = 'white'
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 为每种情感定义固定的颜色和标记
    emotion_styles = {
        'neu': {'color': '#FFCCD4', 'marker': 'v', 'label': 'Neutral'},    # 更鲜艳的粉色
        'hap': {'color': '#FF3333', 'marker': '>', 'label': 'Happy'},      # 更鲜艳的红色
        'ang': {'color': '#0066CC', 'marker': '^', 'label': 'Angry'},      # 更鲜艳的蓝色
        'sad': {'color': '#7AB5FF', 'marker': '<', 'label': 'Sad'},        # 更鲜艳的浅蓝色
    }
    
    # 绘制散点图
    print(f"数据中的唯一标签: {set(labels)}")
    
    for label in set(labels):
        if label not in emotion_styles:
            print(f"警告: 未知的情感标签: {label}")
            continue
            
        style = emotion_styles[label]
        mask = np.array(labels) == label
        points_count = np.sum(mask)
        print(f"标签 {label}: {points_count} 个点")
        
        if points_count > 0:
            plt.scatter(
                umap_results[mask, 0],
                umap_results[mask, 1],
                c=style['color'],
                label=style['label'],
                alpha=0.7,
                s=30,
                marker=style['marker'],
                edgecolors='black',
                linewidth=0.5
            )
    
    # plt.title('UMAP 情感特征可视化', fontsize=15)
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.10),
        ncol=4,
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'umap_visualization.png'),
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    
    print(f"可视化图像已保存至: {os.path.join(output_dir, 'umap_visualization.png')}")
    plt.close()

def extract_features(model_path, emotion2vec_dir, hubert_dir, df, device):
    config = VADConfig(
        emotion2vec_dim=1024,
        hubert_dim=1024,
        hidden_dim=1024,
        num_hidden_layers=4
    )
    
    # 尝试使用CPU，如果GPU内存不足
    try:
        model = VADModelWithGating(config).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except RuntimeError as e:
        print(f"GPU内存不足，切换到CPU: {e}")
        device = torch.device("cpu")
        model = VADModelWithGating(config).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint)
    model.eval()
    
    all_features = []
    all_emotions = []
    
    # 计算批处理大小 - 根据可用内存调整
    batch_size = 1
    
    print("提取特征...")
    # 分批处理以节省内存
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i+batch_size]
        
        for _, row in batch_df.iterrows():
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
                print(f"处理 {filename} 时出错: {str(e)}")
                continue
                
        # 定期清理CUDA缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return np.array(all_features), all_emotions

def main():
    # 忽略警告
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    parser = argparse.ArgumentParser(description='使用UMAP进行特征可视化')
    parser.add_argument('--emotion2vec_dir', type=str, required=True,
                        help='emotion2vec特征目录')
    parser.add_argument('--hubert_dir', type=str, required=True,
                        help='hubert特征目录')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型检查点文件路径 (.pt)')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='标注CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='visualization_results',
                        help='保存可视化结果的目录')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='UMAP的n_neighbors参数 (默认: 15)')
    parser.add_argument('--min_dist', type=float, default=0.1,
                        help='UMAP的min_dist参数 (默认: 0.1)')
    parser.add_argument('--use_cpu', action='store_true', 
                        help='强制使用CPU进行计算（即使有可用的GPU）')
    parser.add_argument('--sample_size', type=int, default=0,
                        help='随机采样数量，设置为0则使用全部数据 (默认: 0)')
    parser.add_argument('--test_only', action='store_true',
                        help='只使用Split_Set列为Test的数据 (默认: False)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 根据参数决定使用CPU还是GPU
    device = torch.device("cpu" if args.use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"使用设备: {device}")
    
    df = pd.read_csv(args.csv_path)
    print(f"从标注文件加载了 {len(df)} 个样本")
    
    # 只选择Split_Set列为Test的数据
    if args.test_only and 'Split_Set' in df.columns:
        test_df = df[df['Split_Set'] == 'Test']
        if len(test_df) > 0:
            print(f"只选择Test集，包含 {len(test_df)} 个样本")
            df = test_df
        else:
            print("警告: 在Split_Set列中未找到Test数据，将使用全部数据")
    elif args.test_only:
        print("警告: 未找到Split_Set列，将使用全部数据")
    
    # 如果指定了样本大小，则随机采样
    if args.sample_size > 0 and args.sample_size < len(df):
        print(f"随机采样 {args.sample_size} 个样本进行可视化")
        df = df.sample(args.sample_size, random_state=42)
    
    try:
        features, emotions = extract_features(
            args.model_path,
            args.emotion2vec_dir,
            args.hubert_dir,
            df,
            device
        )
    except Exception as e:
        print(f"提取特征时出错: {e}")
        print("尝试使用CPU重新运行...")
        device = torch.device("cpu")
        features, emotions = extract_features(
            args.model_path,
            args.emotion2vec_dir,
            args.hubert_dir,
            df,
            device
        )
    
    print(f"提取的特征形状: {features.shape}")
    print(f"数据集中的唯一情感类别: {set(emotions)}")
    
    # 英文标签名称 - 确保这些与你的实际标签匹配
    label_names = {
        'N': 'Neutral',
        'F': 'Happy',
        'W': 'Angry', 
        'T': 'Sad',
        'L': "Boredom",
        'E': "Disgust",
        'A': "Anxiety/Fear"
    }
    
    plot_umap(
        features,
        emotions,
        label_names,
        args.output_dir,
        args.n_neighbors,
        args.min_dist
    )

if __name__ == '__main__':
    main()
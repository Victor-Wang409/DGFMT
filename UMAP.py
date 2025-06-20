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

# 导入模型定义
from model import (
    VADConfig,
    VADModelWithGating
)

def interpolate_features(features, target_len):
    """插值函数，用于对齐特征长度"""
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

def extract_pooled_features(features_dict, model, device):
    """
    提取模型的池化特征
    
    参数:
        features_dict: 特征字典，包含各种特征类型
        model: VAD模型
        device: 设备
        
    返回:
        池化后的特征表示
    """
    model.eval()
    with torch.no_grad():
        try:
            # 限制序列长度以减少内存使用
            max_seq_len = 500  # 根据实际情况调整
            
            # 准备特征字典，转换为tensor并移动到设备
            processed_features = {}
            min_len = float('inf')
            
            # 首先找到最短的序列长度
            for feat_type, feat_data in features_dict.items():
                if feat_data.shape[0] < min_len:
                    min_len = feat_data.shape[0]
            
            # 限制最大长度
            target_len = min(min_len, max_seq_len)
            
            # 处理每种特征
            for feat_type, feat_data in features_dict.items():
                if feat_data.shape[0] > target_len:
                    feat_data = feat_data[:target_len]
                
                processed_features[feat_type] = torch.from_numpy(feat_data).float().unsqueeze(0).to(device)
            
            # 创建padding mask
            padding_mask = torch.zeros(1, target_len).bool().to(device)
            
            # 前向传播获取池化特征
            _, _, pooled_features = model(processed_features, padding_mask)
            result = pooled_features[0].cpu().numpy()
            
            # 立即释放GPU内存
            for feat_tensor in processed_features.values():
                del feat_tensor
            del padding_mask, pooled_features
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
            return result
            
        except RuntimeError as e:
            print(f"运行时错误，尝试在CPU上进行计算: {e}")
            # 如果GPU内存不足，回退到CPU
            device_cpu = torch.device("cpu")
            model = model.to(device_cpu)
            
            # 重新处理特征
            processed_features = {}
            for feat_type, feat_data in features_dict.items():
                processed_features[feat_type] = torch.from_numpy(feat_data).float().unsqueeze(0)
            
            padding_mask = torch.zeros(1, target_len).bool()
            
            _, _, pooled_features = model(processed_features, padding_mask)
            return pooled_features[0].numpy()

def load_model_from_checkpoint(model_path, device):
    """
    从检查点加载模型
    
    参数:
        model_path: 模型文件路径
        device: 设备
        
    返回:
        加载的模型
    """
    try:
        # 尝试加载PyTorch模型文件
        if model_path.endswith('.bin') or model_path.endswith('.pt'):
            # 检查是否是Hugging Face格式的模型
            model_dir = os.path.dirname(model_path)
            config_path = os.path.join(model_dir, 'config.json')
            
            if os.path.exists(config_path):
                # 使用Hugging Face格式加载
                model = VADModelWithGating.from_pretrained(model_dir)
            else:
                # 使用默认配置创建模型
                print("未找到config.json，使用默认配置创建模型...")
                config = VADConfig(
                    emotion2vec_dim=1024,
                    hubert_dim=1024,
                    hidden_dim=1024,
                    num_hidden_layers=4
                )
                model = VADModelWithGating(config)
                
                # 加载模型权重
                model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise ValueError(f"不支持的模型文件格式: {model_path}")
            
        model = model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试使用默认配置...")
        
        # 使用默认配置创建模型
        config = VADConfig(
            emotion2vec_dim=1024,
            hubert_dim=1024,
            hidden_dim=1024,
            num_hidden_layers=4
        )
        model = VADModelWithGating(config).to(device)
        
        # 再次尝试加载模型状态
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            return model
        except Exception as e2:
            print(f"第二次尝试加载模型失败: {e2}")
            raise RuntimeError("无法加载模型，请检查模型路径和格式")

def preprocess_features(features):
    """标准化特征"""
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
        # metric='cosine',
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
        'hap': {'color': '#FF3333', 'marker': '^', 'label': 'Happy'},      # 更鲜艳的红色
        'ang': {'color': '#0066CC', 'marker': '>', 'label': 'Angry'},      # 更鲜艳的蓝色
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
    
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.10),
        ncol=4,
        fontsize=15,
        frameon=False,
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

def extract_features(model_path, emotion2vec_dir, hubert_dir, df, device, wav2vec_dir=None, data2vec_dir=None):
    """
    提取特征
    
    参数:
        model_path: 模型路径
        emotion2vec_dir: emotion2vec特征目录
        hubert_dir: hubert特征目录
        df: 数据DataFrame
        device: 设备
        wav2vec_dir: wav2vec特征目录（可选）
        data2vec_dir: data2vec特征目录（可选）
        
    返回:
        特征数组和标签列表
    """
    # 加载模型
    model = load_model_from_checkpoint(model_path, device)
    
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
                # 加载必需的特征
                emotion2vec_path = os.path.join(emotion2vec_dir, f"{base_filename}.npy")
                hubert_path = os.path.join(hubert_dir, f"{base_filename}.npy")
                
                emotion2vec_features = np.load(emotion2vec_path)
                hubert_features = np.load(hubert_path)
                
                # 对齐特征长度
                target_len = max(emotion2vec_features.shape[0], hubert_features.shape[0])
                emotion2vec_features = interpolate_features(emotion2vec_features, target_len)
                hubert_features = interpolate_features(hubert_features, target_len)
                
                # 构建特征字典
                features_dict = {
                    'emotion2vec': emotion2vec_features,
                    'hubert': hubert_features
                }
                
                # 加载其他特征（如果目录存在）
                if wav2vec_dir:
                    wav2vec_path = os.path.join(wav2vec_dir, f"{base_filename}.npy")
                    if os.path.exists(wav2vec_path):
                        wav2vec_features = np.load(wav2vec_path)
                        wav2vec_features = interpolate_features(wav2vec_features, target_len)
                        features_dict['wav2vec'] = wav2vec_features
                
                if data2vec_dir:
                    data2vec_path = os.path.join(data2vec_dir, f"{base_filename}.npy")
                    if os.path.exists(data2vec_path):
                        data2vec_features = np.load(data2vec_path)
                        data2vec_features = interpolate_features(data2vec_features, target_len)
                        features_dict['data2vec'] = data2vec_features
                
                # 提取池化特征
                pooled_features = extract_pooled_features(features_dict, model, device)
                
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
    parser.add_argument('--wav2vec_dir', type=str, default=None,
                        help='wav2vec特征目录（可选）')
    parser.add_argument('--data2vec_dir', type=str, default=None,
                        help='data2vec特征目录（可选）')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型检查点文件路径 (.pt或.bin)')
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
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 根据参数决定使用CPU还是GPU
    device = torch.device("cpu" if args.use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"使用设备: {device}")
    
    df = pd.read_csv(args.csv_path)
    print(f"从标注文件加载了 {len(df)} 个样本")
    
    try:
        features, emotions = extract_features(
            args.model_path,
            args.emotion2vec_dir,
            args.hubert_dir,
            df,
            device,
            args.wav2vec_dir,
            args.data2vec_dir
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
            device,
            args.wav2vec_dir,
            args.data2vec_dir
        )
    
    print(f"提取的特征形状: {features.shape}")
    print(f"数据集中的唯一情感类别: {set(emotions)}")
    
    # 英文标签名称 - 确保这些与你的实际标签匹配
    label_names = {
        'neu': 'Neutral',
        'hap': 'Happy',
        'ang': 'Angry',
        'sad': 'Sad',
        'sur': 'Surprise',
        'fea': 'Fear',
        'dis': 'Disgust',
        'con': 'Contempt'
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
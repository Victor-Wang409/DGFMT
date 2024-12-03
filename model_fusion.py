import torch
from torch import nn, optim
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import pandas as pd
import logging
import argparse
import json
import os
from collections import defaultdict
from transformers import PretrainedConfig, PreTrainedModel

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def load_checkpoint(model_dir, device):
    """
    加载检查点中的模型
    """
    # 加载配置
    config = VADConfig.from_pretrained(model_dir)
    
    # 初始化模型
    model = EmotionModel(config)
    
    # 加载模型权重
    model_file = os.path.join(model_dir, 'pytorch_model.bin')
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, map_location=device))
    
    # 加载优化器状态(如果存在)
    optimizer_file = os.path.join(model_dir, 'optimizer.pt')
    optimizer_state = None
    if os.path.exists(optimizer_file):
        optimizer_state = torch.load(optimizer_file, map_location=device)
    
    # 加载训练信息(如果存在)
    training_info_file = os.path.join(model_dir, 'training_info.json')
    training_info = None
    if os.path.exists(training_info_file):
        with open(training_info_file, 'r') as f:
            training_info = json.load(f)
    
    return model, optimizer_state, training_info, config

def resume_training(args, fold, epoch):
    """
    从指定的fold和epoch恢复训练
    """
    model_dir = os.path.join(args.save_dir, f'fold{fold}', f'epoch{epoch}')
    model, optimizer_state, training_info, config = load_checkpoint(model_dir, args.device)
    
    return model, optimizer_state, training_info, config

class VADConfig(PretrainedConfig):
    model_type = "vad_emotion"  # 添加模型类型标识
    
    def __init__(
        self,
        emotion2vec_dim=768,
        hubert_dim=768,
        fusion_dim=768,
        hidden_dim=256,
        emotion_hidden_dim=128,
        num_emotion_classes=4,
        projection_dim=768,
        temperature=0.07,
        use_hdgf=False,
        use_cl=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.emotion2vec_dim = emotion2vec_dim
        self.hubert_dim = hubert_dim
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        self.emotion_hidden_dim = emotion_hidden_dim
        self.num_emotion_classes = num_emotion_classes
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.use_hdgf = use_hdgf
        self.use_cl = use_cl

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, emotion2vec_dir, hubert_dir, csv_path):
        self.df = pd.read_csv(csv_path)
        self.emotion2vec_dir = emotion2vec_dir
        self.hubert_dir = hubert_dir
        
        self.vad_labels = []
        for vad_str in self.df['VAD_normalized']:
            vad = eval(vad_str)
            self.vad_labels.append(torch.tensor(vad, dtype=torch.float))
            
        self.emotion_map = {'N': 0, 'H': 1, 'S': 2, 'A': 3}
        self.emotion_labels = [self.emotion_map.get(label, 0) for label in self.df['Label']]
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = os.path.splitext(row['FileName'])[0]
        
        # Load emotion2vec features
        emotion2vec_path = os.path.join(self.emotion2vec_dir, f"{filename}.npy")
        emotion2vec_features = torch.from_numpy(np.load(emotion2vec_path)).float()
        
        # Load hubert features
        hubert_path = os.path.join(self.hubert_dir, f"{filename}.npy")
        hubert_features = torch.from_numpy(np.load(hubert_path)).float()
        
        vad_label = self.vad_labels[idx]
        emotion_label = self.emotion_labels[idx]
        
        return {
            "id": row['FileName'],
            "emotion2vec_features": emotion2vec_features,
            "hubert_features": hubert_features,
            "vad_labels": vad_label,
            "emotion_labels": emotion_label
        }

class FeatureFusion(nn.Module):
    def __init__(self, emotion2vec_dim, hubert_dim, fusion_dim):
        super().__init__()
        
        # 投影层
        self.emotion2vec_proj = nn.Sequential(
            nn.Linear(emotion2vec_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        
        self.hubert_proj = nn.Sequential(
            nn.Linear(hubert_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        
        # 融合门控层
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, emotion2vec_feat, hubert_feat):
        # 投影特征到相同维度
        emotion2vec_proj = self.emotion2vec_proj(emotion2vec_feat)  # [B, T, fusion_dim]
        hubert_proj = self.hubert_proj(hubert_feat)  # [B, T, fusion_dim]
        
        # 计算每个时间步的融合权重
        concat_feat = torch.cat([emotion2vec_proj, hubert_proj], dim=-1)  # [B, T, fusion_dim*2]
        weights = self.fusion_gate(concat_feat)  # [B, T, 2]
        
        # 加权融合 (扩展权重维度以匹配特征维度)
        fused_feat = (weights[..., 0].unsqueeze(-1) * emotion2vec_proj + 
                     weights[..., 1].unsqueeze(-1) * hubert_proj)  # [B, T, fusion_dim]
        
        # 输出层处理
        output = self.output_layer(fused_feat)  # [B, T, fusion_dim]
        
        return output, weights

class MultiGatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.semantic_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.emotion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=-1)
        )
        
        self.transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
    def forward(self, text_features, audio_features):
        semantic_concat = torch.cat([text_features, audio_features], dim=-1)
        semantic_gate = self.semantic_gate(semantic_concat)
        semantic_fusion = semantic_gate * audio_features
        
        emotion_concat = torch.cat([text_features, audio_features], dim=-1)
        emotion_gate = self.emotion_gate(emotion_concat)
        emotion_fusion = emotion_gate * audio_features
        
        weight_input = torch.cat([text_features, semantic_fusion, emotion_fusion], dim=-1)
        weights = self.weight_generator(weight_input)
        
        fusion_features = torch.stack([semantic_fusion, emotion_fusion], dim=1)
        
        fusion_output = torch.bmm(weights.unsqueeze(1), fusion_features).squeeze(1)
        
        fusion_output = fusion_output + text_features
        
        final_output = self.transform(fusion_output)
        
        return final_output, {
            'semantic_gate': semantic_gate,
            'emotion_gate': emotion_gate,
            'fusion_weights': weights
        }

class EmotionModel(PreTrainedModel):
    config_class = VADConfig
    base_model_prefix = "vad_emotion"
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.feature_fusion = FeatureFusion(
            config.emotion2vec_dim,
            config.hubert_dim,
            config.fusion_dim
        )
        
        self.pre_net = nn.Linear(config.fusion_dim, config.hidden_dim)
        self.post_net = nn.Linear(config.hidden_dim, 3)
        self.activate = nn.ReLU()
        
        self.emotion_net = nn.Sequential(
            nn.Linear(config.fusion_dim, config.emotion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.emotion_hidden_dim, config.emotion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.emotion_hidden_dim // 2, config.num_emotion_classes)
        )
        
        if config.use_hdgf:
            self.hdgf = MultiGatedFusion(config.hidden_dim)
        
        if config.use_cl:
            projection_input_dim = config.hidden_dim + config.fusion_dim
            self.projection_head = nn.Sequential(
                nn.Linear(projection_input_dim, config.projection_dim),
                nn.GELU(),
                nn.Linear(config.projection_dim, config.projection_dim)
            )
            self.temperature = config.temperature
        
        self.attention = nn.Sequential(
            nn.Linear(config.fusion_dim, 1),
            nn.Tanh()
        )
        
        self.feature_transform = nn.Sequential(
            nn.Linear(config.fusion_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )

    def save_pretrained(self, save_directory):
        """重写保存方法以符合HuggingFace格式"""
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存配置文件
        self.config.save_pretrained(save_directory)
        
        # 保存模型权重
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # 添加模型卡片信息
        model_card = {
            "model_type": self.config.model_type,
            "architectures": [self.__class__.__name__],
            "_name_or_path": save_directory,
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(model_card, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, *model_args, **kwargs):
        """重写加载方法以符合HuggingFace格式"""
        config = kwargs.pop("config", None)
        if config is None:
            config = VADConfig.from_pretrained(pretrained_model_path)
            
        model = cls(config)
        
        # 加载模型权重
        model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model
    
    def forward(self, emotion2vec_features, hubert_features, padding_mask=None, 
                emotion_labels=None, vad_labels=None, mode="train"):
        features, fusion_weights = self.feature_fusion(emotion2vec_features, hubert_features)
        
        vad_hidden = self.activate(self.pre_net(features))
        if padding_mask is not None:
            vad_hidden = vad_hidden * (1 - padding_mask.unsqueeze(-1).float())
            vad_hidden = vad_hidden.sum(dim=1) / (1 - padding_mask.float()).sum(dim=1, keepdim=True)
        else:
            vad_hidden = vad_hidden.mean(dim=1)
        
        attention_weights = self.attention(features)
        if padding_mask is not None:
            attention_weights = attention_weights.masked_fill(
                padding_mask.unsqueeze(-1).to(torch.bool),
                float('-inf')
            )
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended_features = torch.sum(features * attention_weights, dim=1)
        
        if self.config.use_hdgf:
            transformed_features = self.feature_transform(attended_features)
            fused_features, fusion_info = self.hdgf(transformed_features, vad_hidden)
            vad_output = torch.sigmoid(self.post_net(fused_features))
        else:
            vad_output = torch.sigmoid(self.post_net(vad_hidden))
            fusion_info = None
        
        emotion_logits = self.emotion_net(attended_features)
        
        outputs = {
            'vad': vad_output,
            'emotion': emotion_logits,
            'attention_weights': attention_weights,
            'fusion_weights': fusion_weights
        }
        
        if fusion_info is not None:
            outputs['fusion_info'] = fusion_info
            
        if self.config.use_cl and mode == "train" and emotion_labels is not None:
            features_to_project = torch.cat([
                fused_features if self.config.use_hdgf else vad_hidden, 
                attended_features
            ], dim=-1)
            
            proj_features = self.projection_head(features_to_project)
            similarity = torch.einsum('nc,ck->nk', [proj_features, proj_features.T])
            similarity = torch.log_softmax(similarity / self.temperature, dim=-1)
            
            labels = emotion_labels
            mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
            positives = mask.float()
            
            cl_loss = -(similarity * positives).sum(dim=1) / positives.sum(dim=1)
            cl_loss = cl_loss.mean()
            outputs['cl_loss'] = cl_loss
        
        return outputs

def compute_dimension_ccc(preds, labels):
    preds_mean = torch.mean(preds)
    labels_mean = torch.mean(labels)
    
    preds_var = torch.mean((preds - preds_mean) ** 2)
    labels_var = torch.mean((labels - labels_mean) ** 2)
    
    covar = torch.mean((preds - preds_mean) * (labels - labels_mean))
    
    ccc = 2 * covar / (preds_var + labels_var + (preds_mean - labels_mean) ** 2 + 1e-8)
    return ccc

class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, preds, labels):
        ccc_v = compute_dimension_ccc(preds[:, 0], labels[:, 0])
        ccc_a = compute_dimension_ccc(preds[:, 1], labels[:, 1])
        ccc_d = compute_dimension_ccc(preds[:, 2], labels[:, 2])
        
        mean_ccc = (ccc_v + ccc_a + ccc_d) / 3.0
        return torch.tensor(1.0, device=preds.device) - mean_ccc

class CombinedLoss(nn.Module):
    def __init__(self, vad_weight=1.0, emotion_weight=1.0, cl_weight=0.1, class_weights=None):
        super().__init__()
        self.vad_loss = CCCLoss()
        self.emotion_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.vad_weight = vad_weight
        self.emotion_weight = emotion_weight
        self.cl_weight = cl_weight
        
    def forward(self, outputs, vad_labels, emotion_labels):
            vad_loss = self.vad_loss(outputs['vad'], vad_labels)
            emotion_loss = self.emotion_loss(outputs['emotion'], emotion_labels)
            
            total_loss = self.vad_weight * vad_loss + self.emotion_weight * emotion_loss
            
            loss_dict = {
                'vad_loss': vad_loss.item(),
                'emotion_loss': emotion_loss.item(),
            }
            
            if 'cl_loss' in outputs:
                cl_loss = outputs['cl_loss']
                total_loss += self.cl_weight * cl_loss
                loss_dict['cl_loss'] = cl_loss.item()
                
            if 'fusion_info' in outputs:
                loss_dict.update({
                    'semantic_gate_mean': outputs['fusion_info']['semantic_gate'].mean().item(),
                    'emotion_gate_mean': outputs['fusion_info']['emotion_gate'].mean().item(),
                    'fusion_weights_mean': outputs['fusion_info']['fusion_weights'].mean().item()
                })
            
            return total_loss, loss_dict

def collate_fn(batch):
    max_len_emotion2vec = max([b["emotion2vec_features"].shape[0] for b in batch])
    max_len_hubert = max([b["hubert_features"].shape[0] for b in batch])
    
    batch_emotion2vec = []
    batch_hubert = []
    batch_padding_masks = []
    batch_ids = []
    batch_vad_labels = []
    batch_emotion_labels = []
    
    for item in batch:
        # Process emotion2vec features
        emotion2vec_feats = item["emotion2vec_features"]
        curr_len = emotion2vec_feats.shape[0]
        pad_len = max_len_emotion2vec - curr_len
        
        if pad_len > 0:
            emotion2vec_feats = torch.cat([
                emotion2vec_feats,
                torch.zeros(pad_len, emotion2vec_feats.shape[1])
            ], dim=0)
            padding_mask = torch.cat([
                torch.zeros(curr_len),
                torch.ones(pad_len)
            ], dim=0)
        else:
            padding_mask = torch.zeros(curr_len)
            
        # Process hubert features
        hubert_feats = item["hubert_features"]
        curr_len_hubert = hubert_feats.shape[0]
        pad_len_hubert = max_len_hubert - curr_len_hubert
        
        if pad_len_hubert > 0:
            hubert_feats = torch.cat([
                hubert_feats,
                torch.zeros(pad_len_hubert, hubert_feats.shape[1])
            ], dim=0)
        
        batch_emotion2vec.append(emotion2vec_feats)
        batch_hubert.append(hubert_feats)
        batch_padding_masks.append(padding_mask)
        batch_ids.append(item["id"])
        batch_vad_labels.append(item["vad_labels"])
        batch_emotion_labels.append(item["emotion_labels"])
    
    return {
        "id": batch_ids,
        "emotion2vec_features": torch.stack(batch_emotion2vec),
        "hubert_features": torch.stack(batch_hubert),
        "padding_mask": torch.stack(batch_padding_masks).bool(),
        "vad_labels": torch.stack(batch_vad_labels),
        "emotion_labels": torch.tensor(batch_emotion_labels)
    }

def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    epoch_losses = defaultdict(list)
    
    progress = tqdm(train_loader, desc='Training', leave=False)
    for batch in progress:
        emotion2vec_features = batch["emotion2vec_features"].to(device)
        hubert_features = batch["hubert_features"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        vad_labels = batch["vad_labels"].to(device)
        emotion_labels = batch["emotion_labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(
            emotion2vec_features=emotion2vec_features,
            hubert_features=hubert_features,
            padding_mask=padding_mask,
            emotion_labels=emotion_labels,
            vad_labels=vad_labels,
            mode="train"
        )
        
        loss, loss_dict = criterion(outputs, vad_labels, emotion_labels)
        loss.backward()
        optimizer.step()
        
        for k, v in loss_dict.items():
            epoch_losses[k].append(v)
        epoch_losses['total_loss'].append(loss.item())
        
        progress.set_postfix({k: f'{np.mean(v):.4f}' for k, v in epoch_losses.items()})
    
    return {k: np.mean(v) for k, v in epoch_losses.items()}

def validate_and_test(model, data_loader, device):
    model.eval()
    all_vad_preds = []
    all_vad_labels = []
    all_emotion_preds = []
    all_emotion_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating', leave=False):
            emotion2vec_features = batch["emotion2vec_features"].to(device)
            hubert_features = batch["hubert_features"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            vad_labels = batch["vad_labels"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)
            
            outputs = model(
                emotion2vec_features=emotion2vec_features,
                hubert_features=hubert_features,
                padding_mask=padding_mask,
                emotion_labels=emotion_labels,
                vad_labels=vad_labels,
                mode="eval"
            )
            
            all_vad_preds.append(outputs['vad'])
            all_vad_labels.append(vad_labels)
            all_emotion_preds.append(outputs['emotion'])
            all_emotion_labels.append(emotion_labels)
    
    all_vad_preds = torch.cat(all_vad_preds, dim=0)
    all_vad_labels = torch.cat(all_vad_labels, dim=0)
    all_emotion_preds = torch.cat(all_emotion_preds, dim=0)
    all_emotion_labels = torch.cat(all_emotion_labels, dim=0)
    
    ccc_v = compute_dimension_ccc(all_vad_preds[:, 0], all_vad_labels[:, 0]).item()
    ccc_a = compute_dimension_ccc(all_vad_preds[:, 1], all_vad_labels[:, 1]).item()
    ccc_d = compute_dimension_ccc(all_vad_preds[:, 2], all_vad_labels[:, 2]).item()
    
    emotion_preds = torch.argmax(all_emotion_preds, dim=1)
    emotion_acc = (emotion_preds == all_emotion_labels).float().mean().item()
    
    return {
        'ccc_v': ccc_v,
        'ccc_a': ccc_a,
        'ccc_d': ccc_d,
        'emotion_acc': emotion_acc
    }

def split_by_speaker(df):
    df['speaker'] = df['FileName'].apply(lambda x: x.split('_')[1])
    speakers = sorted(df['speaker'].unique())
    
    n_speakers = len(speakers)
    base_size = n_speakers // 5
    remainder = n_speakers % 5
    
    np.random.shuffle(speakers)
    
    folds = []
    start_idx = 0
    
    for i in range(5):
        fold_size = base_size + (1 if i < remainder else 0)
        end_idx = start_idx + fold_size
        
        fold_speakers = speakers[start_idx:end_idx]
        fold_idx = df[df['speaker'].isin(fold_speakers)].index.values
        
        folds.append(fold_idx)
        start_idx = end_idx
    
    return folds

def main():
    parser = argparse.ArgumentParser(description='Training Multi-modal Emotion Recognition Model')
    parser.add_argument('--emotion2vec_dir', type=str, required=True,
                      help='Directory containing emotion2vec features')
    parser.add_argument('--hubert_dir', type=str, required=True,
                      help='Directory containing hubert features')
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='.models')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=0.01)
    parser.add_argument('--vad_weight', type=float, default=1.0)
    parser.add_argument('--emotion_weight', type=float, default=1.0)
    parser.add_argument('--cl_weight', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--fusion_dim', type=int, default=768)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_hdgf', action='store_true')
    parser.add_argument('--use_cl', action='store_true')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.save_dir, 'training.log'))
        ]
    )
    
    # Initialize wandb if needed
    if args.use_wandb:
        import wandb
        wandb.init(
            project="emotion-recognition-fusion",
            config=vars(args)
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load dataset
    dataset = EmotionDataset(args.emotion2vec_dir, args.hubert_dir, args.csv_path)
    logging.info(f"Dataset size: {len(dataset)}")
    
    # Get feature dimensions
    sample_data = dataset[0]
    emotion2vec_dim = sample_data["emotion2vec_features"].shape[1]
    hubert_dim = sample_data["hubert_features"].shape[1]
    
    # Calculate class weights
    emotion_counts = np.bincount([data["emotion_labels"] for data in dataset])
    class_weights = torch.FloatTensor(1.0 / emotion_counts)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    
    # 5-fold cross validation
    folds = split_by_speaker(dataset.df)
    fold_results = []
    
    for fold in range(5):
        logging.info(f"\n{'='*50}\nFold {fold+1}/5\n{'='*50}")
        
        fold_dir = os.path.join(args.save_dir, f'fold{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Get data splits
        train_idx = np.concatenate([folds[i] for i in range(5) if i != fold and i != (fold+1)%5])
        val_idx = folds[(fold+1)%5]
        test_idx = folds[fold]
        
        # Create data loaders
        train_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(train_idx),
            collate_fn=collate_fn,
            num_workers=args.num_workers
        )
        val_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(val_idx),
            collate_fn=collate_fn,
            num_workers=args.num_workers
        )
        test_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(test_idx),
            collate_fn=collate_fn,
            num_workers=args.num_workers
        )
        
        # Initialize model
        config = VADConfig(
            emotion2vec_dim=emotion2vec_dim,
            hubert_dim=hubert_dim,
            fusion_dim=args.fusion_dim,
            hidden_dim=256,
            emotion_hidden_dim=128,
            num_emotion_classes=len(dataset.emotion_map),
            projection_dim=768,
            temperature=args.temperature,
            use_hdgf=args.use_hdgf,
            use_cl=args.use_cl
        )
        model = EmotionModel(config).to(device)
        
        # Initialize loss and optimizer
        criterion = CombinedLoss(
            vad_weight=args.vad_weight,
            emotion_weight=args.emotion_weight,
            cl_weight=args.cl_weight,
            class_weights=class_weights
        )
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
        
        best_val_metrics = None
        best_val_ccc_avg = float('-inf')
        best_model_state = None
        
        for epoch in range(args.epochs):
            # 训练
            train_metrics = train_one_epoch(model, optimizer, criterion, train_loader, device)
            
            # 验证
            val_metrics = validate_and_test(model, val_loader, device)
            val_ccc_avg = (val_metrics['ccc_v'] + val_metrics['ccc_a'] + val_metrics['ccc_d']) / 3
            
            # 保存每个epoch的模型
            epoch_dir = os.path.join(fold_dir, f'epoch{epoch+1}')
            os.makedirs(epoch_dir, exist_ok=True)
            
            # 使用HuggingFace格式保存模型
            model.save_pretrained(epoch_dir)
            
            # 保存优化器状态
            optimizer_path = os.path.join(epoch_dir, "optimizer.bin")
            torch.save(optimizer.state_dict(), optimizer_path)
            
            # 保存训练信息
            training_info = {
                "epoch": epoch + 1,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "training_args": vars(args)
            }
            
            with open(os.path.join(epoch_dir, "training_info.json"), "w") as f:
                json.dump(training_info, f, indent=2)
            
            # 判断是否是最佳模型
            is_best_model = (best_val_metrics is None) or (val_ccc_avg > best_val_ccc_avg)
            
            if is_best_model:
                best_val_metrics = val_metrics
                best_val_ccc_avg = val_ccc_avg
                best_model_state = {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "training_args": vars(args)
                }
                
                # 保存最佳模型
                best_model_dir = os.path.join(fold_dir, "best_model")
                model.save_pretrained(best_model_dir)
                
                # 保存完整的训练状态
                torch.save(best_model_state, os.path.join(best_model_dir, "training_state.bin"))
                logging.info(f"Saved new best model at epoch {epoch+1}")
            
            # 日志记录
            log_info = (
                f"Fold {fold+1}, Epoch {epoch+1} | "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val CCC: V={val_metrics['ccc_v']:.3f}, "
                f"A={val_metrics['ccc_a']:.3f}, "
                f"D={val_metrics['ccc_d']:.3f} | "
                f"Avg={val_ccc_avg:.3f} | "
                f"Emotion Acc={val_metrics['emotion_acc']:.3f}"
            )
            logging.info(log_info)
            
            # Early stopping 检查
            early_stopping(1 - val_ccc_avg)
            if early_stopping.early_stop:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # 加载最佳模型进行测试
        if best_model_state is not None:
            model.load_state_dict(best_model_state["model"])
        
        # 测试阶段
        test_metrics = validate_and_test(model, test_loader, device)
        test_ccc_avg = (test_metrics['ccc_v'] + test_metrics['ccc_a'] + test_metrics['ccc_d']) / 3
        
        fold_results.append(test_metrics)
        logging.info(
            f"\nFold {fold+1} Results (Best Epoch {best_model_state['epoch']+1 if best_model_state else 'N/A'}):\n"
            f"Test CCC: V={test_metrics['ccc_v']:.3f}, "
            f"A={test_metrics['ccc_a']:.3f}, "
            f"D={test_metrics['ccc_d']:.3f} | "
            f"Avg={test_ccc_avg:.3f} | "
            f"Emotion Acc={test_metrics['emotion_acc']:.3f}"
        )
    
    # Calculate average results across all folds
    avg_metrics = {
        'ccc_v': np.mean([res['ccc_v'] for res in fold_results]),
        'ccc_a': np.mean([res['ccc_a'] for res in fold_results]),
        'ccc_d': np.mean([res['ccc_d'] for res in fold_results]),
        'emotion_acc': np.mean([res['emotion_acc'] for res in fold_results])
    }
    
    std_metrics = {
        'ccc_v': np.std([res['ccc_v'] for res in fold_results]),
        'ccc_a': np.std([res['ccc_a'] for res in fold_results]),
        'ccc_d': np.std([res['ccc_d'] for res in fold_results]),
        'emotion_acc': np.std([res['emotion_acc'] for res in fold_results])
    }
    
    # Print and save final results
    final_results = (
        f"\nFinal Cross-Validation Results\n"
        f"{'='*50}\n"
        f"Average ± std:\n"
        f"CCC Valence: {avg_metrics['ccc_v']:.3f} ± {std_metrics['ccc_v']:.3f}\n"
        f"CCC Arousal: {avg_metrics['ccc_a']:.3f} ± {std_metrics['ccc_a']:.3f}\n"
        f"CCC Dominance: {avg_metrics['ccc_d']:.3f} ± {std_metrics['ccc_d']:.3f}\n"
        f"Average CCC: {(avg_metrics['ccc_v'] + avg_metrics['ccc_a'] + avg_metrics['ccc_d'])/3:.3f}\n"
        f"Emotion Accuracy: {avg_metrics['emotion_acc']:.3f} ± {std_metrics['emotion_acc']:.3f}\n"
        f"{'='*50}"
    )
    
    logging.info(final_results)
    
    # Save final results
    results_path = os.path.join(args.save_dir, 'final_results.txt')
    with open(results_path, 'w') as f:
        f.write(final_results)
    
    if args.use_wandb:
        wandb.log({
            'final/avg_ccc': (avg_metrics['ccc_v'] + avg_metrics['ccc_a'] + avg_metrics['ccc_d'])/3,
            **{f'final/{k}': v for k, v in avg_metrics.items()},
            **{f'final/{k}_std': v for k, v in std_metrics.items()}
        })
        wandb.finish()

if __name__ == '__main__':
    main()
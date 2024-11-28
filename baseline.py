from torch import nn, optim
import soundfile as sf
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import pandas as pd
import logging
import fairseq
import argparse
import os
from transformers import PretrainedConfig, PreTrainedModel

class VADConfig(PretrainedConfig):
    def __init__(self, input_dim=768, hidden_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

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

def collate_fn(batch):
    max_len = max([b["net_input"]["feats"].shape[0] for b in batch])
    
    batch_feats = []
    batch_padding_masks = []
    batch_ids = []
    batch_labels = []
    
    for item in batch:
        feats = item["net_input"]["feats"]
        curr_len = feats.shape[0]
        pad_len = max_len - curr_len
        
        if pad_len > 0:
            feats = torch.cat([
                feats,
                torch.zeros(pad_len, feats.shape[1])
            ], dim=0)
            padding_mask = torch.cat([
                torch.zeros(curr_len),
                torch.ones(pad_len)
            ], dim=0)
        else:
            padding_mask = torch.zeros(curr_len)
        
        batch_feats.append(feats)
        batch_padding_masks.append(padding_mask)
        batch_ids.append(item["id"])
        batch_labels.append(item["labels"])
    
    return {
        "id": batch_ids,
        "net_input": {
            "feats": torch.stack(batch_feats),
            "padding_mask": torch.stack(batch_padding_masks).bool()
        },
        "labels": torch.stack(batch_labels)
    }

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_path, feature_extractor):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        
        self.vad_labels = []
        for vad_str in self.df['VAD_normalized']:
            vad = eval(vad_str)
            self.vad_labels.append(torch.tensor(vad, dtype=torch.float))
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.root_dir, f"{row['FileName']}")
        
        features = self.feature_extractor.extract_features(audio_path)
        label = self.vad_labels[idx]
        
        return {
            "id": row['FileName'],
            "net_input": {
                "feats": features,
                "padding_mask": torch.zeros(features.size(0))
            },
            "labels": label
        }

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

def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0.0
    total_batches = 0
    
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for batch in progress_bar:
        ids = batch["id"]
        feats = batch["net_input"]["feats"].to(device)
        padding_mask = batch["net_input"]["padding_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(feats, padding_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_batches += 1
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / total_batches

def validate_and_test(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(data_loader, desc='Evaluating', leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            feats = batch["net_input"]["feats"].to(device)
            padding_mask = batch["net_input"]["padding_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(feats, padding_mask)
            
            all_preds.append(outputs)
            all_labels.append(labels)
            
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    ccc_v = compute_dimension_ccc(all_preds[:, 0], all_labels[:, 0]).item()
    ccc_a = compute_dimension_ccc(all_preds[:, 1], all_labels[:, 1]).item()
    ccc_d = compute_dimension_ccc(all_preds[:, 2], all_labels[:, 2]).item()
    
    return ccc_v, ccc_a, ccc_d

def split_by_speaker(df):
    # 从文件名中提取说话人ID (例如: MSP-PODCAST_1353 -> 1353)
    df['speaker'] = df['FileName'].apply(lambda x: x.split('_')[1])
    speakers = sorted(df['speaker'].unique())
    
    # 计算基本每折大小和余数
    n_speakers = len(speakers)
    base_size = n_speakers // 5
    remainder = n_speakers % 5
    
    # 随机打乱说话人顺序
    np.random.shuffle(speakers)
    
    # 分配说话人到每个fold
    folds = []
    start_idx = 0
    
    for i in range(5):
        # 如果有余数，前remainder个fold多分配一个说话人
        fold_size = base_size + (1 if i < remainder else 0)
        end_idx = start_idx + fold_size
        
        # 获取当前fold的说话人
        fold_speakers = speakers[start_idx:end_idx]
        
        # 获取属于这些说话人的所有样本索引
        fold_idx = df[df['speaker'].isin(fold_speakers)].index.values
        
        folds.append(fold_idx)
        start_idx = end_idx
        
    # 打印每个fold的大小，用于调试
    for i, fold in enumerate(folds):
        print(f"Fold {i+1} size: {len(fold)} samples, "
              f"with {len(set(df.iloc[fold]['speaker']))} speakers")
              
    return folds

def get_train_eval_test_split(df, fold_idx, all_folds):
    """
    将数据集划分为训练、验证和测试集
    fold_idx对应的fold作为测试集
    (fold_idx + 1) % 5对应的fold作为验证集
    其余的fold作为训练集
    """
    test_idx = np.array(all_folds[fold_idx])
    eval_idx = np.array(all_folds[(fold_idx + 1) % 5])
    
    # 其余fold作为训练集
    train_folds = [all_folds[i] for i in range(5) 
                  if i != fold_idx and i != (fold_idx + 1) % 5]
    train_idx = np.concatenate(train_folds)
    
    # 打印划分信息
    print(f"\nFold {fold_idx + 1} split info:")
    print(f"Train set: {len(train_idx)} samples")
    print(f"Eval set: {len(eval_idx)} samples")
    print(f"Test set: {len(test_idx)} samples\n")
    
    return train_idx, eval_idx, test_idx

def main():
    parser = argparse.ArgumentParser(description='Training VAD prediction model')
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.save_dir, 'training.log'))
        ]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    feature_extractor = Emotion2vecExtractor(args.model_path, args.checkpoint_path)
    dataset = EmotionDataset(args.audio_path, args.csv_path, feature_extractor)
    
    folds = split_by_speaker(dataset.df)
    fold_results = []
    
    for fold in range(5):
        logging.info(f"\n{'='*50}\nFold {fold+1}/5\n{'='*50}")
        
        train_idx, eval_idx, test_idx = get_train_eval_test_split(dataset.df, fold, folds)
        
        train_sampler = SubsetRandomSampler(train_idx)
        eval_sampler = SubsetRandomSampler(eval_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        
        train_loader = DataLoader(dataset, batch_size=args.batch_size,
                                sampler=train_sampler, collate_fn=collate_fn)
        eval_loader = DataLoader(dataset, batch_size=args.batch_size,
                               sampler=eval_sampler, collate_fn=collate_fn)
        test_loader = DataLoader(dataset, batch_size=args.batch_size,
                               sampler=test_sampler, collate_fn=collate_fn)
        
        config = VADConfig()
        model = VADModel(config).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        criterion = CCCLoss()
        
        best_eval_ccc = -float('inf')
        best_model = None
        
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
            eval_v, eval_a, eval_d = validate_and_test(model, eval_loader, device)
            eval_ccc_avg = (eval_v + eval_a + eval_d) / 3
            
            if eval_ccc_avg > best_eval_ccc:
                best_eval_ccc = eval_ccc_avg
                best_model = model.state_dict()
                save_directory = os.path.join(args.save_dir, f'best_model_fold{fold+1}')
                model.save_pretrained(save_directory)
                config.save_pretrained(save_directory)
            
            logging.info(
                f"Fold {fold+1}, Epoch {epoch+1:3d} | "
                f"Loss: {train_loss:.4f} | "
                f"Eval CCC: V={eval_v:.3f}, A={eval_a:.3f}, D={eval_d:.3f} | "
                f"Avg={eval_ccc_avg:.3f}"
            )
        
        model.load_state_dict(best_model)
        test_v, test_a, test_d = validate_and_test(model, test_loader, device)
        test_ccc_avg = (test_v + test_a + test_d) / 3
        
        fold_results.append((test_v, test_a, test_d))
        logging.info(
            f"\nFold {fold+1} Test Results | "
            f"CCC: V={test_v:.3f}, A={test_a:.3f}, D={test_d:.3f} | "
            f"Avg={test_ccc_avg:.3f}"
        )
    
    avg_v = np.mean([res[0] for res in fold_results])
    avg_a = np.mean([res[1] for res in fold_results])
    avg_d = np.mean([res[2] for res in fold_results])
    avg_all = (avg_v + avg_a + avg_d) / 3
    
    std_v = np.std([res[0] for res in fold_results])
    std_a = np.std([res[1] for res in fold_results])
    std_d = np.std([res[2] for res in fold_results])
    
    logging.info(f"\n{'='*50}\nFinal Cross-Validation Results\n{'='*50}")
    logging.info(
        f"Average CCC ± std: \n"
        f"Valence: {avg_v:.3f} ± {std_v:.3f}\n"
        f"Arousal: {avg_a:.3f} ± {std_a:.3f}\n"
        f"Dominance: {avg_d:.3f} ± {std_d:.3f}\n"
        f"Overall: {avg_all:.3f}"
    )

if __name__ == '__main__':
   main()
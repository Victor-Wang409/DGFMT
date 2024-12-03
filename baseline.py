import torch
from torch import nn, optim
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import pandas as pd
import logging
import argparse
import os
from transformers import PretrainedConfig, PreTrainedModel

# EarlyStopping类保持不变
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

class VADConfig(PretrainedConfig):
    def __init__(self, input_dim=768, hidden_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

def collate_fn(batch):
    max_len = max([b["features"].shape[0] for b in batch])
    
    batch_feats = []
    batch_padding_masks = []
    batch_ids = []
    batch_labels = []
    
    for item in batch:
        feats = item["features"]
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
        "features": torch.stack(batch_feats),
        "padding_mask": torch.stack(batch_padding_masks).bool(),
        "labels": torch.stack(batch_labels)
    }

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, feature_dir, csv_path):
        self.df = pd.read_csv(csv_path)
        self.feature_dir = feature_dir
        
        self.vad_labels = []
        for vad_str in self.df['VAD_normalized']:
            vad = eval(vad_str)
            self.vad_labels.append(torch.tensor(vad, dtype=torch.float))
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_path = os.path.join(self.feature_dir, f"{os.path.splitext(row['FileName'])[0]}.npy")
        
        features = torch.from_numpy(np.load(feature_path)).float()
        label = self.vad_labels[idx]
        
        return {
            "id": row['FileName'],
            "features": features,
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

def split_by_speaker(df):
    """基于说话人ID划分数据集为5折"""
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
        
    for i, fold in enumerate(folds):
        print(f"Fold {i+1} size: {len(fold)} samples, "
              f"with {len(set(df.iloc[fold]['speaker']))} speakers")
              
    return folds

def get_train_eval_test_split(df, fold_idx, all_folds):
    """获取当前折的训练、验证和测试集划分"""
    test_idx = np.array(all_folds[fold_idx])
    eval_idx = np.array(all_folds[(fold_idx + 1) % 5])
    
    train_folds = [all_folds[i] for i in range(5) 
                  if i != fold_idx and i != (fold_idx + 1) % 5]
    train_idx = np.concatenate(train_folds)
    
    print(f"\nFold {fold_idx + 1} split info:")
    print(f"Train set: {len(train_idx)} samples")
    print(f"Eval set: {len(eval_idx)} samples")
    print(f"Test set: {len(test_idx)} samples\n")
    
    return train_idx, eval_idx, test_idx

def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0.0
    total_batches = 0
    
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for batch in progress_bar:
        features = batch["features"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(features, padding_mask)
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
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating', leave=False):
            features = batch["features"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(features, padding_mask)
            
            all_preds.append(outputs)
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    ccc_v = compute_dimension_ccc(all_preds[:, 0], all_labels[:, 0]).item()
    ccc_a = compute_dimension_ccc(all_preds[:, 1], all_labels[:, 1]).item()
    ccc_d = compute_dimension_ccc(all_preds[:, 2], all_labels[:, 2]).item()
    
    return ccc_v, ccc_a, ccc_d

def main():
    parser = argparse.ArgumentParser(description='Training VAD prediction model')
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default='./models')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=0.01)
    
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
    
    dataset = EmotionDataset(args.feature_dir, args.csv_path)
    
    # 获取输入特征维度
    sample_feature = np.load(os.path.join(args.feature_dir, 
                           f"{os.path.splitext(dataset.df['FileName'].iloc[0])[0]}.npy"))
    input_dim = sample_feature.shape[1]
    
    # 基于说话人进行5折交叉验证
    folds = split_by_speaker(dataset.df)
    fold_results = []
    
    for fold in range(5):
        logging.info(f"\n{'='*50}\nFold {fold+1}/5\n{'='*50}")
        
        # 创建当前fold的保存目录
        fold_dir = os.path.join(args.save_dir, f'fold{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        train_idx, eval_idx, test_idx = get_train_eval_test_split(dataset.df, fold, folds)
        
        train_loader = DataLoader(dataset, batch_size=args.batch_size,
                                sampler=SubsetRandomSampler(train_idx),
                                collate_fn=collate_fn)
        eval_loader = DataLoader(dataset, batch_size=args.batch_size,
                               sampler=SubsetRandomSampler(eval_idx),
                               collate_fn=collate_fn)
        test_loader = DataLoader(dataset, batch_size=args.batch_size,
                               sampler=SubsetRandomSampler(test_idx),
                               collate_fn=collate_fn)
        
        config = VADConfig(input_dim=input_dim)
        model = VADModel(config).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        criterion = CCCLoss()
        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
        
        best_val_ccc = -float('inf')
        best_model = None
        
        # 创建性能跟踪文件
        metrics_file = os.path.join(fold_dir, 'metrics.csv')
        with open(metrics_file, 'w') as f:
            f.write('epoch,train_loss,val_ccc_v,val_ccc_a,val_ccc_d,val_ccc_avg\n')
        
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
            val_v, val_a, val_d = validate_and_test(model, eval_loader, device)
            val_ccc_avg = (val_v + val_a + val_d) / 3
            
            # 保存每个epoch的模型
            epoch_dir = os.path.join(fold_dir, f'epoch{epoch+1}')
            os.makedirs(epoch_dir, exist_ok=True)
            
            # 保存模型和配置
            model.save_pretrained(epoch_dir)
            
            # 保存优化器状态
            torch.save(optimizer.state_dict(), os.path.join(epoch_dir, 'optimizer.pt'))
            
            # 保存训练指标
            with open(metrics_file, 'a') as f:
                f.write(f'{epoch+1},{train_loss:.4f},{val_v:.4f},{val_a:.4f},{val_d:.4f},{val_ccc_avg:.4f}\n')
            
            logging.info(
                f"Fold {fold+1}, Epoch {epoch+1:3d} | "
                f"Loss: {train_loss:.4f} | "
                f"Val CCC: V={val_v:.3f}, A={val_a:.3f}, D={val_d:.3f} | "
                f"Avg={val_ccc_avg:.3f}"
            )
            
            # 更新最佳模型
            if val_ccc_avg > best_val_ccc:
                best_val_ccc = val_ccc_avg
                best_model = model.state_dict()
                # 创建并保存最佳模型
                best_model_dir = os.path.join(fold_dir, 'best_model')
                os.makedirs(best_model_dir, exist_ok=True)
                model.save_pretrained(best_model_dir)
                logging.info(f"Saved new best model with val_ccc={val_ccc_avg:.3f}")
            
            # 保存checkpoint以便恢复训练
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_ccc': best_val_ccc,
                'config': config.to_dict()
            }
            torch.save(checkpoint, os.path.join(fold_dir, 'checkpoint.pt'))
            
            # 早停检查
            early_stopping(1 - val_ccc_avg)
            if early_stopping.early_stop:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # 加载最佳模型进行测试
        model.load_state_dict(best_model)
        test_v, test_a, test_d = validate_and_test(model, test_loader, device)
        test_ccc_avg = (test_v + test_a + test_d) / 3
        
        # 保存测试结果
        with open(os.path.join(fold_dir, 'test_results.txt'), 'w') as f:
            f.write(f"Test CCC:\nValence: {test_v:.3f}\nArousal: {test_a:.3f}\n"
                   f"Dominance: {test_d:.3f}\nAverage: {test_ccc_avg:.3f}")
        
        fold_results.append((test_v, test_a, test_d))
        logging.info(
            f"\nFold {fold+1} Test Results | "
            f"CCC: V={test_v:.3f}, A={test_a:.3f}, D={test_d:.3f} | "
            f"Avg={test_ccc_avg:.3f}"
        )
    
    # 计算并保存所有fold的最终结果
    avg_v = np.mean([res[0] for res in fold_results])
    avg_a = np.mean([res[1] for res in fold_results])
    avg_d = np.mean([res[2] for res in fold_results])
    avg_all = (avg_v + avg_a + avg_d) / 3
    
    std_v = np.std([res[0] for res in fold_results])
    std_a = np.std([res[1] for res in fold_results])
    std_d = np.std([res[2] for res in fold_results])
    
    final_results = (
        f"Final Cross-Validation Results\n"
        f"Average CCC ± std:\n"
        f"Valence: {avg_v:.3f} ± {std_v:.3f}\n"
        f"Arousal: {avg_a:.3f} ± {std_a:.3f}\n"
        f"Dominance: {avg_d:.3f} ± {std_d:.3f}\n"
        f"Overall: {avg_all:.3f}"
    )
    
    logging.info(f"\n{'='*50}\n{final_results}\n{'='*50}")
    
    # 保存最终结果
    with open(os.path.join(args.save_dir, 'final_results.txt'), 'w') as f:
        f.write(final_results)

if __name__ == '__main__':
    main()
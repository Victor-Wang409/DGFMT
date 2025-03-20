"""
数据处理模块
负责批次数据的收集和处理
"""

import torch

class DataProcessor:
    """
    数据处理类，处理批次数据的收集和处理
    """
    @staticmethod
    def collate_fn(batch):
        """
        数据批处理函数,用于DataLoader
        
        参数:
            batch: 批次数据
            
        返回:
            处理后的批次数据
        """
        max_len = max([b["emotion2vec_features"].shape[0] for b in batch])
        
        batch_e2v_feats = []
        batch_hub_feats = []
        batch_padding_masks = []
        batch_ids = []
        batch_labels = []
        batch_emotion_labels = []
        
        for item in batch:
            curr_len = item["emotion2vec_features"].shape[0]
            pad_len = max_len - curr_len
            
            if pad_len > 0:
                # 使用特征的实际维度创建padding
                e2v_padding = torch.zeros(pad_len, item["emotion2vec_features"].shape[1])  # 1024
                hub_padding = torch.zeros(pad_len, item["hubert_features"].shape[1])       # 1024
                
                e2v_feats = torch.cat([item["emotion2vec_features"], e2v_padding], dim=0)
                hub_feats = torch.cat([item["hubert_features"], hub_padding], dim=0)
                
                padding_mask = torch.cat([
                    torch.zeros(curr_len),
                    torch.ones(pad_len)
                ], dim=0)
            else:
                e2v_feats = item["emotion2vec_features"]
                hub_feats = item["hubert_features"]
                padding_mask = torch.zeros(curr_len)
            
            batch_e2v_feats.append(e2v_feats)
            batch_hub_feats.append(hub_feats)
            batch_padding_masks.append(padding_mask)
            batch_ids.append(item["id"])
            batch_labels.append(item["labels"])
            batch_emotion_labels.append(item["emotion_labels"])
        
        return {
            "id": batch_ids,
            "emotion2vec_features": torch.stack(batch_e2v_feats),
            "hubert_features": torch.stack(batch_hub_feats),
            "padding_mask": torch.stack(batch_padding_masks).bool(),
            "labels": torch.stack(batch_labels),
            "emotion_labels": torch.stack(batch_emotion_labels)
        }
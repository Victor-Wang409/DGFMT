# 情感语音分析项目

这个项目是一个基于深度学习的情感语音分析系统，用于预测语音的Valence-Arousal-Dominance (VAD) 值。项目采用了多模态特征融合的方法，结合了Emotion2Vec和HuBERT特征，通过门控机制和Transformer编码器进行特征提取和预测。

## 项目结构

项目代码按照高内聚低耦合的原则进行了重构，主要包含以下模块：

```
.
├── main.py                    # 主程序入口
├── __init__.py                # 模块初始化
├── early_stopping.py          # 早停机制
├── lr_scheduler.py            # 学习率调度器
├── data_processor.py          # 数据处理工具
├── dataset.py                 # 数据集定义
├── loss_functions.py          # 损失函数
├── model_components.py        # 模型组件
├── model.py                   # 模型定义
├── trainer.py                 # 训练管理
└── trainer_executor.py        # 训练执行器
```

## 模块说明

1. **early_stopping.py**: 实现了早停机制，防止模型训练过程中的过拟合。

2. **lr_scheduler.py**: 提供了多种学习率调度策略，包括余弦调度、线性调度和步长调度。

3. **data_processor.py**: 处理批次数据的收集和处理，主要包含DataLoader的collate_fn函数。

4. **dataset.py**: 定义了情感数据集类，负责加载和处理特征数据。

5. **loss_functions.py**: 实现了多种损失函数，包括CCC损失和监督对比损失。

6. **model_components.py**: 包含模型的各种组件，如注意力池化、多头注意力、Transformer编码器和门控特征融合。

7. **model.py**: 定义了VAD模型及其配置类。

8. **trainer.py**: 实现了模型训练和评估的核心功能。

9. **trainer_executor.py**: 负责整个训练流程的执行，包括数据加载、模型创建、训练评估和结果保存。

10. **main.py**: 主程序入口，解析命令行参数并执行训练流程。

## 使用方法

### 训练模型

```bash
python main.py \
    --emotion2vec_dir ./emo2vec_large_features \
    --hubert_dir ./hubert_large_features \
    --csv_path ./csv_files/MSP_Podcast.csv \
    --save_dir ./models \
    --epochs 80 \
    --lr 2e-5 \
    --effective_batch_size 24 \
    --gradient_accumulation_steps 4 \
    --warmup_epochs 5 \
    --lr_scheduler cosine \
    --min_lr 1e-6
```

### 参数说明

- `--emotion2vec_dir`: emotion2vec特征目录
- `--hubert_dir`: hubert特征目录
- `--csv_path`: 标注CSV文件路径
- `--save_dir`: 模型保存目录
- `--epochs`: 训练轮数
- `--lr`: 初始学习率
- `--effective_batch_size`: 有效批次大小
- `--gradient_accumulation_steps`: 梯度累积步数
- `--warmup_epochs`: 预热轮数
- `--lr_scheduler`: 学习率调度策略 (cosine/linear/step)
- `--min_lr`: 最小学习率

## 模型架构

模型采用了门控特征融合机制，将Emotion2Vec和HuBERT两种特征进行融合，然后通过Transformer编码器进行特征提取，最后通过注意力池化和全连接层得到预测结果。模型架构如下：

1. **特征融合层**: 使用门控机制融合Emotion2Vec和HuBERT特征
2. **Transformer编码层**: 多层Transformer编码器进行特征提取
3. **注意力池化层**: 提取全局特征表示
4. **输出层**: 预测VAD值

## 训练策略

训练采用了多种优化策略，包括：

1. **梯度累积**: 支持大批次训练
2. **学习率调度**: 支持多种学习率调度策略
3. **早停机制**: 防止过拟合
4. **对比学习**: 提高特征表示的区分性
5. **交叉验证**: 进行5折交叉验证，提高模型的泛化能力

## 评估指标

模型使用一致性相关系数(CCC)作为评估指标，针对Valence、Arousal和Dominance三个维度分别计算CCC值，并取平均
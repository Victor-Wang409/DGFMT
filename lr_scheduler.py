"""
学习率调度器模块
用于创建和管理各种学习率调度策略
"""

import math
from torch import optim

class LRSchedulerFactory:
    """
    学习率调度器工厂，用于创建各种学习率调度策略
    """
    @staticmethod
    def create_scheduler(optimizer, args, num_training_steps):
        """
        创建学习率调度器
        
        参数:
            optimizer: 优化器
            args: 参数配置
            num_training_steps: 每个epoch的训练步数
        
        返回:
            学习率调度器
        """
        num_warmup_steps = args.warmup_epochs * num_training_steps
        
        if args.lr_scheduler == 'cosine':
            return LRSchedulerFactory._get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps * args.epochs,
                min_lr=args.min_lr
            )
        elif args.lr_scheduler == 'linear':  
            return LRSchedulerFactory._get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps * args.epochs
            )
        elif args.lr_scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.lr_decay_step,
                gamma=args.lr_decay_rate
            )
        else:
            raise ValueError(f"Unknown scheduler type: {args.lr_scheduler}")

    @staticmethod
    def _get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0):
        """
        创建带有warmup的余弦学习率调度器
        
        参数:
            optimizer: 优化器
            num_warmup_steps: 预热步数
            num_training_steps: 总训练步数
            min_lr: 最小学习率
        
        返回:
            余弦调度器
        """
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def _get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        """
        创建带有warmup的线性学习率调度器
        
        参数:
            optimizer: 优化器
            num_warmup_steps: 预热步数
            num_training_steps: 总训练步数
        
        返回:
            线性调度器
        """
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
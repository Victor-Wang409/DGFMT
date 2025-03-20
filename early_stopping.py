"""
早停机制模块
用于防止模型训练过程中的过拟合
"""

class EarlyStopping:
    """
    早停机制，用于防止过拟合
    """
    def __init__(self, patience=10, min_delta=0):
        """
        初始化早停对象
        
        参数:
            patience: 容忍连续性能不提升的轮数
            min_delta: 性能提升的最小差异
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        """
        检查是否应该早停
        
        参数:
            val_loss: 当前验证损失
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss >= self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
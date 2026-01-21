"""
学习率调度器 (Learning Rate Scheduler)

实现原论文 "Attention Is All You Need" 的学习率调度策略：
lr = factor × d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))

特点：
- Warmup 期间学习率从 0 线性增长到峰值
- Warmup 后学习率按 step^(-0.5) 衰减
- 按 step (而非 epoch) 更新学习率
"""


class TransformerScheduler:
    """
    原论文学习率调度器

    lr = factor × d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))

    Args:
        optimizer: PyTorch 优化器
        d_model: 模型维度
        warmup_steps: warmup 步数 (原论文使用 4000)
        factor: 学习率缩放因子
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.current_step = 0

    def step(self):
        """每个训练 step 后调用，更新学习率"""
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        """计算当前 step 的学习率"""
        step = max(1, self.current_step)
        scale = self.d_model ** (-0.5)
        lr = scale * min(step ** (-0.5), step * (self.warmup_steps ** (-1.5)))
        return self.factor * lr

    def get_last_lr(self):
        """返回当前学习率 (兼容 PyTorch scheduler 接口)"""
        return [self._get_lr()]

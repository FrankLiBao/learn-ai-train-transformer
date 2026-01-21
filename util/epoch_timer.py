"""
训练计时工具 (Epoch Timer)

计算每个 epoch 的训练时间，返回分钟和秒数
用于监控训练进度
"""


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

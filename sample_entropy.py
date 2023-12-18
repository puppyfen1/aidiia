import nolds
import numpy as np

# 生成一个简单的时间序列数据
np.random.seed(42)
time_series = np.random.rand(100)

# 设置参数
window_size = 10
dimension = 3

# 计算样本熵
sample_entropy = nolds.sampen(time_series, emb_dim=dimension, tolerance=0.2)

print("样本熵:", sample_entropy)

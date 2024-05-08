# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：plot_maySmallBox.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/7/21 15:46 
"""
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# 定义行数和列数
num_rows = 20
num_cols = 5

# 生成随机颜色
def generate_random_color():

    return np.random.rand(3,)

# 创建一个新的图像
fig, ax = plt.subplots(num_rows, num_cols, figsize=(5, 20))
# colors = tuple(["#87CEFA","#FF6F61"])
# colors = tuple(["#FFD133","#FFB233","#FF8D33","#FF5733","#fadf53","#F8E2A7","#F9D9A4"])
# colors = tuple(list(plt.get_cmap("tab20").colors[::-1]))
# colors = tuple(list(plt.get_cmap("tab10").colors[::-1]))
cmap = plt.get_cmap('bwr')
# cmap = plt.get_cmap('coolwarm')
# 遍历每个格子并设置随机颜色
for i in range(num_rows):
    for j in range(num_cols):
        # color = generate_random_color()
        # index=np.random.choice(np.arange(len(colors)))
        # color=colors[index]
        color = cmap(np.random.rand())

        ax[i, j].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
        ax[i, j].set_aspect('equal')  # 设置每个格子为正方形
        ax[i, j].axis('off')  # 关闭坐标轴

plt.tight_layout()  # 调整子图的布局
plt.show()  # 显示图像


# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：draw_hotmap_ordered.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/6/22 11:54 
"""
import pandas as pd
import sys

# sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/CNNC-master/utils")

import numpy as np

file_name = "/mnt/dandancao/public/for_yijun/avg_expr.csv"

data = pd.read_csv(file_name,index_col=0)
import numpy as np
import matplotlib.pyplot as plt

# 创建一个示例数据
data = np.random.rand(10, 10)

# 绘制热力图
plt.imshow(data, cmap='hot', interpolation='nearest')

# 添加颜色栏
plt.colorbar()

# 显示图形
plt.show()

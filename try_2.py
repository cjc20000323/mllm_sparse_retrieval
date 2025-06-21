import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = [800, 3000, 4000, 5000]  # X轴数据: 1到10
y1 = [0.7976, 0.817, 0.8136, 0.8134]  # Y轴数据
y2 = [0.3554, 0.3648, 0.3486, 0.3708]
y3 = [0.5658, 0.5894, 0.5742, 0.5906]

# 创建图形和坐标轴
plt.figure(figsize=(10, 6))  # 设置图形大小

# 绘制折线图
'''
plt.plot(x, y1)
plt.scatter(29800, 0.8398, s=300, c='green', alpha=0.7, edgecolor='black')
'''

'''
plt.plot(x, y2)
plt.scatter(29800, 0.3182, s=300, c='red', alpha=0.7, edgecolor='black')

'''
plt.plot(x, y3)
plt.scatter(29800, 0.5524, s=300, c='blue', alpha=0.7, edgecolor='black')



# 添加标题和标签
plt.xlabel('few-shot sum', fontsize=12)
plt.ylabel('r@1', fontsize=12)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图形
plt.tight_layout()
plt.show()
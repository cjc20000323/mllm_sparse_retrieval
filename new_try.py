import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.array([9, 16, 48, 80, 128])  # 生成从0到10的100个等间距点
y1 = np.array([0.9378, 0.9515, 0.9668, 0.9714, 0.9765])
y2 = np.array([0.9745, 0.9825, 0.9875, 0.9895, 0.9905])

y3 = np.array([0.7754, 0.778, 0.777333333, 0.774866667, 0.776133333])
y4 = np.array([0.846, 0.846333333, 0.846333333, 0.843666667, 0.843333333])
plt.figure()
plt.plot(x, y1, label='t2i', color='blue', linestyle='-', linewidth=2)   # 自定义样式
plt.plot(x, y2, label='i2t', color='red', linestyle='--', linewidth=2)  # 自定义样式
plt.title('Sparse Retrieval (R@100+R@200)/2')
plt.xlabel('sparse length')
plt.ylabel('r@k')
plt.legend()  # 显示图例
plt.grid(True)
plt.show()

plt.figure()
plt.plot(x, y3, label='t2i', color='blue', linestyle='-', linewidth=2)   # 自定义样式
plt.plot(x, y4, label='i2t', color='red', linestyle='--', linewidth=2)  # 自定义样式
plt.title('Hybrid Retrieval (R@1+R@5+R@10)/3')
plt.xlabel('sparse length')
plt.ylabel('r@k')
plt.legend()  # 显示图例
plt.grid(True)
plt.show()

x = np.array([20, 30, 40])  # 生成从0到10的100个等间距点
y1 = np.array([0.809, 0.810533333, 0.811333333])
y2 = np.array([0.872333333, 0.873333333, 0.867666667])
y3 = np.array([0.840666667, 0.841933333, 0.8395])
plt.figure()
plt.plot(x, y1, label='t2i', color='blue', linestyle='-', linewidth=2)   # 自定义样式
plt.plot(x, y2, label='i2t', color='red', linestyle='--', linewidth=2)  # 自定义样式
plt.plot(x, y3, label='mean', color='green', linestyle='--', linewidth=2)  # 自定义样式
plt.title('Hybrid Retrieval (R@1+R@5+R@10)/3')
plt.xlabel('sparse length')
plt.ylabel('r@k')
plt.legend()  # 显示图例
plt.grid(True)
plt.show()
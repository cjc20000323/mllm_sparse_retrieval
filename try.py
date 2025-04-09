import numpy as np
from matplotlib import pyplot as plt

data1 = np.random.normal(0, 1, 500)
data2 = np.random.normal(3, 1.5, 200)
print(data1)
print(data2)
# 绘制叠加直方图
plt.hist(data1, bins=30, alpha=0.5, label='Group 1', color='blue')
plt.hist(data2, bins=30, alpha=0.5, label='Group 2', color='red')

# 添加图例和标签
plt.legend(loc='upper right')
plt.title('Comparison of Two Groups')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.show()

import string
print(string.punctuation)
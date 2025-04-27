import torch
from matplotlib import pyplot as plt

x = [100, 200, 400, 800, 1000, 1200, 2000]
y1 = [0.343902439, 0.388004798, 0.417153139, 0.438544582, 0.430507797, 0.432946821, 0.426989204]
y2 = [0.110235906, 0.127229108, 0.153538585, 0.183726509, 0.165333866, 0.184686126, 0.163534586]
y3 = [0.280807677, 0.271851259, 0.299840064, 0.359416234, 0.33942423, 0.348140744, 0.33682527]

# 绘制两条线
plt.plot(x, y1, label='dense', color='red', linestyle='-')
plt.plot(x, y2, label='sparse', color='green', linestyle='-')
plt.plot(x, y3, label='fusion/hybrid', color='blue', linestyle='-')

# 添加标注
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.show()
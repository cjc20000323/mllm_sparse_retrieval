import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 数据准备
categories = ['A', 'B', 'C', 'D']
values = [5, 10, 15, 20]
total_frames = 100  # 总动画帧数
interval_ms = 10    # 帧间隔（毫秒）

# 创建画布和坐标轴
fig, ax = plt.subplots()
bars = ax.bar(categories, [0]*len(values))  # 初始化高度为0的柱子
patches = bars.patches  # 获取所有柱子对象

# 设置图表样式
ax.set_ylim(0, max(values)*1.1)  # y轴留10%余量
ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Animated Bar Growth')

def init():
    """初始化函数：重置柱子高度"""
    for bar in patches:
        bar.set_height(0)
    return patches

def update(frame):
    """动画更新函数：计算当前高度"""
    progress = (frame + 1) / total_frames  # 计算动画进度
    for i, bar in enumerate(patches):
        bar.set_height(values[i] * progress)  # 设置当前帧高度
    return patches

# 创建动画
ani = FuncAnimation(
    fig=fig,
    func=update,
    frames=total_frames,
    init_func=init,
    blit=True,        # 优化渲染
    interval=interval_ms,
    repeat=False  # 关键修改：禁止重复播放
)

plt.show()
"""
# @Time: 2025/4/4 12:58
# @File: draw_figure.py
"""
import numpy as np
from matplotlib import pyplot as plt


def draw_bar(categories, values, title="", x_label="", y_label=""):
    bars = plt.bar(categories, values,
                   color='#4C72B0',  # 柱体颜色
                   width=0.5)  # 柱体宽度
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.,
                 height,
                 f'{height}',
                 ha='center',
                 va='bottom')
    plt.xticks(
        ticks=categories,  # 指定刻度的位置（必须与柱的位置一致）
        labels=categories,  # 指定显示的标签文本
        rotation=15,  # 标签旋转角度（避免重叠）
        ha='center'  # 对齐方式（右对齐）
    )
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # 水平虚线网格
    plt.tight_layout()  # 自动调整布局
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()


if __name__ == '__main__':
    c = ["a", "b", "d"]
    v = [np.random.randint(10, 100) for _ in c]
    draw_bar(c, v)

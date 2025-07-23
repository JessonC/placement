# visualize_best5.py

import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_pcb_data(data, title="PCB Layout"):
    """
    绘制单个 PCB 布局数据（cellList + netList）。
    """
    cell_list = data["cellList"]
    net_list  = data.get("netList", [])

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab20.colors

    # 画器件轮廓和引脚
    for idx, cell in enumerate(cell_list):
        c = colors[idx % len(colors)]
        contour = np.array(eval(cell["contour"]))
        xs = np.append(contour[:, 0], contour[0, 0])
        ys = np.append(contour[:, 1], contour[0, 1])
        ax.plot(xs, ys, color=c, linewidth=2)

        # 器件名称
        cx, cy = eval(cell["center"])
        ax.text(cx, cy, cell["cellName"],
                color=c, fontsize=9, ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor=c, boxstyle='round,pad=0.2', alpha=0.8))

        # 引脚位置
        for pin in cell["pinList"]:
            px, py = eval(pin["center"])
            ax.plot(px, py, marker='o', color=c, markersize=4, alpha=0.8)

    # 画连线 (nets)
    for net in net_list:
        pin_coords = []
        for ref in net["pinList"]:
            # 找出对应 cell 和 pin
            cell = next(c for c in cell_list if c["cellName"] == ref["cellName"])
            pin  = next(p for p in cell["pinList"] if p["pinName"] == ref["pinName"])
            pin_coords.append(eval(pin["center"]))
        # 依次连线
        for i in range(len(pin_coords) - 1):
            x1, y1 = pin_coords[i]
            x2, y2 = pin_coords[i + 1]
            ax.plot([x1, x2], [y1, y2], color='gray', linestyle='-', linewidth=1, alpha=0.7)

    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 读取保存的 top5 快照
    with open("best5_envs.pkl", "rb") as f:
        best5 = pickle.load(f)

    # 按 rank 从高到低依次可视化
    for key, data in best5.items():
        plot_pcb_data(data, title=key)

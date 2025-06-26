import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
# from sympy.printing.pretty.pretty_symbology import line_width

# area的数据 (外部多边形)
area = [[25211.670609982186, 23026.64054039088], [25210.700279714358, 23005.553960242258], [25197.49270977967, 22972.95896735191],
         [25042.4631659487, 22867.138671875], [24838.01172726787, 22867.138671875], [24820.124328220205, 22900.589516550746],
         [24814.436783978494, 22919.035963881837], [24089.91284636492, 27012.048063099897], [24066.403907080727, 27395.402289497822],
         [24065.87039157395, 27438.08353003989], [24066.854407785642, 27459.467524132742], [24078.13835403946, 27487.315201619],
         [24170.754315452406, 27581.08167186937], [24676.422820929696, 27859.539446599723], [24679.729407659954, 27861.308679191017],
         [24772.8286344145, 27909.7421875], [25075.984411070982, 27909.7421875], [25198.614638612224, 27782.595590660945],
         [25208.907090242566, 27644.419484539205], [25211.670609982186, 23026.64054039088]]

# 多个comp_shape的数据 (每个comp_shape代表一个内部多边形)
comp_shapes = [
    [[-30.713204078150017, -15.764920542537082], [-34.009400953259274, -6.7374597312345745],
     [-34.3743088737678, 4.6442119210912125], [-30.86648194803409, 15.477990470189576],
     [-30.671800000000076, 15.748999999999796], [3.936999999998079, 15.75], [30.5753999999979, 15.75],
     [30.887879758848623, 15.43738935029902], [34.0520309691189, 6.528691254976008],
     [34.34378340975806, -4.855090537784577], [30.766499999997905, -15.748999999999796],
     [-30.713204078150017, -15.764920542537082]],
    [[-105.10229999999865, -96.0], [-111.37800000000061, -89.72429999999804], [-111.37800000000061, 88.46459999999934],
     [-103.84219999999914, 96.0], [103.84219999999914, 96.0], [111.37800000000061, 88.46459999999934],
     [111.37800000000061, -89.72429999999804], [105.10229999999865, -96.0], [-105.10229999999865, -96.0]]
]

import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_demo(area, comp_shapes, test_id, folder_path):
    """
    绘制区域和多个形状，支持 torch.Tensor 格式的 comp_shapes
    """
    # 将 area 转换为 NumPy 数组
    area = np.array(area)

    # 将 comp_shapes 转换为 NumPy 数组
    comp_shapes = [comp.numpy() if isinstance(comp, torch.Tensor) else np.array(comp) for comp in comp_shapes]
    print("len", len(area))
    print("len", len(comp_shapes))
    # 创建一个绘图窗口
    # plt.figure(figsize=(50, 50))
    plt.figure(figsize=(max(10, len(area)//2), max(8, len(comp_shapes) // 5)))  # 根据数据量调整大小

    # 绘制 area (外部多边形)
    plt.plot(area[:, 0], area[:, 1], label="Area Shape", color="blue", linestyle='-')

    # 绘制多个 comp_shapes (多个内部多边形)
    for i, comp_shape in enumerate(comp_shapes):
        plt.plot(comp_shape[:, 0], comp_shape[:, 1], label=f"Comp Shape {i+1}", linestyle='-')

    # 添加标签和图例
    plt.title("Area and Multiple Component Shapes")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    # plt.legend()
    # 将图例放置在右上角，并避免遮挡图形
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    # 显示图形
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # 确保指定文件夹存在，不存在则创建
    # folder_path = 'Demo/placement'
    # folder_path = 'Demo/placement/train'
    os.makedirs(folder_path, exist_ok=True)

    plt.savefig(f"{folder_path}/data_{test_id}.png")  # 保存为静态图片


def plot_demo_ppo(area, comp_shapes, pins, test_id, folder_path, env):
    """
    绘制区域和多个形状，支持 torch.Tensor 格式的 comp_shapes
    """
    # 将 area 转换为 NumPy 数组
    area = np.array(area)

    # 将 comp_shapes 转换为 NumPy 数组
    comp_shapes = [comp.numpy() if isinstance(comp, torch.Tensor) else np.array(comp) for comp in comp_shapes]
    comp_names = env.comp_names

    # 创建一个绘图窗口
    # plt.figure(figsize=(50, 50))
    plt.figure(figsize=(max(10, len(area)//2), max(8, len(comp_shapes) // 5)))  # 根据数据量调整大小

    # 绘制 area (外部多边形)
    plt.plot(area[:, 0], area[:, 1], label="Area Shape", color="blue", linestyle='-')

    # 绘制多个 comp_shapes (多个内部多边形)
    for i, comp_shape in enumerate(comp_shapes):
        plt.plot(comp_shape[:, 0], comp_shape[:, 1], label=f"Comp Shape {i+1}", linestyle='-')

        # 计算多边形的中心点作为组件名称的显示位置
        center_x = np.mean(comp_shape[:, 0])
        center_y = np.mean(comp_shape[:, 1])
        # 添加组件名称
        comp_name = comp_names[i] if i < len(comp_names) else f"Comp {i + 1}"
        plt.text(center_x, center_y, comp_name, fontsize=6, color="red", ha='center', va='center')
        plt.savefig(f"{folder_path}/ppo_data_{test_id + 4 + i}.png")  # 保存为静态图片

    # 绘制 pin 脚的坐标和名称
    for pin in pins:
        pin_x, pin_y = pin['pin_x'], pin['pin_y']
        pin_net_name = pin['pin_net_name']
        comp_name = pin['comp_name']

        # 绘制 pin 的位置（使用小圆点表示）
        plt.scatter(pin_x, pin_y, color='green', s=6,
                        label='Pin Position' if 'Pin Position' not in plt.gca().get_legend_handles_labels()[1] else "")

        # # 在 pin 附近显示名称（网络名称和组件名）
        # plt.text(pin_x, pin_y, f"{pin_net_name}", fontsize=3, color="purple", ha='center', va='center')

    # 分组：按照 pin_net_name 将 pins 分组
    pins_by_net = defaultdict(list)
    # 按照放置顺序对 pin 列表排序
    pins = sorted(pins, key=lambda pin: env.layout_order.index(pin['comp_name']))
    for pin in pins:
        pins_by_net[pin['pin_net_name']].append((pin['pin_x'], pin['pin_y']))
    print("pin_by_net", pins_by_net)
    print("pins", pins)

    # 连线：将相同 pin_net_name 的 pin 连线
    for net_name, coordinates in pins_by_net.items():
        if len(coordinates) > 1:  # 至少两个点才能连线
            coordinates = np.array(coordinates)  # 转换为 NumPy 数组
            print("coordinates", coordinates)
            plt.plot(coordinates[:, 0], coordinates[:, 1], linestyle='-', marker='o', label=f"Net: {net_name}", linewidth=0.5)
    # 添加标签和图例
    plt.title("Area and Multiple Component Shapes")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    # plt.legend()
    # # 将图例放置在右上角，并避免遮挡图形
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    # 显示图形
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # 确保指定文件夹存在，不存在则创建
    # folder_path = 'Demo/placement'
    # folder_path = 'Demo/placement/train'
    os.makedirs(folder_path, exist_ok=True)

    plt.savefig(f"{folder_path}/ppo_data_{test_id -1}.png")  # 保存为静态图片
    print("已经保存为" + f"{folder_path}/ppo_data_{test_id -1}.png")


if __name__ == '__main__':
    plot_demo(area, comp_shapes, test_id=0, folder_path='demo')

from collections import defaultdict

import networkx as nx
import numpy as np

base_reward = 50000





def calculate_layout_area(placement_shape):
    """
    placement_shape是所有器件的形状
    """
    # 计算整体布局面积
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    # 遍历 self.placement_shape
    for shape in placement_shape:
        # 检查 shape 类型
        if isinstance(shape, list):  # 如果是 list 格式的多边形点坐标
            x_coords = [point[0] for point in shape]
            y_coords = [point[1] for point in shape]
        elif isinstance(shape, np.ndarray):  # 如果是 numpy 数组格式
            x_coords = shape[:, 0]
            y_coords = shape[:, 1]
        else:
            raise ValueError(f"Unsupported shape format: {type(shape)}")

        # 更新全局的最小和最大坐标
        x_min = min(x_min, min(x_coords))
        y_min = min(y_min, min(y_coords))
        x_max = max(x_max, max(x_coords))
        y_max = max(y_max, max(y_coords))

    # 计算布局面积
    layout_area = (x_max - x_min) * (y_max - y_min)
    return layout_area


def calculate_mst_manhattan(pins):
    """
    基于曼哈顿距离计算最小生成树的总线长
    :param pins: 引脚的坐标列表 [(x1, y1), (x2, y2), ...]
    :return: 最小生成树的曼哈顿距离总和
    """
    # 构建完整的图
    graph = nx.Graph()

    # 添加边，权重为曼哈顿距离
    for i in range(len(pins)):
        for j in range(i + 1, len(pins)):
            x1, y1 = pins[i]
            x2, y2 = pins[j]
            manhattan_distance = abs(x1 - x2) + abs(y1 - y2)
            graph.add_edge(i, j, weight=manhattan_distance)

    # 计算最小生成树
    mst = nx.minimum_spanning_tree(graph, algorithm="kruskal")

    # 累加最小生成树中的所有边的权重
    total_length = sum(edge[2]['weight'] for edge in mst.edges(data=True))
    return total_length

def calculate_hpwl(pins):
    """
    使用 HPWL 模型计算一个网络的线长
    :param pins: 引脚坐标列表 [(x1, y1), (x2, y2), ...]
    :return: 该网络的 HPWL
    """
    x_coords = [pin[0] for pin in pins]
    y_coords = [pin[1] for pin in pins]

    # 计算外包矩形的半周长
    hpwl = (max(x_coords) - min(x_coords)) + (max(y_coords) - min(y_coords))
    return hpwl


def calculate_total_hpwl(pins):
    """
    计算所有网络的总 HPWL
    :param pins: 包含所有引脚信息的列表，每个引脚是一个字典
    :return: 总 HPWL
    """
    # 按网络名称分组引脚
    net_to_pins = defaultdict(list)
    for pin in pins:
        net_name = pin['pin_net_name']
        net_to_pins[net_name].append((pin['pin_x'], pin['pin_y']))

    # 计算每个网络的 HPWL，并加总
    total_hpwl = 0
    for net_name, net_pins in net_to_pins.items():
        total_hpwl += calculate_hpwl(net_pins)

    return total_hpwl

def calculate_alignment_score(previous_placements):
    """
            计算所有器件之间的对齐度得分
            :return: 对齐度得分（值越高，表示器件越对齐）
            """
    total_alignment_score = 0
    num_comparisons = 0

    # 遍历每一对器件
    for i in range(len(previous_placements)):
        for j in range(i + 1, len(previous_placements)):
            # 获取器件中心点坐标
            center_x1, center_y1, _ = previous_placements[i]
            center_x2, center_y2, _ = previous_placements[j]

            # 计算水平和垂直距离
            horizontal_distance = abs((center_x1 - center_x2))
            vertical_distance = abs((center_y1 - center_y2))

            # 对距离进行归一化（假设 grid_space 是单位间距）
            horizontal_score = max(0, horizontal_distance)
            vertical_score = max(0, vertical_distance)

            # 累加得分
            total_alignment_score += (horizontal_score + vertical_score) / 2
            num_comparisons += 1

    # 返回平均对齐度得分
    return total_alignment_score / num_comparisons if num_comparisons > 0 else 0


def calculate_net_wire_length(net_pins):
    """
    计算 net_pins 的总线长，按照相邻点之间的曼哈顿距离累加
    :param net_pins: 引脚的坐标列表 [(x1, y1), (x2, y2), ...]
    :return: net_pins 的总线长
    """
    net_length = 0  # 初始化当前 net_pins 的总线长

    # 遍历 net_pins，依次计算相邻点之间的曼哈顿距离
    for i in range(len(net_pins) - 1):
        x1, y1 = net_pins[i]  # 前一个点的坐标
        x2, y2 = net_pins[i + 1]  # 后一个点的坐标
        manhattan_distance = abs(x1 - x2) + abs(y1 - y2)  # 计算曼哈顿距离
        net_length += manhattan_distance  # 累加到当前 net_pins 的总线长

    return net_length


def manhattan_distance(pin1, pin2):
    return abs(pin1['pin_x'] - pin2['pin_x']) + abs(pin1['pin_y'] - pin2['pin_y'])


def calculate_reward(placement_shape, pins):
    # 计算线长
    total_wire_length = 0
    total_chip_wire_length = 0
    nets = {}
    comp_pin_count = {}  # 统计每个器件的引脚数量
    # 遍历所有 pin，并按照网络名称分组
    for pin in pins:
        comp_name = pin['comp_name']
        net_name = pin['pin_net_name']
        pin_x = pin['pin_x']
        pin_y = pin['pin_y']

        if net_name not in nets:
            nets[net_name] = []
        nets[net_name].append((pin_x, pin_y))

        if comp_name in comp_pin_count:
            comp_pin_count[comp_name] += 1
        else:
            comp_pin_count[comp_name] = 1
    # print(comp_pin_count)
    # print(nets)
    # 遍历每个网络，计算其最小生成树的总曼哈顿距离
    for net_pins in nets.values():
        if len(net_pins) > 1:  # 只有包含多个 pin 的网络才需要计算
            # 计算最小生成树的曼哈顿距离
            total_wire_length += calculate_mst_manhattan(net_pins)  # 计算总的线长

    # -------------------------------
    # 计算与核心器件的最短距离奖励
    # -------------------------------
    max_count = 0
    chip_device = None
    for comp, count in comp_pin_count.items():
        if count > max_count:
            max_count = count
            chip_device = comp

    # 创建net_to_components字典
    net_to_components = {}
    for pin in pins:
        net_name = pin['pin_net_name']
        comp_name = pin['comp_name']
        if net_name not in net_to_components:
            net_to_components[net_name] = []
        net_to_components[net_name].append(pin)

    # print("net_to_components", net_to_components)
    # 找出芯片的所有pin脚
    chip_pins = [pin for pin in pins if pin['comp_name'] == chip_device]
    # print("chip_device", chip_device)
    # print(chip_pins)

    # 遍历最核心器件的每个引脚
    for core_pin in chip_pins:
        net_name = core_pin['pin_net_name']
        net_pins = net_to_components[net_name]

        # 找到net中不属于core_device的引脚
        connected_pins = [pin for pin in net_pins if pin['comp_name'] != chip_device]
        if not connected_pins:
            continue  # 没有其他引脚，跳过

        min_distance = min(manhattan_distance(core_pin, pin) for pin in connected_pins)
        total_chip_wire_length += min_distance      # 最短线长

    # layout_area = calculate_layout_area(placement_shape)  # 计算布局面积
    # alignment_bonus = calculate_alignment_score(real_coord)     # 计算对齐程度
    # reward = base_reward - 0.003 * layout_area - 0.5 * total_wire_length - 0.3 * alignment_bonus - 8 * total_chip_wire_length
    reward = base_reward - 2 * total_wire_length - 8 * total_chip_wire_length
    return reward


if __name__ == '__main__':
    # 计算总奖励

    # layout_area = calculate_layout_area(placement_shape)    # placement_shape是所有器件的形状，弄成一个数组

    pins =  [{'comp_name': 'UD41', 'pin_net_name': 'NetCD42_1', 'pin_x': 27399.6260625, 'pin_y': 17619.649646875},
             {'comp_name': 'UD41', 'pin_net_name': 'NetCD49_2', 'pin_x': 27293.3270625, 'pin_y': 17582.248046875},
             {'comp_name': 'UD41', 'pin_net_name': 'NetCD41_1', 'pin_x': 27399.6260625, 'pin_y': 17582.248046875},
             {'comp_name': 'UD41', 'pin_net_name': 'NetRD44_1', 'pin_x': 27293.3269625, 'pin_y': 17619.649546875},
             {'comp_name': 'UD41', 'pin_net_name': 'NetRD41_1', 'pin_x': 27293.3267625, 'pin_y': 17544.846446875},
             {'comp_name': 'UD41', 'pin_net_name': 'GND', 'pin_x': 27399.6260625, 'pin_y': 17544.846346875},
             {'comp_name': 'LD41', 'pin_net_name': 'NetCD41_1', 'pin_x': 27566.4766625, 'pin_y': 17700.122146875},
             {'comp_name': 'LD41', 'pin_net_name': '3V3_STB', 'pin_x': 27566.4765625, 'pin_y': 17534.374146875},
             {'comp_name': 'CD42', 'pin_net_name': 'NetCD42_1', 'pin_x': 27258.4449625, 'pin_y': 17752.248146875},
             {'comp_name': 'CD42', 'pin_net_name': 'GND', 'pin_x': 27334.5081625, 'pin_y': 17752.248046875},
             {'comp_name': 'CD43', 'pin_net_name': 'NetCD42_1', 'pin_x': 27221.4760625, 'pin_y': 17480.515746875},
             {'comp_name': 'CD43', 'pin_net_name': 'GND', 'pin_x': 27221.477062500002, 'pin_y': 17513.980246875},
             {'comp_name': 'CD45', 'pin_net_name': '3V3_STB', 'pin_x': 27081.4770625, 'pin_y': 17613.980246875},
             {'comp_name': 'CD45', 'pin_net_name': 'GND', 'pin_x': 27081.476062499998, 'pin_y': 17580.515846875},
             {'comp_name': 'CD46', 'pin_net_name': '3V3_STB', 'pin_x': 27488.665862500002, 'pin_y': 17422.248046875},
             {'comp_name': 'CD46', 'pin_net_name': 'GND', 'pin_x': 27544.2875625, 'pin_y': 17422.248046875},
             {'comp_name': 'CD47', 'pin_net_name': '3V3_STB', 'pin_x': 27376.4762625, 'pin_y': 17364.437046875002},
             {'comp_name': 'CD47', 'pin_net_name': 'GND', 'pin_x': 27376.4762625, 'pin_y': 17420.059046875},
             {'comp_name': 'CD48', 'pin_net_name': '3V3_STB', 'pin_x': 27099.2873625, 'pin_y': 17677.248046875},
             {'comp_name': 'CD48', 'pin_net_name': 'GND', 'pin_x': 27043.665862500002, 'pin_y': 17677.248046875},
             {'comp_name': 'CD49', 'pin_net_name': 'NetCD49_2', 'pin_x': 27016.477062500002, 'pin_y': 17608.980246875002},
             {'comp_name': 'CD49', 'pin_net_name': 'GND', 'pin_x': 27016.4760625, 'pin_y': 17575.515846875},
             {'comp_name': 'RD45', 'pin_net_name': 'NetCD49_2', 'pin_x': 27744.7441625, 'pin_y': 17627.248046875},
             {'comp_name': 'RD45', 'pin_net_name': 'GND', 'pin_x': 27778.2087625, 'pin_y': 17627.248146875},
             {'comp_name': 'RD46', 'pin_net_name': 'NetCD44_1', 'pin_x': 27204.7441625, 'pin_y': 17622.248146874997},
             {'comp_name': 'RD46', 'pin_net_name': 'GND', 'pin_x': 27238.2088625, 'pin_y': 17622.248046875},
             {'comp_name': 'RD5', 'pin_net_name': 'NetCD42_1', 'pin_x': 27296.4765625, 'pin_y': 17437.027746875},
             {'comp_name': 'RD5', 'pin_net_name': '12V_5V_DCDC_IN', 'pin_x': 27296.4765625, 'pin_y': 17407.248046875}, {'comp_name': 'RD5', 'pin_net_name': '12V_5V_DCDC_IN', 'pin_x': 27296.4765625, 'pin_y': 17377.468846875}, {'comp_name': 'RD42', 'pin_net_name': 'NetCD49_2', 'pin_x': 27116.4765625, 'pin_y': 17488.980346875}, {'comp_name': 'RD42', 'pin_net_name': '12V_5V_DCDC_IN', 'pin_x': 27116.4765625, 'pin_y': 17455.515746875}, {'comp_name': 'CD41', 'pin_net_name': 'NetCD41_1', 'pin_x': 27783.2087625, 'pin_y': 17682.247546875}, {'comp_name': 'CD41', 'pin_net_name': 'NetCD41_2', 'pin_x': 27749.7442625, 'pin_y': 17682.248446875}, {'comp_name': 'RD41', 'pin_net_name': 'NetCD41_2', 'pin_x': 27141.4765625, 'pin_y': 17603.980346875}, {'comp_name': 'RD41', 'pin_net_name': 'NetCD41_2', 'pin_x': 27141.4765625, 'pin_y': 17587.248046875}, {'comp_name': 'RD41', 'pin_net_name': 'NetRD41_1', 'pin_x': 27141.4765625, 'pin_y': 17570.515746875}, {'comp_name': 'RD44', 'pin_net_name': 'NetCD44_1', 'pin_x': 27231.4765625, 'pin_y': 17682.248046875}, {'comp_name': 'RD44', 'pin_net_name': 'NetCD44_1', 'pin_x': 27214.7440625, 'pin_y': 17682.247946875003}, {'comp_name': 'RD44', 'pin_net_name': 'NetRD44_1', 'pin_x': 27248.2089625, 'pin_y': 17682.248046875}, {'comp_name': 'CD44', 'pin_net_name': '3V3_STB', 'pin_x': 27199.7442625, 'pin_y': 17417.248346875}, {'comp_name': 'CD44', 'pin_net_name': 'NetCD44_1', 'pin_x': 27233.2087625, 'pin_y': 17417.247546874998}, {'comp_name': 'RD43', 'pin_net_name': '3V3_STB', 'pin_x': 27061.476662499997, 'pin_y': 17480.515846875}, {'comp_name': 'RD43', 'pin_net_name': 'NetCD44_1', 'pin_x': 27061.476662499997, 'pin_y': 17513.980446875}]
    # 计算线长
    total_wire_length = 0
    total_chip_wire_length = 0
    nets = {}
    comp_pin_count = {}     # 统计每个器件的引脚数量
    # 遍历所有 pin，并按照网络名称分组
    for pin in pins:
        comp_name = pin['comp_name']
        net_name = pin['pin_net_name']
        pin_x = pin['pin_x']
        pin_y = pin['pin_y']

        if net_name not in nets:
            nets[net_name] = []
        nets[net_name].append((pin_x, pin_y))

        if comp_name in comp_pin_count:
            comp_pin_count[comp_name] += 1
        else:
            comp_pin_count[comp_name] = 1
    # print(comp_pin_count)
    # print(nets)
    # 遍历每个网络，计算其最小生成树的总曼哈顿距离
    for net_pins in nets.values():
        if len(net_pins) > 1:  # 只有包含多个 pin 的网络才需要计算
            # 计算最小生成树的曼哈顿距离
            total_wire_length += calculate_net_wire_length(net_pins)    # 计算总的线长


    #-------------------------------
    # 计算与核心器件的最短距离奖励
    #-------------------------------
    max_count = 0
    chip_device = None
    for comp, count in comp_pin_count.items():
        if count > max_count:
            max_count = count
            chip_device= comp

    # 创建net_to_components字典
    net_to_components = {}
    for pin in pins:
        net_name = pin['pin_net_name']
        comp_name = pin['comp_name']
        if net_name not in net_to_components:
            net_to_components[net_name] = []
        net_to_components[net_name].append(pin)

    # print("net_to_components", net_to_components)
    # 找出芯片的所有pin脚
    chip_pins = [pin for pin in pins if pin['comp_name'] == chip_device]
    # print("chip_device", chip_device)
    # print(chip_pins)

    # 遍历最核心器件的每个引脚
    for core_pin in chip_pins:
        net_name = core_pin['pin_net_name']
        net_pins = net_to_components[net_name]

        # 找到net中不属于core_device的引脚
        connected_pins = [pin for pin in net_pins if pin['comp_name'] != chip_device]
        if not connected_pins:
            continue  # 没有其他引脚，跳过

        min_distance = min(manhattan_distance(core_pin, pin) for pin in connected_pins)
        total_chip_wire_length += min_distance

    # print(chip_device)
    # print(total_wire_length)
    # alignment_bonus = calculate_alignment_score(placement_coord)    # placement_coord是所有器件的质心坐标
    # reward = base_reward - 0.1 * layout_area - 0.5 * total_wire_length - 0.4 * alignment_bonus
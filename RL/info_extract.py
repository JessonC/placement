import matplotlib
from networkx.algorithms.efficiency_measures import efficiency
# from sympy.physics.units import current
from torch import dtype

# Use 'Agg' for non-interactive backend (suitable for servers without display)
matplotlib.use('Agg')

import json
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import os
from data_process import *
from Env.meshgrid import *
from feature_calculation import *


def load_module_data(filename):
    with open(filename, 'r') as f:
        module_data = json.load(f)
    return module_data

def process_module_data(module_data):
    components = []  # List to store component data
    pins = []        # List to store pin data

    # Access the 'area' key in the JSON data, assuming it's a list of coordinates like 'comp_shape'
    area = module_data.get('area', [])

    # # Process area to extract features, similar to comp_shape

    area = np.array(area)
    area_min_x, area_min_y = area.min(axis=0)  # 未网格化前的坐标
    area_max_x, area_max_y = area.max(axis=0)

    area_grid = grid_area(area)  # 将area进行网格化
    area_min_x_grid, area_min_y_grid = area_grid.min(axis=0)
    area_max_x_grid, area_max_y_grid = area_grid.max(axis=0)
    area_width = area_max_x_grid - area_min_x_grid
    area_height = area_max_y_grid - area_min_y_grid
    area_centroid = np.round(area_grid.mean(axis=0)).astype(int)
    area_area_grid = calculate_grid_area(area_grid)
    area_perimeter_grid = calculate_polygon_perimeter(area_grid)
    area_features = {
        # 'area_width': area_width,
        # 'area_height': area_height,
        # 'area_centroid_x': area_centroid[0],
        # 'area_centroid_y': area_centroid[1],
        'area_area': area_area_grid,
        'area_perimeter_grid': area_perimeter_grid,
    }


    # Access the 'comps_info' key in the JSON data
    comps_info = module_data.get('comps_info', {})

    current_info = module_data.get('current_info', {})
    input_current = current_info['input_current']
    output_current = current_info['output_current']
    input_voltage = current_info['input_voltage']
    output_voltage = current_info['output_voltage']
    efficiency = current_info['efficenty']


    for comp_name, comp in comps_info.items():
        comp_data = {
            'comp_name': comp_name,
            'comp_rotation': comp.get('comp_rotation', 0.0),
            'comp_shape': comp.get('comp_shape', []),
            'comp_x': comp.get('comp_x', 0.0),
            'comp_y': comp.get('comp_y', 0.0),
        }

        comp_type_map = {
            'RF': 0,
            'CD': 1,
            'LD': 2,
            'RD': 3,
            'UD': 4,
            'ED': 5
        }

        comp_type = comp_type_map.get(comp_name[0:2], None)  # 如果找不到对应的键，返回 None

        # Process shape to extract features
        if comp_data['comp_shape']:
            comp_shape = np.array(comp_data['comp_shape'])
            min_x, min_y = comp_shape.min(axis=0)
            max_x, max_y = comp_shape.max(axis=0)
            comp_shape_grid = grid_comp_shape(comp_shape, area_min_x, area_min_y)     # 将comp_shape进行网格化
            # print("comp_shape_grid", comp_shape_grid)
            comp_area_grid = calculate_grid_area(comp_shape_grid)
            comp_perimeter_grid = calculate_polygon_perimeter(comp_shape_grid)
            # print("comp_area_grid", comp_area_grid)
            min_x_grid, min_y_grid = comp_shape_grid.min(axis=0)
            max_x_grid, max_y_grid = comp_shape_grid.max(axis=0)
            centroid = comp_shape_grid.mean(axis=0)
        else:
            min_x = min_y = max_x = max_y = centroid_x = centroid_y = 0.0
            min_x_grid = min_y_grid = max_x_grid = max_y_grid = centroid_x = centroid_y = 0.0
            centroid = np.array([centroid_x, centroid_y])

        # Feature vector
        # comp的原始坐标
        comp_x = comp_data['comp_x']
        comp_y = comp_data['comp_y']    # 提取出绝对坐标
        comp_rotation = comp_data['comp_rotation']
        comp_coord = np.array((comp_x, comp_y))
        # print("comp_coord:", comp_coord)

        # 网格化后的comp坐标
        comp_coord_grid = grid_comp_coord(comp_coord, area_min_x, area_min_y)
        # print("comp_coord_grid", comp_coord_grid)
        comp_x_grid = comp_coord_grid[0]
        comp_y_grid = comp_coord_grid[1]
        # print("comp_x_grid:", comp_x_grid)
        # print("comp_y_grid:", comp_y_grid)
        comp_width_grid = max_x_grid - min_x_grid
        comp_height_grid = max_y_grid - min_y_grid
        complexity = len(comp_shape_grid)
        # 网格化后的comp特征
        comp_data['features'] = [
            comp_x_grid,        # 预测网格化后的绝对坐标
            comp_y_grid,        # 预测网格化后的绝对坐标
            comp_rotation,      # 这三个布局是预测标签
            # min_x_grid - comp_x_grid, min_y_grid - comp_y_grid,   # 器件的形状是用相对坐标的形式训练和预测的
            # max_x_grid - comp_x_grid, max_y_grid - comp_y_grid,   # 器件的形状是用相对坐标的形式训练和预测的
            comp_type,
            comp_width_grid, comp_height_grid,               # 形状轮廓
            comp_width_grid / comp_height_grid,             # 形状长宽比
            complexity,                                     # 形状复杂度
            comp_area_grid,                               # comp面积
            comp_perimeter_grid,                          # comp周长
        ]
        comp_data.update(area_features)
        components.append(comp_data)
        # print("components", components)
        # Access the 'comp_pin' key for pins
        for pin_number, pin in comp.get('comp_pin', {}).items():
            pin_x = pin.get('pin_x')
            pin_y = pin.get('pin_y')
            # 得把pin_x和pin_y换成相对于comp_coord的坐标，因为测试集是这样的形式给进去的
            pin_rotation = pin.get('pin_rotation')
            pin_rotation_relative = pin_rotation - comp_rotation    # 把旋转的相对角度给进去
            pin_x_grid = round((pin_x - comp_x) / 0.1)
            pin_y_grid = round((pin_y - comp_y) / 0.1)
            pin_data = {
                'comp_name': comp_name,
                'pin_number': pin_number,
                'pin_net_name': pin.get('pin_net_name'),
                'pin_x': pin_x_grid,
                'pin_y': pin_y_grid,  # 这两个也要换成相对坐标的形式
                'pin_rotation': pin_rotation_relative,
            }
            pins.append(pin_data)

    return components, pins, area_features

def process_module_data_test(module_data):
    components = []  # List to store component data
    pins = []        # List to store pin data

    # Access the 'area' key in the JSON data, assuming it's a list of coordinates like 'comp_shape'
    area = module_data.get('area', [])


    # # Process area to extract features, similar to comp_shape
    if area:
        area = np.array(area)
        area_min_x, area_min_y = area.min(axis=0)   # 未网格化前的坐标
        area_max_x, area_max_y = area.max(axis=0)

        area_grid = grid_area(area)  # 将area进行网格化
        area_min_x_grid, area_min_y_grid = area_grid.min(axis=0)
        area_max_x_grid, area_max_y_grid = area_grid.max(axis=0)
        area_width = area_max_x_grid - area_min_x_grid
        area_height = area_max_y_grid - area_min_y_grid
        # area_centroid = round(area_grid.mean(axis=0))
        area_centroid = np.round(area_grid.mean(axis=0)).astype(int)
        area_area_grid = calculate_grid_area(area_grid)
        area_perimeter_grid = calculate_polygon_perimeter(area_grid)
        area_features = {
            # 'area_width': area_width,
            # 'area_height': area_height,
            # 'area_centroid_x': area_centroid[0],
            # 'area_centroid_y': area_centroid[1],
            'area_area': area_area_grid,
            'area_perimeter_grid': area_perimeter_grid
        }

        area_origin = {'area_min_x': area_min_x, 'area_min_y': area_min_y}  # 网格原点的坐标值
        # print(area_max_x - area_min_x)
        # print(area_max_y - area_min_y)
        # print(area_centroid[0])
        # print(area_centroid[1])
    else:
        area_features = {
            'area_min_x': 0.0,
            'area_min_y': 0.0,
            'area_max_x': 0.0,
            'area_max_y': 0.0,
            'area_width': 0.0,
            'area_height': 0.0,
            'area_centroid_x': 0.0,
            'area_centroid_y': 0.0,
        }

    # Access the 'comps_info' key in the JSON data
    comps_info = module_data.get('comps_info', {})

    for comp_name, comp in comps_info.items():
        comp_data = {
            'comp_name': comp_name,
            'comp_rotation': comp.get('comp_rotation', 0.0),
            'comp_shape': comp.get('comp_shape', []),
            'comp_x': comp.get('comp_x', 0.0),
            'comp_y': comp.get('comp_y', 0.0),
        }

        comp_type_map = {
            'RF': 0,
            'CD': 1,
            'LD': 2,
            'RD': 3,
            'UD': 4,
            'ED': 5,
        }

        comp_type = comp_type_map.get(comp_name[0:2], None)  # 如果找不到对应的键，返回 None


        # Process shape to extract features
        if comp_data['comp_shape']:
            comp_shape = np.array(comp_data['comp_shape'])
            min_x, min_y = comp_shape.min(axis=0)
            max_x, max_y = comp_shape.max(axis=0)
            comp_shape_grid = np.round(comp_shape / 0.1)     # 将comp_shape进行网格化

            min_x_grid = round(min_x / 0.1)
            min_y_grid = round(min_y / 0.1)
            max_x_grid = round(max_x / 0.1)
            max_y_grid = round(max_y / 0.1)    # 已经是相对坐标了，直接除以grid的精度即可
            comp_area_grid = calculate_grid_area(comp_shape_grid)
            comp_perimeter_grid = calculate_polygon_perimeter(comp_shape_grid)
            centroid = comp_shape.mean(axis=0)
        else:
            min_x = min_y = max_x = max_y = centroid_x = centroid_y = 0.0
            min_x_grid = min_y_grid = max_x_grid = max_y_grid = centroid_x = centroid_y = 0.0
            centroid = np.array([centroid_x, centroid_y])

        # Feature vector
        # comp的原始坐标
        comp_x = comp_data['comp_x']
        comp_y = comp_data['comp_y']    # 提取出绝对坐标
        comp_rotation = comp_data['comp_rotation']
        comp_coord = np.array((comp_x, comp_y))     # [0, 0]
        # print("comp_coord:", comp_coord)

        # 网格化后的comp坐标
        # comp_coord_grid = grid_comp_coord(comp_coord, area_min_x, area_min_y)
        # # print("comp_coord_grid", comp_coord_grid)
        # comp_x_grid = comp_coord_grid[0]
        # comp_y_grid = comp_coord_grid[1]
        # print("comp_x_grid:", comp_x_grid)
        # print("comp_y_grid:", comp_y_grid)
        comp_width_grid = max_x_grid - min_x_grid
        comp_height_grid = max_y_grid - min_y_grid
        comp_x_grid, comp_y_grid = 0, 0
        complexity = len(comp_shape_grid)
        # 网格化后的comp特征
        comp_data['features'] = [
            comp_x_grid,        # 预测网格化后的绝对坐标
            comp_y_grid,        # 预测网格化后的绝对坐标
            comp_rotation,      # 这三个布局是预测标签
            # min_x_grid - comp_x_grid, min_y_grid - comp_y_grid,   # comp_x_grid和comp_y_grid的值都为0
            # max_x_grid - comp_x_grid, max_y_grid - comp_y_grid,   # comp_x_grid和comp_y_grid的值都为0
            comp_type,
            comp_width_grid, comp_height_grid,               # 形状轮廓
            comp_width_grid / comp_height_grid,                # 形状长宽比
            complexity,
            comp_area_grid,                                  # 形状面积
            comp_perimeter_grid,                             # 形状周长
        ]
        # print("comp_width:", max_x_grid - min_x_grid)
        # print("comp_height:", max_y_grid - min_y_grid)
        comp_data.update(area_features)
        components.append(comp_data)
        # print("components", components)
        # Access the 'comp_pin' key for pins
        for pin_number, pin in comp.get('comp_pin', {}).items():
            pin_x = pin.get('pin_x')
            pin_y = pin.get('pin_y')
            # 得把pin_x和pin_y换成相对于comp_coord的坐标，因为测试集是这样的形式给进去的
            pin_x_grid = round((pin_x - 0) / 0.1)
            pin_y_grid = round((pin_y - 0) / 0.1)
            pin_data = {
                'comp_name': comp_name,
                'pin_number': pin_number,
                'pin_net_name': pin.get('pin_net_name'),
                'pin_x': pin_x_grid,
                'pin_y': pin_y_grid,  # 这两个也要换成相对坐标的形式
                'pin_rotation': pin.get('pin_rotation'),
            }
            pins.append(pin_data)

    return components, pins, area_features, area_origin


def collect_components_and_pins(file_list):
    """
    遍历所有JSON文件，提取组件和引脚信息，统计每个组件的引脚数量
    """
    comp_pin_count = defaultdict(int)  # 记录每个组件的最大引脚数量

    for filename in file_list:
        module_data = load_module_data(filename)
        components, pins, area_features = process_module_data(module_data)

        # 统计引脚数量
        comp_dict = defaultdict(int)
        for pin in pins:
            comp_name = pin['comp_name']
            comp_dict[comp_name] += 1

        for comp_name, count in comp_dict.items():
            if count > comp_pin_count[comp_name]:
                comp_pin_count[comp_name] = count


    # 找到所有组件中最大的引脚数量
    max_pins = max(comp_pin_count.values()) if comp_pin_count else 0

    return max_pins

def process_pin_data(pins, comp_names, max_pins):
    """
    处理pins信息，生成统一长度的引脚特征矩阵
    """
    # Initialize a dictionary to group data by comp_name
    comp_dict = defaultdict(list)

    # Group pin information by comp_name
    for item in pins:
        comp_name = item['comp_name']
        pin_info = [
            item['pin_x'],
            item['pin_y'],
            item['pin_rotation']
        ]
        comp_dict[comp_name].append(pin_info)

    # Prepare the data for the tensor
    fill_value = -5e10  # Value to fill in for missing pins
    array_data = []

    for comp_name in comp_names:
        pins = comp_dict.get(comp_name, [])
        # Flatten the pin data for the component
        # Each pin contributes three values: pin_x, pin_y, pin_rotation
        flat_pins = [value for pin in pins for value in pin]
        # Calculate how many values are missing to reach max_pins * 3
        missing_values = max_pins * 3 - len(flat_pins)
        # Pad the flat_pins list with the fill_value
        flat_pins.extend([fill_value] * missing_values)
        # Append to the list
        array_data.append(flat_pins)

    if array_data:
        pin_info = torch.tensor(array_data, dtype=torch.float)
    else:
        # Handle case with no pins
        pin_info = torch.empty((0, max_pins * 3), dtype=torch.float)

    return pin_info

def build_graph(components, pins, max_pins, area_features):
    # Build net to components mapping
    net_to_comps = defaultdict(set)

    for pin in pins:
        net_name = pin['pin_net_name']
        comp_name = pin['comp_name']
        if net_name:
            net_to_comps[net_name].add(comp_name)

    # Build edges
    edges = set()

    for net, comps in net_to_comps.items():
        comps = list(comps)
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                src = comps[i]
                dst = comps[j]
                # Ensure consistent ordering to avoid duplicate edges
                edge = (min(src, dst), max(src, dst))
                edges.add(edge)

    # Assign IDs to components
    comp_name_to_id = {comp['comp_name']: idx for idx, comp in enumerate(components)}
    id_to_comp_name = {idx: comp['comp_name'] for idx, comp in enumerate(components)}
    comp_names = [comp['comp_name'] for comp in components]

    # Node features
    comp_feature = [comp['features'] for comp in components]
    comp_feature = torch.tensor(comp_feature, dtype=torch.float)
    # coordinate = [comp['area_features'] for comp in components]
    # print("coordinate", coordinate)
    pin_feature = process_pin_data(pins, comp_names, max_pins)
    # print("Y", Y)
    area_data = torch.tensor(list(area_features.values()), dtype=torch.float)  # area_data为4维，分别为横宽，纵宽，横中心和纵中心,还有面积
    area_data = area_data.repeat(comp_feature.shape[0], 1)
    # print("area_feature", area_data.shape)
    # Edge index
    edges_id = [(comp_name_to_id[src], comp_name_to_id[dst]) for src, dst in edges]
    if edges_id:
        edge_index = torch.tensor(edges_id, dtype=torch.long).t().contiguous()
    else:
        # Handle case with no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Create data object
    data = Data(x=comp_feature, y=area_data, z=pin_feature, edge_index=edge_index)    # X: (43, 11), comp_features, 第一维为器件特征 Y: (43, 123) pin_features
    # data.coordinate = torch.tensor([area_min_x, area_min_y], dtype=torch.float)
    # data.min_area_x = torch.tensor([area_min_x], dtype=torch.float)  # Add min_area_x as an attribute
    # data.min_area_y = torch.tensor([area_min_y], dtype=torch.float)  # Add min_area_y as an attribute
    # print("data", data)
    return data, comp_name_to_id, id_to_comp_name

def build_graph_test(components, pins, max_pins, area_features, area_origin):
    # Build net to components mapping
    net_to_comps = defaultdict(set)

    for pin in pins:
        net_name = pin['pin_net_name']
        comp_name = pin['comp_name']
        if net_name:
            net_to_comps[net_name].add(comp_name)

    # Build edges
    edges = set()

    for net, comps in net_to_comps.items():
        comps = list(comps)
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                src = comps[i]
                dst = comps[j]
                # Ensure consistent ordering to avoid duplicate edges
                edge = (min(src, dst), max(src, dst))
                edges.add(edge)

    # Assign IDs to components
    comp_name_to_id = {comp['comp_name']: idx for idx, comp in enumerate(components)}
    id_to_comp_name = {idx: comp['comp_name'] for idx, comp in enumerate(components)}
    comp_names = [comp['comp_name'] for comp in components]

    # Node features
    comp_feature = [comp['features'] for comp in components]
    comp_feature = torch.tensor(comp_feature, dtype=torch.float)
    # coordinate = [comp['area_features'] for comp in components]
    # print("coordinate", coordinate)
    pin_feature = process_pin_data(pins, comp_names, max_pins)
    # print("Y", Y)
    area_data = torch.tensor(list(area_features.values()), dtype=torch.float)  # area_data为4维，分别为横宽，纵宽，横中心和纵中心
    area_data = area_data.repeat(comp_feature.shape[0], 1)
    # print("area_feature", area_data.shape)
    area_origin_data = torch.tensor(list(area_origin.values()), dtype=torch.float).unsqueeze(0)     # 网格化前的原点坐标值
    area_origin_data = area_origin_data.repeat(comp_feature.shape[0], 1)
    # print("area_origin", area_origin_data.shape)
    # Edge index
    edges_id = [(comp_name_to_id[src], comp_name_to_id[dst]) for src, dst in edges]
    if edges_id:
        edge_index = torch.tensor(edges_id, dtype=torch.long).t().contiguous()
    else:
        # Handle case with no edges
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Create data object
    data = Data(x=comp_feature, y=area_data, z=pin_feature, k = area_origin_data, edge_index=edge_index)    # X: (43, 11), comp_features, 第一维为器件特征 Y: (43, 123) pin_features
    # data.coordinate = torch.tensor([area_min_x, area_min_y], dtype=torch.float)
    # data.min_area_x = torch.tensor([area_min_x], dtype=torch.float)  # Add min_area_x as an attribute
    # data.min_area_y = torch.tensor([area_min_y], dtype=torch.float)  # Add min_area_y as an attribute
    # print("data", data)
    return data, comp_name_to_id, id_to_comp_name

def visualize_graph(data, id_to_comp_name, filename):
    # Convert PyTorch Geometric data to NetworkX data
    G = nx.Graph()

    # Add nodes with labels
    for idx in range(data.num_nodes):
        comp_name = id_to_comp_name[idx]
        G.add_node(idx, label=comp_name)

    # Add edges
    edge_index = data.edge_index.numpy()
    for src, dst in edge_index.T:
        G.add_edge(src, dst)

    # Draw the data
    plt.figure(figsize=(12, 8))

    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes with labels
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7)

    plt.title(f'Component Connection Graph - {filename}')
    plt.axis('off')

    # Save the plot to a file
    plt.savefig(f'./graph/train/graph_{filename}.png', dpi=300)
    plt.close()
    # If using interactive backend, you can display the plot
    # plt.show()

def visualize_graph_test(data, id_to_comp_name, filename):
    # Convert PyTorch Geometric data to NetworkX data
    G = nx.Graph()

    # Add nodes with labels
    for idx in range(data.num_nodes):
        comp_name = id_to_comp_name[idx]
        G.add_node(idx, label=comp_name)

    # Add edges
    edge_index = data.edge_index.numpy()
    for src, dst in edge_index.T:
        G.add_edge(src, dst)

    # Draw the data
    plt.figure(figsize=(12, 8))

    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes with labels
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7)

    plt.title(f'Component Connection Graph - {filename}')
    plt.axis('off')

    # Save the plot to a file
    plt.savefig(f'./graph/test/graph_{filename}.png', dpi=300)
    plt.close()
    # If using interactive backend, you can display the plot
    # plt.show()






if __name__ == '__main__':
    # 列出所有JSON文件
    file_list = ['./data/data1.json', './data/data2.json', './data/data3.json', './data/data4.json', './data/data5.json']

    # 首先遍历所有文件，收集组件和引脚信息，统计max_pins
    max_pins = collect_components_and_pins(file_list)
    print(f"Global max_pins: {max_pins}")

    # 存储每个图的数据对象
    data_list = []

    # 逐个处理每个文件
    for idx, filename in enumerate(file_list):
        module_data = load_module_data(filename)
        components, pins, area_features = process_module_data(module_data)

        # 构建图，使用全局的max_pins
        data, comp_name_to_id, id_to_comp_name = build_graph(components, pins, max_pins, area_features)
        # data里包含x(comp_feature), y(pins_feature)和edge_index

        # 可视化图
        visualize_graph(data, id_to_comp_name, os.path.basename(filename))

        # 将节点特征和引脚信息拼接
        data_combined = torch.cat((data.x, data.y, data.z), dim=1)
        # data_combined_np = data_combined.cpu().detach().numpy()
        # 划分特征和标签
        data_x, data_y = divide_dataset(data_combined)
        # print("data_x", data_x.shape)
        # print("data_y", data_y)
        # 标准化（可选，根据需求）
        # data_norm_x, data_norm_y, mean_x, std_x, mean_y, std_y = data_normalization(data_x, data_y)

        # 将处理后的数据存储到Data对象中
        data.x = data_x
        data.y = data_y

        # 将Data对象添加到列表中
        data_list.append(data)

    data_list_norm = data_normalization(data_list)
    # print("data_list_norm", data_list_norm.y)
    # print(data_list[0].x)
        # 保存Data对象到文件（可选）
        # torch.save(data, f'module_graph_data_{idx+1}.pt')

        # # 打印一些信息
        # print(f"\nProcessed {filename}:")
        # print(f"Number of components (nodes): {data.num_nodes}")
        # print(f"Number of connections (edges): {data.num_edges}")
        # print(f"Node feature matrix shape: {data.x.shape}")
        # print(f"Edge index shape: {data.edge_index.shape}")

    # data_list中包含了所有图的数据对象，每个Data对象代表一个图
    # 现在，你可以使用data_list进行后续的处理或训练

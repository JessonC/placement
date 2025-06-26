import copy

import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
# from Demos.mmapfile_demo import offset
from networkx.algorithms.bipartite.cluster import clustering
from numpy.lib.function_base import place
from setuptools.command.rotate import rotate
# from sympy import floor
from torch.fx.experimental.unification.unification_tools import first
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from reward_model import calculate_reward

# matplotlib.use("Agg")

import numpy as np
import shapely
import torch.nn as nn
from gym import spaces
import math
# from info_extract import *
from shapely.geometry import Polygon
from shapely.geometry import LineString  # 引入Shapely库的LineString类，用于判断线段是否相交
import heapq

# 每个格子为5mil
grid_space = 5


class PlacementEnv(object):
    def __init__(self, area_shape, comp_shapes, comp_names, comp_coords, pins, edges_id, comp_name_to_id, comps_type, net_to_comps, comp_pins):

        self.area_shape = area_shape  # 未网格化前的area形状
        self.area_min_x, self.area_min_y = np.min(self.area_shape, axis=0)  # 最小的x值和y值
        self.area_max_x, self.area_max_y = np.max(self.area_shape, axis=0)  # 最大的x值和y值
        self.area_grid_shape = self.gridize_box(area_shape)     # 网格化
        self.area_grid_shape_origin = np.copy(self.area_grid_shape)  # 网格化后的area形状,已经在里面的形状里的用0表示，在形状外面的用1表示
        self.area_width = self.area_grid_shape.shape[1]  # 网格化后的area宽度
        self.area_height = self.area_grid_shape.shape[0]  # 网格化后的area高度
        # print("area_height", self.area_height)
        # print("area_width", self.area_width)
        self.comp_shapes = comp_shapes  # comp形状列表
        self.comp_coords = comp_coords

        self.calculate_relative_comp_shapes()                                # 将comp_shapes变成相对值
        # print("self.comp_shapes", self.comp_shapes)
        self.comp_types = comps_type    # 每种comp的类型
        self.num_shapes = len(comp_shapes)  # 要放置的comp个数
        self.comp_names = comp_names  # comp的名字, 是一个列表
        # print(self.comp_names)

        self.pins = pins
        # self.calculate_relative_pin_coord()     # 将pin变为相对坐标
        # print("pins", self.pins)
        self.original_pins = copy.deepcopy(self.pins)  # 保存 pins 的初始状态
        self.comp_grid_shapes = [self.gridize_comp(comp_shape) for comp_shape in self.comp_shapes]
        self.edges_id = edges_id
        self.comp_name_to_id = comp_name_to_id
        self.mask = np.zeros((self.area_height, self.area_width), dtype=bool)  # 全部的网格都可用
        self.net_to_comps = net_to_comps
        self.comp_pins = comp_pins
        self.last_placement_center = (0, 0)
        self.action_set = []
        self.placement_set = []  # 存储comp的长度和宽度
        self.placement_shape = []   # 放置的comp的绝对形状
        # self.previous_placements = []  # 保存每个放置器件的中心点坐标
        self.previous_wires = []        # 保存所有走线
        self.area_score_list = []
        self.is_input_cap = []
        self.is_output_cap = []

        # 假设 self.comp_grid_shapes 是组件形状列表，self.comp_names 是组件名称列表
        # 每个 comp_grid_shape 是一个二维形状，例如 (width, height)

        # 计算每个组件的面积，并与名称和形状一起打包
        components_with_area = [
            {"index": i, "name": name, "grid_shape": grid_shape, "shape":shape, "area": grid_shape.shape[0] * grid_shape.shape[1], "type": comps_type}
            for i, (name, grid_shape, shape, comps_type) in enumerate(zip(self.comp_names, self.comp_grid_shapes, self.comp_shapes, self.comp_types))
        ]

        layout_order, comp_id = self.generate_placement_order(self.net_to_comps, self.comp_pins)
        layout_order = ['UD5', 'LD51', 'CD53', 'RD53', 'RD54', 'RD57', 'RD60', 'RF56', 'CD51',
                        'CD52', 'CD54', 'CD55', 'CD56', 'RD55', 'RD51', 'RD52', 'RD59', 'RD58']

        self.layout_order = layout_order
        self.comp_id = comp_id
        sorted_components = sorted(components_with_area,  key=lambda x: layout_order.index(x['name']))
        # sorted_components = self.generate_placement_order(self.edges_id, components_with_area)

        # print(sorted_components)
        # print("layout_order", layout_order)
        # print("comp_id", comp_id)
        # # 按面积从大到小排序
        # sorted_components = sorted(components_with_area, key=lambda x: x["area"], reverse=True)  # False为由小到大

        # 创建新的列表，分别存储排序后的 shapes 和 names以及index
        sorted_comp_grid_shapes = [comp["grid_shape"] for comp in sorted_components]
        sorted_comp_shapes = [comp["shape"] for comp in sorted_components]  # 器件的形状
        sorted_comp_names = [comp["name"] for comp in sorted_components]    # 器件的名字
        sorted_comp_indices = [comp["index"] for comp in sorted_components]  # 保存序号
        sorted_comp_types = [comp["type"] for comp in sorted_components]    # 器件的类型
        # print(sorted_comp_indices)
        # print(sorted_comp_types)
        # 将结果保存到新属性中
        self.sorted_comp_grid_shapes = sorted_comp_grid_shapes
        self.sorted_comp_shapes = sorted_comp_shapes
        self.sorted_comp_names = sorted_comp_names
        self.sorted_comp_indices = sorted_comp_indices
        self.sorted_comp_types = sorted_comp_types

        # self.placement_set = [
        #     (self.sorted_comp_grid_shapes[i].shape[0] / 2, self.sorted_comp_grid_shapes[i].shape[1] / 2)
        #     for i in range(self.num_shapes)
        # ]
        self.placement_set = []

        # 动作空间和观测空间
        # self.action_space = spaces.Discrete(self.valid_action_space)
        self.action_space = spaces.Discrete(self.area_width * self.area_height)     # 离散空间，每个格子是一个动作
        self.observation_space = spaces.Box(0, 1, shape=(self.area_height, self.area_width))    # 连续空间，每个格子的状态值
        self.num_rotations = 4
        self.temp_placement_shape_1 = []
        self.temp_placement_shape_2 = []

    # def generate_placement_order(self, pin_to_comps, comp_to_pins):
    #     # 1. 计算每个器件的引脚数量，找出引脚最多的器件
    #     comp_pin_count = {comp: len(pins) for comp, pins in comp_to_pins.items()}
    #     sorted_comps = sorted(comp_pin_count.items(), key=lambda x: -x[1])
    #     start_comp = sorted_comps[0][0]  # 引脚最多的器件
    #
    #     # 2. 开始布局
    #     placed = set()  # 已放置的器件
    #     layout_order = []  # 器件布局顺序
    #     queue = [start_comp]  # 初始化队列，从引脚最多的器件开始
    #
    #     print("comp_to_pins", comp_to_pins)
    #     print("pin_to_comps", pin_to_comps)
    #
    #     while queue:
    #         current_comp = queue.pop(0)
    #         if current_comp in placed:
    #             continue
    #
    #         # 放置当前器件
    #         placed.add(current_comp)
    #         layout_order.append(current_comp)
    #
    #         # print("sort", sorted(comp_to_pins[current_comp]))
    #         # 查找与当前器件相连的器件
    #         for pin in sorted(comp_to_pins[current_comp], reverse=True):  # 遍历当前器件的引脚
    #             for neighbor_comp in sorted(pin_to_comps[pin]):  # 通过引脚找到其他器件
    #                 if neighbor_comp not in placed:
    #                     queue.append(neighbor_comp)
    #
    #     # 3. 放置剩余未连接的器件（孤立器件）
    #     for comp in comp_to_pins.keys():
    #         if comp not in placed:
    #             layout_order.append(comp)
    #     # print("layout_order", layout_order)
    #     return layout_order

    def generate_placement_order(self, pin_to_comps, comp_to_pins):
        # 1. 计算每个器件的引脚数量，找出引脚最多的器件
        comp_pin_count = {comp: len(pins) for comp, pins in comp_to_pins.items()}
        sorted_comps = sorted(comp_pin_count.items(), key=lambda x: -x[1])
        # print("sorted_comps", sorted_comps)
        start_comp = sorted_comps[0][0]  # 引脚最多的器件

        # 2. 初始化
        placed = set()  # 已放置的器件
        layout_order = []  # 器件布局顺序
        queue = [start_comp]  # 初始化队列，从起始器件开始
        comp_id = {start_comp: 0}  # 记录每个器件的编号，start_comp 编号为 0
        next_id = 1  # 下一个可用的编号

        # 初始化输入输出电容标记列表
        is_input_cap = []
        is_output_cap = []

        while queue:
            current_comp = queue.pop(0)  # 从队列中取出当前器件
            if current_comp in placed:
                continue

            # 放置当前器件
            placed.add(current_comp)
            layout_order.append(current_comp)

            # 判断当前器件是否是输入电容
            is_cap = "CD" in current_comp  # 判断 comp_name 是否包含 "CD"
            if is_cap:
                # 检查当前器件的引脚是否包含 "5V_M"
                has_5v_m = any("5V_M" in pin for pin in comp_to_pins[current_comp])
                self.is_input_cap.append(is_cap and has_5v_m)
            else:
                self.is_input_cap.append(False)

            # 判断当前器件是否是输入电容
            is_cap = "CD" in current_comp  # 判断 comp_name 是否包含 "CD"
            if is_cap:
                # 检查当前器件的引脚是否包含 "GND"
                has_gnd = any("GND" in pin for pin in comp_to_pins[current_comp])
                self.is_output_cap.append(is_cap and has_gnd)
            else:
                self.is_output_cap.append(False)
            # 获取当前器件的所有管脚
            current_pins = comp_to_pins[current_comp]

            # 3. 对当前器件的管脚按连接器件数量进行排序（从大到小）
            pin_connections = {}
            for pin in sorted(current_pins):
                connected_comps = [neighbor for neighbor in pin_to_comps[pin] if neighbor not in placed]
                pin_connections[pin] = len(connected_comps)

            # 修改排序规则，将 GND 放到最后
            sorted_pins = sorted(
                pin_connections.items(),
                key=lambda x: (
                    x[0] != "5V_M",  # 5V_M 排在最前面
                    x[0] == "GND",  # GND 排在最后面
                    -x[1],  # 连接器件数量从大到小
                    -ord(x[0][0])  # 引脚名称的第一个字符 ASCII 码从大到小
                )
            )

            # 4. 遍历排序后的管脚，优先处理连接器件数量最多的管脚
            for pin, _ in sorted_pins:
                connected_comps = sorted(pin_to_comps[pin], reverse=True)  # 遍历管脚连接的器件
                for neighbor_comp in sorted(connected_comps):
                    if neighbor_comp not in placed and neighbor_comp not in queue:
                        queue.append(neighbor_comp)
                        # 为相连的器件分配编号
                        if neighbor_comp not in comp_id:
                            comp_id[neighbor_comp] = next_id
                next_id += 1

        # 5. 放置剩余未连接的器件（孤立器件）
        for comp in comp_to_pins.keys():
            if comp not in placed:
                layout_order.append(comp)
                # 为孤立器件分配编号
                if comp not in comp_id:
                    comp_id[comp] = next_id
                    next_id += 1

                # 判断孤立器件是否是输入电容
                is_cap = "CD" in comp  # 判断 comp_name 是否包含 "CD"
                if is_cap:
                    # 检查孤立器件的引脚是否包含 "5V_M"
                    has_5v_m = any("5V_M" in pin for pin in comp_to_pins[comp])
                    self.is_input_cap.append(is_cap and has_5v_m)
                else:
                    self.is_input_cap.append(False)
                # print("is_input_cap:", self.is_input_cap)
                # 判断孤立器件是否是输出电容
                if is_cap:
                    # 检查孤立器件的引脚是否包含 "GND"
                    has_gnd = any("GND" in pin for pin in comp_to_pins[comp])
                    self.is_output_cap.append(is_cap and has_gnd)
                else:
                    self.is_output_cap.append(False)

        # 输出 comp_id 和 is_input_cap
        # print("comp_id:", comp_id)
        # print("is_input_cap:", self.is_input_cap)
        # print("is_output_cap:", self.is_output_cap)
        # self.is_input_cap = is_input_cap
        # self.is_output_cap = is_output_cap
        # 将 comp_id 转换为列表
        comp_id = list(comp_id.values())

        return layout_order, comp_id

    def calculate_relative_comp_shapes(self):
        # 将形状数据转换为相对于质心的坐标
        for i, shape in enumerate(self.comp_shapes):
            centroid_x, centroid_y = self.comp_coords[i]  # 找到对应的质心
            self.comp_shapes[i] = [[x - centroid_x, y - centroid_y] for x, y in shape]

    def calculate_valid_grid_ids(self, shape, rotations=[0, 1, 2, 3]):
        """
        计算箱子内部的有效网格 ID，支持旋转
        :param shape: 原始形状，类型为二维 numpy 数组
        :param rotations: 支持的旋转列表 (0=0°, 1=90°, 2=180°, 3=270°)
        :return: 有效网格 ID 和对应旋转信息的列表 [(y, x, rotation), ...]
        """
        valid_ids = []

        # 遍历每种旋转状态
        for rotation in rotations:
            # 根据旋转计算新的形状
            rotated_shape = self.rotation_shape(shape, rotation)

            # 遍历可能的左上角位置
            for y in range(self.area_height - rotated_shape.shape[0] + 1):
                for x in range(self.area_width - rotated_shape.shape[1] + 1):
                    # 提取当前区域的子矩阵
                    sub_area = self.area_grid_shape[y:y + rotated_shape.shape[0], x:x + rotated_shape.shape[1]]
                    sub_mask = self.mask[y:y + rotated_shape.shape[0], x:x + rotated_shape.shape[1]]

                    # 检查子矩阵中的值是否有效
                    if np.any(sub_area != 0) or np.any(sub_mask != 0):
                        continue

                    # 如果通过所有检查，添加到有效网格中
                    valid_ids.append((y, x, rotation))

        return valid_ids

    def rotation_shape(self, shape, rotation=0):
        """
        形状旋转的函数
        """
        if rotation == 0:   # 保持原状
            rotated_shape = shape
        elif rotation == 1:     # 旋转90度
            rotated_shape = shape.T
        elif rotation == 2:     # 旋转180度
            rotated_shape = shape
        else:
            rotated_shape = shape.T

        return rotated_shape

    # 定义计算旋转后真实形状的方法
    def calculate_rotated_shape(self, center_x, center_y, shape_points, rotation, grid_space, area_min_x, area_min_y):
        """
        计算旋转后的组件真实形状
        :param center_x: 当前组件中心的全局 x 坐标
        :param center_y: 当前组件中心的全局 y 坐标
        :param shape_points: 组件的形状轮廓点（局部坐标）
        :param rotation_angle: 旋转角度（单位：度）
        :param grid_space: 网格间距
        :param area_min_x: 区域最小 x 坐标
        :param area_min_y: 区域最小 y 坐标
        :return: 旋转后组件的真实全局形状点列表
        """
        # 将角度转换为弧度, 旋转方向为逆时针旋转
        rotation_angle = rotation * 90
        # print("rotation_angle", rotation_angle)
        angle_rad = np.radians(rotation_angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # 计算全局中心位置
        global_center_x = center_x * grid_space + area_min_x
        global_center_y = center_y * grid_space + area_min_y
        global_center_y = self.area_max_y - global_center_y + self.area_min_y
        # print("global_center_x", global_center_x)
        # print("global_center_y", global_center_y)

        # 存储旋转后的形状点
        rotated_shape = []

        # 遍历每个局部形状点
        for local_x, local_y in shape_points:
            # 旋转公式（相对于中心点旋转）
            rotated_x = cos_angle * local_x - sin_angle * local_y
            rotated_y = sin_angle * local_x + cos_angle * local_y

            # 转换为全局坐标
            global_x = global_center_x * 1 + rotated_x
            global_y = global_center_y * 1 + rotated_y

            # 保存旋转后的点
            rotated_shape.append((global_x, global_y))
            # print("rotated_shape", rotated_shape)
        rotated_shape = np.array(rotated_shape)

        return rotated_shape

    def reset(self):
        """
        重置area
        """
        self.area_grid_shape = np.copy(self.area_grid_shape_origin)
        self.mask = np.zeros((self.area_height, self.area_width), dtype=bool)
        current_place = np.zeros((self.area_height, self.area_width), dtype=bool)
        # # 每个物品的当前位置和旋转角度
        # self.positions = [(0, 0, 0) for _ in range(self.num_shapes)]

        # 物品是否已成功放入箱子
        self.shape_placed = [False for _ in range(self.num_shapes)]

        # 计算每个物品的面积，后续可能用于评估
        self.shape_areas = [shape.shape[0] * shape.shape[1] for shape in self.comp_grid_shapes]

        self.valid_grid_ids = self.calculate_valid_grid_ids(self.sorted_comp_grid_shapes[0],
                                                            rotations=[0, 1, 2, 3])  # 每次重置时计算一次有效的区域
        # print("len", len(self.valid_grid_ids))
        self.previous_placements = []  # 清空之前保存的放置信息
        self.previous_wires = []
        self.placement_set = []
        self.placement_shape = []
        self.last_placement_center = (0, 0)  # 清空上一个放置中心
        self.pins = copy.deepcopy(self.original_pins)  # 恢复初始状态
        return self.area_grid_shape, self.mask, current_place

    def step(self, action, steps):
        # x = action % self.area_width
        # y = (action // self.area_width) % self.area_height
        # rotation = [0, 90, 180, 270][(action // (self.area_width * self.area_height)) % 4]

        """执行动作"""
        # 选择当前放入的形状
        # print("len", len(self.valid_grid_ids))
        # print("action", action)

        # if (len(self.valid_grid_ids) == 0) or (action >= len(self.valid_grid_ids)):
        #     current_place = self.area_grid_shape
        #     done = True
        #     game_over = False
        #     return self.area_grid_shape, self.mask, current_place, -500000, done, game_over

        if len(self.valid_grid_ids) == 0:   # 没有有效区域了
            current_place = self.area_grid_shape
            done = True
            game_over = False
            return self.area_grid_shape, self.mask, current_place, 0, done, game_over

        # action --> x, y, rotation
        if steps == 0:
            y = round(self.area_height / 2 - self.sorted_comp_grid_shapes[steps].shape[0] / 2 + 5)
            x = round(self.area_width / 2 - self.sorted_comp_grid_shapes[steps].shape[1] / 2 - 15)
            rotation = 3

            # # 1. 解 rotation
            # rotation = action // (self.area_width * self.area_height)
            #
            # # 2. 解 remaining
            # remaining = action % (self.area_width * self.area_height)
            #
            # # 3. 解 y
            # y = remaining // self.area_width
            #
            # # 4. 解 x
            # x = remaining % self.area_width
        else:
            # y, x, rotation = self.valid_grid_ids[action]  # action可能会超出维度

            rotation = action // (self.area_width * self.area_height)

            remaining = action % (self.area_width * self.area_height)

            y = remaining // self.area_width

            x = remaining % self.area_width

        # print("y", y)
        # print("x", x)
        # print("rotation", rotation)
        # print("len", len(self.valid_grid_ids))
        # print("self.shape_placed", self.shape_placed)

        # 依次放置所有器件
        for i in range(self.num_shapes):
            if self.shape_placed[i]: continue

            shape = self.sorted_comp_grid_shapes[i]
            # 将形状进行旋转
            rotated_shape = self.rotation_shape(shape, rotation)
            if (i < self.num_shapes - 1):
                next_shape = self.sorted_comp_grid_shapes[i + 1]
            else:
                next_shape = np.zeros((2, 2))
            type = self.sorted_comp_types[i]    # 这次放置器件的类型
            # 检查能否放置comp
            if self.enable_place(rotated_shape, y, x):
                # if i == 0:
                #     reward, current_place= self.calculate_reward(shape, i, y, x, self.sorted_comp_names, copy.deepcopy(self.pins))
                # else:
                #     # reward, current_place, pins = self.calculate_reward(shape, i, y, x, self.sorted_comp_names, pins)
                #     reward, current_place = self.calculate_reward(
                #         shape, i, y, x, self.sorted_comp_names
                #     )
                reward, current_place = self.calculate_reward(rotated_shape, i, y, x, self.sorted_comp_names, rotation)
                self.shape_placed[i] = True
                self.action_set.append((y, x, rotation))

                # 更新 mask
                self.update_mask(rotated_shape, next_shape, i, y, x, type)

                # self.plot_area_demo(self.area_grid_shape, i)
                if (i < self.num_shapes - 1):
                    self.valid_grid_ids = self.calculate_valid_grid_ids(next_shape, rotations=[0, 1, 2, 3])  # 重新计算有效网格
                    if len(self.valid_grid_ids) == 0:
                        # print("有效动作空间为0")
                        done = True
                        game_over = False
                        return self.area_grid_shape, self.mask, current_place, reward, done, game_over  # (next_state, reward, done)
                    else:
                        done = False
                        game_over = False
                        return self.area_grid_shape, self.mask, current_place, reward, done, game_over  # (next_state, reward, done)
                else:
                    done = True
                    game_over = True
                    return self.area_grid_shape, self.mask, current_place, reward, done, game_over  # (next_state, reward, done)
            else:
                # 如果动作不在边界内或者相互重叠
                done = True
                game_over = False
                current_place = self.area_grid_shape
                # print("动作超出边界或重叠")
                return self.area_grid_shape, self.mask, current_place, -500000, done, game_over

    def calculate_reward(self, shape, i, y, x, sorted_comp_names, rotation):
        # shape已经是旋转以后的形状

        base_reward = 200000
        min_spacing = 100  # 设置最小安全间距
        same_type_spacing = min_spacing * 0.5  # 同类器件允许更近一些
        diff_type_spacing = min_spacing * 0.8  # 不同类型器件需要更大间隔

        # 计算当前器件的中心坐标
        curr_center_y = y + shape.shape[0] / 2
        curr_center_x = x + shape.shape[1] / 2

        comp_name = sorted_comp_names[i]
        current_comp_pins = [pin for pin in self.pins if pin['comp_name'] == comp_name]
        # print(current_comp_pins)
        area_grid_before = self.area_grid_shape  # 没放前的area形状
        self.place_area(shape, y, x)  # 放置该comp
        area_grid_after = self.area_grid_shape  # 放置comp后的area形状

        current_place = area_grid_after - area_grid_before

        # # 放置位置的真实坐标
        # if rotation == 0:
        #     placement_shape = np.array([curr_center_x, curr_center_y]) *
        #     grid_space + np.array([self.area_min_x, self.area_min_y]) + self.comp_shapes[i]

        # else:
        placement_shape = self.calculate_rotated_shape(
                center_x=curr_center_x,
                center_y=curr_center_y,
                shape_points=self.sorted_comp_shapes[i],
                rotation=rotation,
                grid_space=grid_space,
                area_min_x=self.area_min_x,
                area_min_y=self.area_min_y
            )

        for pin in self.pins:
            if pin['comp_name'] == comp_name:
                # 获取原始引脚的局部坐标（相对于组件）
                local_x = pin['pin_x']
                local_y = pin['pin_y']

                # 获取组件中心点的全局坐标
                center_x = curr_center_x * grid_space + self.area_min_x
                center_y = curr_center_y * grid_space + self.area_min_y
                center_y = self.area_max_y - center_y + self.area_min_y
                if rotation == 0:
                    pin['pin_x'] = center_x + local_x
                    pin['pin_y'] = center_y + local_y
                elif rotation == 1:     # 旋转90度
                    pin['pin_x'] = center_x - local_y
                    pin['pin_y'] = center_y + local_x
                elif rotation == 2:  # 顺时针旋转 180°
                    pin['pin_x'] = center_x - local_x
                    pin['pin_y'] = center_y - local_y
                elif rotation == 3:  # 顺时针旋转 270°
                    pin['pin_x'] = center_x + local_y
                    pin['pin_y'] = center_y - local_x

        # if(i == 0):
        #     print(self.pins)
        # 保存当前器件信息
        self.previous_placements.append((curr_center_y, curr_center_x, comp_name))
        self.placement_shape.append(placement_shape)
        self.placement_set.append((shape.shape[0]/2, shape.shape[1]/2))
        # 奖励项
        alignment_bonus = 0
        pair_bonus = 0
        distance_penalty = 0
        pin_net_alignment_bonus = 0
        first_placement_bonus = 0
        wire_cross_penalty = 0

        reward = 0
        # 前面没有放完，不给奖励
        if i < len(sorted_comp_names) - 1:
            reward = 0

        # 放了最后一个再给奖励
        elif i == len(sorted_comp_names) - 1:
            # 计算总奖励
            reward = calculate_reward(self.placement_shape, self.pins)
        # # 遍历之前放置的器件
        # for j in range(i):
        #     prev_y, prev_x, prev_comp_name = self.previous_placements[j]
        #     prev_comp_pins = [pin for pin in self.pins if pin['comp_name'] == prev_comp_name]
        #
        #     # 检查引脚之间的走线
        #     for curr_pin in current_comp_pins:
        #         for prev_pin in prev_comp_pins:
        #             # 跳过相同网络的引脚
        #             if curr_pin['pin_net_name'] == prev_pin['pin_net_name']:
        #                 continue
        #
        #             # 构建当前走线并检查交叉
        #             line1 = LineString([(curr_pin['pin_x'], curr_pin['pin_y']),
        #                                 (prev_pin['pin_x'], prev_pin['pin_y'])])
        #
        #             # 交叉计数器
        #             cross_count = 0
        #             for previous_line in self.previous_wires:
        #                 if line1.intersects(previous_line):
        #                     cross_count += 1
        #
        #             # 根据交叉数量调整惩罚
        #             if cross_count > 0:
        #                 wire_cross_penalty -= 500 * cross_count  # 每次交叉增加惩罚，惩罚值与交叉次数成比例
        #             else:
        #                 wire_cross_penalty += 10000  # 如果没有交叉，给予奖励
        #
        #             # 将当前走线保存到已记录走线列表
        #             self.previous_wires.append(line1)


        return reward, current_place

    # def enable_place(self, shape, y, x):
    #     """
    #     检查能否放在(y, x)上
    #     """
    #
    #     for dy in range(shape.shape[0]):
    #         for dx in range(shape.shape[1]):
    #             if shape[dy, dx] == 1:
    #                 if x + dx >= self.area_width or y + dy >= self.area_height or self.area_grid_shape[
    #                     y + dy, x + dx] == 1 or self.mask[y, x] == 1:
    #                     if self.mask[y, x] == 1:
    #                         print("与原来形状重叠")
    #                     else:
    #                         print("超出边界")
    #                     return False
    #     return True

    def calculate_layout_area(self):
        # 计算整体布局面积
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')
        # 遍历 self.placement_shape
        for shape in self.placement_shape:
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

    def calculate_mst_manhattan(self, pins):
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

    def calculate_hpwl(self, pins):
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

    def calculate_alignment_score(self):
        """
        计算所有器件之间的对齐度得分
        :return: 对齐度得分（值越高，表示器件越对齐）
        """
        total_alignment_score = 0
        num_comparisons = 0

        # 遍历每一对器件
        for i in range(len(self.previous_placements)):
            for j in range(i + 1, len(self.previous_placements)):
                # 获取器件中心点坐标
                center_x1, center_y1, _ = self.previous_placements[i]
                center_x2, center_y2, _ = self.previous_placements[j]

                # 计算水平和垂直距离
                horizontal_distance = abs((center_x1 - center_x2) * grid_space)
                vertical_distance = abs((center_y1 - center_y2) * grid_space)

                # 对距离进行归一化（假设 grid_space 是单位间距）
                horizontal_score = max(0, horizontal_distance)
                vertical_score = max(0, vertical_distance)

                # 累加得分
                total_alignment_score += (horizontal_score + vertical_score) / 2
                num_comparisons += 1

        # 返回平均对齐度得分
        return total_alignment_score / num_comparisons if num_comparisons > 0 else 0

    def calculate_crossing_count(self, nets):
        """
        计算所有连线的交叉条数（基于最小生成树的连线段）
        :param nets: 包含所有网络和其连线点的字典
        :return: 交叉的总条数
        """
        segments = []

        # 遍历所有网络，生成每个网络的最小生成树连线段
        for net_pins in nets.values():
            if len(net_pins) > 1:
                segments.extend(self.generate_mst_segments(net_pins))

        # 统计线段交叉数
        crossing_count = 0
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                if self.is_intersecting(segments[i][0], segments[i][1], segments[j][0], segments[j][1]):
                    crossing_count += 1

        return crossing_count

    def generate_mst_segments(self, net_pins):
        """
        根据网络中的点生成最小生成树的连线段
        :param net_pins: 网络中的点 [(x1, y1), (x2, y2), ...]
        :return: 最小生成树的连线段 [(p1, p2), ...]
        """
        from scipy.spatial.distance import pdist, squareform
        from scipy.sparse.csgraph import minimum_spanning_tree

        # 计算点之间的曼哈顿距离矩阵
        # 将 net_pins 转化为 numpy 数组，确保格式为二维数组
        pin_coords = np.array(net_pins, dtype=np.float64)  # 强制转换为 float64 类型
        # print("pin_coords", pin_coords)
        # print("type(pin_coords):", type(pin_coords))
        # print("pin_coords.shape:", pin_coords.shape)
        # print("pin_coords has NaN or inf:", np.any(np.isnan(pin_coords)) or np.any(np.isinf(pin_coords)))
        if pin_coords.ndim != 2 or pin_coords.shape[1] != 2:
            raise ValueError("net_pins must be a list of 2D points, e.g., [(x1, y1), (x2, y2), ...]")
        dist_matrix = squareform(pdist(pin_coords, metric='cityblock'))

        # 生成最小生成树
        mst = minimum_spanning_tree(dist_matrix).toarray().astype(float)

        # 提取 MST 的连线段
        segments = []
        for i in range(len(net_pins)):
            for j in range(i + 1, len(net_pins)):
                if mst[i, j] > 0 or mst[j, i] > 0:
                    segments.append((net_pins[i], net_pins[j]))

        return segments

    def is_intersecting(self, p1, p2, q1, q2):
        """
        判断两条线段是否相交
        :param p1: 第一条线段的起点
        :param p2: 第一条线段的终点
        :param q1: 第二条线段的起点
        :param q2: 第二条线段的终点
        :return: True 表示相交，False 表示不相交
        """

        def cross_product(a, b):
            return a[0] * b[1] - a[1] * b[0]

        def vector(a, b):
            return (b[0] - a[0], b[1] - a[1])

        # 向量叉积
        v1 = vector(p1, q1)
        v2 = vector(p1, q2)
        v3 = vector(q1, p1)
        v4 = vector(q1, p2)

        # 线段是否相交
        return (cross_product(vector(p1, p2), v1) * cross_product(vector(p1, p2), v2) < 0 and
                cross_product(vector(q1, q2), v3) * cross_product(vector(q1, q2), v4) < 0)

    def enable_place(self, shape, y, x):
        """
        检查能否放置形状在 (y, x) 位置上，尽量避免使用 for 循环
        """
        # 获取形状的高度和宽度
        shape_height, shape_width = shape.shape

        # 预先检查是否越界
        if x + shape_width > self.area_width or y + shape_height > self.area_height:
            # print("超出边界")
            return False

        # 提取要放置区域的子矩阵
        area_submatrix = self.area_grid_shape[y:y + shape_height, x:x + shape_width]
        mask_submatrix = self.mask[y:y + shape_height, x:x + shape_width]

        # 使用布尔数组来检查重叠或是否超出边界
        # 检查 area_grid_shape 或 mask 中对应位置是否为 1
        shape_mask = shape == 1  # 只考虑形状中为 1 的部分
        if np.any((area_submatrix == 1) & shape_mask) or np.any((mask_submatrix == 1) & shape_mask):
            # print("shape", shape.shape)
            # print("与原来形状重叠")
            return False

        return True

    def place_area(self, shape, y, x):
        """
        将comp放入到area的指定位置后，相应的位置填1
        """

        # for dx in range(shape.shape[1]):
        #     for dy in range(shape.shape[0]):
        #         if shape[dy, dx] == 1:      # 有效区域为0
        #             self.area_grid_shape[y + dy, x + dx] = 1    # 填了以后，将该区域置为1


        # 提取目标区域的切片
        target_area = self.area_grid_shape[y:y + shape.shape[0], x:x + shape.shape[1]]

        # 将目标区域中对应 shape 的位置设置为 1
        target_area[shape == 1] = 1

    def update_mask(self, shape, next_shape, i, y, x, type):
        """
        更新 mask，屏蔽被占用的网格、左边外接网格，以及区域外的网格
        """

        # ======== 屏蔽当前组件占用的网格 ===========
        shape_height, shape_width = shape.shape
        # mask掉的部分要比实际的器件更大，这样可以为器件之间留出一些距离，根据器件类型决定向外扩多少（offset）
        offset = 0
        keywords_1 = ['IC', '电感', 'Inductor', '电解']
        keywords_2 = ['PIP', '插座', '端子']
        if any(keyword in type for keyword in keywords_1):
            offset = math.ceil((0.5 * 39.3701) / grid_space)
        elif any(keyword in type for keyword in keywords_2):
            offset = math.ceil((1 * 39.3701) / grid_space)
        else:
            offset = 2
        y_end = min(y + shape_height + offset, self.mask.shape[0])
        x_end = min(x + shape_width + offset, self.mask.shape[1])
        y_start = max(0, y - offset)
        x_start = max(0, x - offset)
        self.mask[y_start:y_end, x_start:x_end] = 1

        # offset = 0
        # # ===== 根据下一个器件的形状进行mask ======
        # next_shape_height, next_shape_width = next_shape.shape
        #
        # # 上方区域
        # if y - next_shape_height - offset >= 0:
        #     self.mask[y - next_shape_height - offset:y, x:x + shape_width] = 1
        #
        # else:
        #     self.mask[0:y, x:x + shape_width] = 1
        #
        # # 左侧区域
        # if x - next_shape_width - offset >= 0:
        #     self.mask[y:y + shape_height, x - next_shape_width - offset:x] = 1
        #
        # else:
        #     self.mask[y:y + shape_height, 0:x] = 1
        #
        # # 左上方区域
        # if y - next_shape_height - offset >= 0 and x - next_shape_width - offset >= 0:
        #     self.mask[y - next_shape_height - offset:y, x - next_shape_width - offset:x] = 1

        # self.visualize_mask(i)

    def visualize_mask(self, id):
        """
        可视化当前的 mask。
        """
        plt.figure(figsize=(20, 20))
        plt.imshow(self.mask, cmap="gray", origin="upper")  # 用灰度图显示mask
        plt.title("Current Mask State")
        plt.colorbar(label="Mask Value")  # 添加颜色条
        plt.xlabel("Width (x)")
        plt.ylabel("Height (y)")
        plt.xticks(np.arange(0, self.area_width, 1))  # 设置x轴网格线
        plt.yticks(np.arange(0, self.area_height, 1))  # 设置y轴网格线
        plt.grid(which="both", color="black", linestyle="-", linewidth=0.5)  # 添加网格
        plt.show()
        plt.savefig(f'./mask/mask_{id}.png')
        plt.close()

    def gridize_box(self, box_shape):
        """将area的形状网格化"""
        # 将由顶点坐标列表表示的多边形，转换为shapely Polygon对象
        polygon = Polygon(box_shape)
        # 得到多边形的坐标边界，这四个坐标得到了一个能包括多边形的矩形
        min_x, min_y, max_x, max_y = polygon.bounds

        # 将坐标值映射到网格索引
        min_x = int(np.floor(min_x / grid_space))
        min_y = int(np.floor(min_y / grid_space))
        max_x = int(np.ceil(max_x / grid_space))
        max_y = int(np.ceil(max_y / grid_space))

        # 得到网格化后的状态矩阵
        # 把有效区域设为0，无效区域设为1
        grid_box = np.ones((int(max_y - min_y + 1), int(max_x - min_x + 1)), dtype=np.uint8)
        # print("grid_box", grid_box.shape)
        for x in range(grid_box.shape[1]):
            for y in range(grid_box.shape[0]):
                # point = shapely.geometry.Point(x + min_x, y + min_y)
                point_x = x * grid_space + min_x * grid_space
                point_y = y * grid_space + min_y * grid_space
                point = shapely.geometry.Point(point_x, point_y)    # 格子左上角的真实坐标
                # if polygon.contains(point) or polygon.boundary.contains(point):
                # 用多边形对象来判定点是否在 多边形内部 或 边界上
                if polygon.contains(point) or polygon.boundary.contains(point):
                    grid_box[grid_box.shape[0] - y, x] = 0
        return grid_box

    def gridize_comp(self, box_shape):
        """将comp的形状网格化"""
        # 转换箱子形状为shapely Polygon对象
        polygon = Polygon(box_shape)

        # 生成网格化的箱子形状
        min_x, min_y, max_x, max_y = polygon.bounds
        min_x = int(np.floor(min_x / grid_space))
        min_y = int(np.floor(min_y / grid_space))
        max_x = int(np.ceil(max_x / grid_space))
        max_y = int(np.ceil(max_y / grid_space))
        grid_box = np.ones((int(max_x - min_x + 1), int(max_y - min_y + 1)), dtype=np.uint8)
        # 对于comp，将其外包络里面的格子全设为0
        for x in range(grid_box.shape[0]):
            for y in range(grid_box.shape[1]):
                # point = shapely.geometry.Point(x + min_x, y + min_y)
                point_x = x * grid_space + min_x * grid_space
                point_y = y * grid_space + min_y * grid_space
                point = shapely.geometry.Point(point_x, point_y)
                if polygon.contains(point) or polygon.boundary.contains(point):
                    grid_box[x, y] = 1
        return grid_box.T

    def plot_area_demo(self, area_grid, id):
        """可视化网格化的箱子形状"""
        plt.figure(figsize=(40, 40))

        plt.imshow(area_grid, origin='upper', cmap='Blues', extent=(0, area_grid.shape[1], 0, area_grid.shape[0]))
        # plt.matshow(area_grid, cmap='Blues', fignum=1)  # 使用灰度图
        # plt.colorbar(label="Value")  # 可选，显示颜色条
        # plt.colorbar(label="Occupied")
        # 设置网格
        plt.grid(visible=True, color='black', linestyle='--', linewidth=0.5)
        plt.gca().invert_yaxis()  # 翻转 y 轴，使坐标轴向下递增
        # plt.xticks(np.arange(0, area_grid.shape[0] + 1, 1), fontsize=8)  # 横坐标刻度
        # plt.yticks(np.arange(0, area_grid.shape[1] + 1, 1), fontsize=8)  # 纵坐标刻度
        plt.title("Gridized Box Shape Visualization", fontsize=14)
        # plt.gca().set_facecolor('white')
        plt.xlabel("X Coordinate (mil)", fontsize=14)
        plt.ylabel("Y Coordinate (mil)", fontsize=14)
        plt.show()
        # plt.savefig('comp_shape_{}.png')
        plt.savefig(f'./area_shape/area_shape_{id}.png')
        plt.close()  # 关闭当前图像，防止叠加

    def plot_comp_demo(self, area_grid, id):
        """可视化网格化的箱子形状"""
        plt.figure(figsize=(20, 20))
        # plt.imshow(area_grid.T, origin='lower', cmap='Greys', extent=(0, area_grid.shape[1], 0, area_grid.shape[0]))
        plt.matshow(area_grid, cmap='Blues', fignum=1)  # 使用灰度图
        plt.colorbar(label="Value")  # 可选，显示颜色条
        # plt.colorbar(label="Occupied")
        # 设置网格
        plt.grid(visible=True, color='black', linestyle='--', linewidth=0.5)
        # plt.xticks(np.arange(0, area_grid.shape[0] + 1, 1), fontsize=8)  # 横坐标刻度
        # plt.yticks(np.arange(0, area_grid.shape[1] + 1, 1), fontsize=8)  # 纵坐标刻度
        plt.title("Gridized Box Shape Visualization", fontsize=14)
        # plt.gca().set_facecolor('white')
        plt.xlabel("X Coordinate (mil)", fontsize=14)
        plt.ylabel("Y Coordinate (mil)", fontsize=14)
        plt.show()
        # plt.savefig('comp_shape_{}.png')
        plt.savefig(f'./comp_shape/comp_shape_{id}.png')
        plt.close()  # 关闭当前图像，防止叠加

    # def grid_area(self, area_shape):
    #     """
    #        将整个area区域进行网格化
    #        网格化的单位为0.1mil
    #        """
    #     # 转换为numpy数组
    #     area_coords = np.array(area_shape)
    #
    #     # 获取最小的x和y值，作为原点
    #     min_x = np.min(area_coords[:, 0])
    #     min_y = np.min(area_coords[:, 1])
    #     max_x = np.max(area_coords[:, 0])
    #     max_y = np.max(area_coords[:, 1])
    #
    #     # 定义网格间距
    #     grid_spacing = grid_space  # 设置网格化的单位为1mil
    #
    #     # 将坐标平移，使原点为(0, 0)
    #     shifted_coords = area_coords - [min_x, min_y]
    #     # 将坐标转换为网格索引，每个网格索引对应于0.1 mil
    #     grid_indices = shifted_coords / grid_spacing
    #
    #     # 初始化结果
    #     integer_grid_coords = []
    #     # 根据点的位置，动态调整取整策略
    #     for x, y in grid_indices:
    #         if x == min_x / grid_spacing:
    #             # 左边界向下取整
    #             grid_x = np.floor(x)
    #         else:
    #             # 其他点默认向上取整
    #             grid_x = np.ceil(x)
    #
    #         if y == min_y / grid_spacing:
    #             # 下边界向下取整
    #             grid_y = np.floor(y)
    #         else:
    #             # 其他点默认向上取整
    #             grid_y = np.ceil(y)
    #
    #         integer_grid_coords.append([grid_x, grid_y])
    #     # 根据需要对网格索引进行取整（就近取整、向下取整或向上取整）
    #     # 这里选择就近取整
    #     integer_grid_coords = np.array(grid_indices).astype(int)
    #     return integer_grid_coords




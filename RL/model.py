
# ---------------------------------------
# 前面的特征嵌入层
# ---------------------------------------

import json
import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import os
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from shapely.geometry import Polygon
from collections import defaultdict
from typing import List, Dict, Tuple
import networkx as nx


class CircuitDataset(Dataset):
    def __init__(self, json_dir: str, compute_reward_func, transform=None):
        """
        Args:
            json_dir (str): 存储JSON文件的目录路径。
            compute_reward_func (callable): 计算reward的函数，接受data字典并返回float。
            transform: 可选的转换函数。
        """
        self.json_dir = json_dir
        self.compute_reward = compute_reward_func
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.transform = transform

    def load_json(self, filepath: str) -> Dict:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def compute_perimeter_area(self, shape: List[List[float]]) -> Tuple[float, float]:
        """
        计算多边形的周长和面积。

        Args:
            shape (List[List[float]]): 多边形顶点列表。

        Returns:
            perimeter (float), area (float)
        """
        polygon = Polygon(shape)
        perimeter = polygon.length
        area = polygon.area
        return perimeter, area

    # 计算圆度
    def calculate_roundness(self, area, perimeter):
        if perimeter == 0:  # 防止除以零
            return 0
        roundness = (4 * np.pi * area) / (perimeter ** 2)
        return roundness


    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        filepath = os.path.join(self.json_dir, self.json_files[idx])
        data = self.load_json(filepath)

        # 提取所有pin信息
        if len(data.get('comps_info', {})) == 0:
            comps_info = data
        else:
            comps_info = data.get('comps_info', {})
        pins = []
        for comp_id, comp_details in comps_info.items():
            comp_name = comp_details.get('comp_name', comp_id)
            pins_info = comp_details.get('comp_pin', {})
            for pin_id, pin in pins_info.items():
                pin_net_name = pin.get('pin_net_name')
                pin_x = pin.get('pin_x')
                pin_y = pin.get('pin_y')
                if pin_net_name is not None and pin_x is not None and pin_y is not None:
                    pins.append({
                        'comp_name': comp_name,
                        'pin_net_name': pin_net_name,
                        'pin_x': pin_x,
                        'pin_y': pin_y,
                        'comp_shape': comp_details.get('comp_shape', [])
                    })

        num_nodes = len(pins)
        if num_nodes == 0:
            # 处理没有节点的情况，可以跳过或返回空图
            raise ValueError(f"No pins found in file: {filepath}")

        # 计算节点特征（周长和面积）todo 增加特征
        node_features = []
        for pin in pins:
            perimeter, area = self.compute_perimeter_area(pin['comp_shape'])
            shape_features = compute_shape_features(pin)  # 长、宽
            comp_shape_cur = pin['comp_shape']
            x_coords = [coord[0] for coord in comp_shape_cur]
            y_coords = [coord[1] for coord in comp_shape_cur]
            max_x = max(x_coords)
            min_x = min(x_coords)
            max_y = max(y_coords)
            min_y = min(y_coords)
            comp_width = max_x - min_x
            comp_height = max_y - min_y
            outline = [comp_width, comp_height]
            ratio = comp_width / comp_height
            convexity = self.calculate_roundness(area, perimeter)
            node_features.append([perimeter, area] + shape_features + outline + [ratio] + [convexity])

        x = torch.tensor(node_features, dtype=torch.float)  # [num_nodes, 2]

        # 构建边列表，根据net_name连接同一net的pins
        net_to_pins = defaultdict(list)
        for idx_pin, pin in enumerate(pins):
            net = pin['pin_net_name']
            net_to_pins[net].append(idx_pin)

        edge_index = []
        for net, pin_indices in net_to_pins.items():
            if len(pin_indices) > 1:
                for i in range(len(pin_indices)):
                    for j in range(i + 1, len(pin_indices)):
                        edge_index.append([pin_indices[i], pin_indices[j]])
                        edge_index.append([pin_indices[j], pin_indices[i]])  # 双向边

        if len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]

        # 计算reward
        reward = self.compute_reward(data)  # 返回一个float
        y = torch.tensor([reward], dtype=torch.float)  # [1]

        # 创建 PyG 的 Data 对象
        data_pyg = Data(x=x, edge_index=edge_index, y=y)

        if self.transform:
            data_pyg = self.transform(data_pyg)

        return data_pyg


# 定义 CircuitTrainingModel 类
class CircuitTrainingModel(nn.Module):
    """基于PyTorch Geometric的GCN模型，用于预测reward。"""

    def __init__(
            self,
            num_node_features: int = 6,  # 特征数量
            hidden_dim: int = 64,
            graph_embedding_dim: int = 128,
            num_gcn_layers: int = 3,
    ):
        super(CircuitTrainingModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_dim))
        for _ in range(num_gcn_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.fc1 = nn.Linear(hidden_dim, graph_embedding_dim)
        # self.fc2 = nn.Linear(graph_embedding_dim, 1)  # 输出一个标量reward

    def forward(self, data):
        """
        前向传播。
        Args:
            data (Data): 图数据，包括 x, edge_index, batch。
        Returns:
            torch.Tensor: 预测的reward，形状 [batch_size, 1]。
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # 全局池化（平均池化）
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]

        # 全连接层
        x = F.relu(self.fc1(x))     # [batch_size, 128]
        # reward = self.fc2(x)  # [batch_size, 1]

        return x


class PolicyNet(nn.Module):
    def __init__(self, input_dim=128, output_shape=(131, 217, 4)):
        super(PolicyNet, self).__init__()
        self.output_shape = output_shape

        # 全连接层，将输入从 128 维扩展到较小的中间维度
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),  # 扩展到 1024 维
            nn.ReLU(),
            nn.Linear(1024, 4096),       # 扩展到 4096 维
            nn.ReLU(),
        )

        # 转置卷积层，逐步调整形状
        self.conv_transpose = nn.Sequential(
            # 第一层转置卷积：从 [8, 64, 8, 8] -> [8, 32, 16, 16]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 第二层转置卷积：从 [8, 32, 16, 16] -> [8, 16, 32, 32]
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 第三层转置卷积：从 [8, 16, 32, 32] -> [8, 8, 64, 64]
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 第四层转置卷积：从 [8, 8, 64, 64] -> [8, 4, 131, 217]
            nn.ConvTranspose2d(8, 4, kernel_size=(4, 30), stride=(2, 3), padding=(0, 2), output_padding=(1, 2)),
        )

        # 初始化权重
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, mask=None):
        # 输入 x 的形状: (batch_size, 128)
        x = self.fc(x)  # 全连接层扩展维度
        x = x.view(x.size(0), 64, 8, 8)  # 调整为适合转置卷积的输入形状
        x = self.conv_transpose(x)  # 转置卷积调整形状
        x = torch.flatten(x, start_dim=1)
        if mask is not None:
            x = x * mask     # 屏蔽无效动作
        # 直接构建离散概率分布，解决softmax的数值稳定性问题
        dist = torch.distributions.Categorical(logits=x)
        # x = F.softmax(x, dim=-1)  # [b, n_actions]  计算每个动作的概率
        return dist


class ValueNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ValueNet, self).__init__()
        # self.conv1 = nn.Conv2d(input_channels, 4, kernel_size=3, stride=1, padding=1)  # 输出: 32xHxW
        # conv_out_size = self._get_conv_output_size(input_shape)
        # self.fc1 = nn.Linear(conv_out_size, 32)
        # self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1)  # 输出: 64xHxW
        self.fc1 = nn.Linear(128, 128)  # 假设最终的特征图尺寸为 4x4
        self.fc2 = nn.Linear(128, 1)

        # 初始化权重
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x = torch.relu(self.conv1(x))
        # x = torch.flatten(x, start_dim=1)  # 展平成一维
        # x = torch.relu(self.fc1(x))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

    def _get_conv_output_size(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape)
        conv_output = self.conv(dummy_input)
        return int(np.prod(conv_output.size()))


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        # 嵌入特征提取网络
        self.embedding = CircuitTrainingModel()
        # 策略网络
        self.actor = PolicyNet(input_dim, output_dim)
        # 价值网络
        self.critic = ValueNet(input_dim, output_dim)

    def forward(self, x, mask=None):
        # 前向传播： 输入x， 分别获得动作概率分布和状态价值估计
        graph_features = self.embedding(x)
        action_dist = self.actor(graph_features, mask=mask)     # 返回的离散概率分布
        state_value = self.critic(graph_features)
        return action_dist, state_value


def compute_shape_features(pin: Dict) -> List[float]:
    """
    计算与pin所属组件相关的形状特征：宽度、高度、边数。

    Args:
        pin (Dict): 包含 'comp_shape' 的引脚信息字典。'comp_shape' 是一个包含 [x, y] 坐标的列表。

    Returns:
        List[float]: 包含宽度、高度和边数的列表。
    """
    comp_shape = pin.get('comp_shape', [])

    if not comp_shape or len(comp_shape) < 3:
        # 如果形状信息不足，返回默认值或处理方式
        return [0.0, 0.0, 0]

    # 提取x和y坐标
    x_coords = [point[0] for point in comp_shape]
    y_coords = [point[1] for point in comp_shape]

    # 计算宽度和高度（使用边界框）
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)

    # 计算边数
    # num_edges = len(comp_shape)

    return [width, height]#, num_edges]
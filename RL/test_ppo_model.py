import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from PPO2 import *
from Placement_Env import *
from demo import *

# 测试设置
MODEL_PATH = 'ppo_model_7.pth'  # 保存的模型路径
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
id = 1


def load_model(agent, model_path):
    """
    加载模型和优化器状态
    """
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    agent.embedding.load_state_dict(checkpoint['embedding_state_dict'])
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    print("Model loaded successfully")


def plot_test_demo(out_coord, pins, id, env):
    """
    绘制测试集的demo，注意out_coord已经是按照真实顺序排列的了
    """
    file_path = '../data_test'
    # filename = file_path + './' + 'data{id}.json'
    filename = f"{file_path}/data{id}.json"
    folder_path_test = './demo'
    module_data = load_module_data(filename)
    area = module_data.get('area', [])
    area = np.array(area)
    coord = np.array(out_coord)[:, :2]    # 取出(x, y)

    rotations = np.array(out_coord)[:, 2]   # 取出rotation

    comps_info = module_data.get('comps_info', {})
    all_comp_shapes = []
    for comp_name, comp in comps_info.items():
        comp_shape = np.array(comp.get('comp_shape', []))
        all_comp_shapes.append(comp_shape)  # 这个是提取出测试集的形状轮廓，是一个相对值
    # print("all_comp_shapes", all_comp_shapes)

    # rotations = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.])
    rotated_shapes = []
    for shape, rotation in zip(all_comp_shapes, rotations):
        # 将旋转角度从单位90°转换为弧度
        angle_rad = np.radians(rotation * 90)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # 对每个点应用旋转公式
        rotated_shape = []
        for x, y in shape:
            rotated_x = cos_angle * x - sin_angle * y
            rotated_y = sin_angle * x + cos_angle * y
            rotated_shape.append([rotated_x, rotated_y])

        # 将旋转后的结果添加到列表中
        rotated_shapes.append(np.array(rotated_shape))

    all_comp_shapes_real = []
    for i in range(len(out_coord)):
        # print("out_coord_shape", out_coord[0].shape)
        # print("env.placement_shape", env.placement_shape[0].shape)
        comp_shape_real = coord[i] + rotated_shapes[i]
        all_comp_shapes_real.append(comp_shape_real)
    print("all_comp_shapes_real", all_comp_shapes_real)
    # print("comp形状的grid值为:", )
    plot_demo_ppo(area, all_comp_shapes_real, pins, id, folder_path_test, env)


def modify_real_coord_order(real_coord_order):
    """
    人手工调整器件的位置
    """
    real_coord_order[1, 1] += 10
    real_coord_order[2, 1] += 10
    real_coord_order[4, 1] += 10
    real_coord_order[5, 1] += 10
    real_coord_order[6, 1] += 10
    real_coord_order[13, 1] += 10
    real_coord_order[15, 1] += 10
    real_coord_order[16, 1] += 10
    real_coord_order[17, 1] += 10
    real_coord_order[2, 2] = 1
    real_coord_order[17, 2] = 1
    real_coord_order[16, 2] = 1
    real_coord_order[5, 2] = 1
    real_coord_order[4, 2] = 1


    return real_coord_order


def run(agent, env, pins):
    """
    使用训练好的模型测试环境并保存布局过程为动画
    """
    reward = 0
    grid, mask, current_place = env.reset()
    state = agent.generate_state([], comp_coords, env.placement_shape, env.pins, reward)
    done = False
    game_over = False
    steps = 0

    frames = []  # 存储每一步的环境状态
    while not (game_over or done):
        # 选择动作
        action, select_action, logp = agent.take_action(env, state, steps, deterministic=True)
        grid, mask, current_place, reward, done, game_over = env.step(select_action, steps)

        real_coord = [((x1 + x2) * grid_space + env.area_min_x, env.area_max_y - (y1 + y2) * grid_space, rotation * 90)
                      for (y1, x1, rotation), (y2, x2) in zip(env.action_set, env.placement_set)]
        next_state = agent.generate_state(env.sorted_comp_names[:steps + 1], real_coord, env.placement_shape, env.pins,
                                          reward)
        state = next_state
        steps += 1
        # print("pins", env.pins)
    # print(env.action_set)
    # print(env.placement_set)

    real_coord = [((x1 + x2) * grid_space + env.area_min_x, env.area_max_y - (y1 + y2) * grid_space, rotation) for
                  (y1, x1, rotation), (y2, x2) in zip(env.action_set, env.placement_set)]
    print("real_coord", real_coord)

    # 生成值到索引的映射
    value_to_index = {value: idx for idx, value in enumerate(env.sorted_comp_indices)}

    # 按照 value_to_index 对应的顺序重新排序 real_coord
    real_coord_order = [real_coord[value_to_index[i]] for i in range(len(real_coord))]
    print(env.sorted_comp_indices)
    real_coord_order = np.array(real_coord_order)

    # real_coord_order = modify_real_coord_order(real_coord_order)
    print("real_coord_order", real_coord_order)
    # 生成值到索引的映射
    value_to_index = {value: idx for idx, value in enumerate(env.sorted_comp_indices)}
    # 真实的相对位置
    env.placement_shape = [env.placement_shape[value_to_index[i]] for i in range(len(real_coord))]
    print(env.sorted_comp_indices)

    # 计算引脚绝对坐标
    absolute_pins = []

    # 假设器件顺序与质心列表一一对应
    comp_names = [pin['comp_name'] for pin in pins]  # 提取器件名称列表
    unique_comps = list(dict.fromkeys(comp_names))  # 去重并保持顺序

    for pin in pins:
        # 找到器件名称在 unique_comps 中的索引
        comp_index = unique_comps.index(pin['comp_name'])
        print("comp_index", comp_index)
        centroid_x, centroid_y, rotation = real_coord_order[comp_index]  # 获取对应的质心坐标

        relative_x = pin['pin_x']
        relative_y = pin['pin_y']
        # 根据 rotation 计算旋转后的坐标
        if rotation == 0:
            rot_x, rot_y = relative_x, relative_y  # 无需旋转
        elif rotation == 1:
            rot_x = -relative_y
            rot_y = relative_x
        elif rotation == 2:
            rot_x = -relative_x
            rot_y = -relative_y
        elif rotation == 3:
            rot_x = relative_y
            rot_y = -relative_x
        else:
            raise ValueError("Invalid rotation value, must be 0, 1, 2, or 3.")
        # 计算绝对坐标
        abs_x = centroid_x + rot_x
        abs_y = centroid_y + rot_y

        # 保存结果
        absolute_pins.append({
            'comp_name': pin['comp_name'],
            'pin_net_name': pin['pin_net_name'],
            'pin_x': abs_x,
            'pin_y': abs_y
        })
    print(absolute_pins)
    plot_test_demo(real_coord_order, absolute_pins, id, env)


# def get_data():
#     file_path = '../data_test'
#     filename = f"{file_path}/data{id}.json"
#
#     module_data = load_module_data(filename)
#     area = module_data.get('area', [])
#     area_shape = np.array(area)
#     comps_name = []
#     comps_shape = []
#     pins = []
#     comps_info = module_data.get('comps_info', {})
#
#     for comp_name, comp in comps_info.items():
#         comp_data = {
#             'comp_name': comp_name,
#             'comp_rotation': comp.get('comp_rotation', 0.0),
#             'comp_shape': comp.get('comp_shape', []),
#             'comp_x': comp.get('comp_x', 0.0),
#             'comp_y': comp.get('comp_y', 0.0),
#         }
#         comps_name.append(comp_data['comp_name'])
#         comps_shape.append(comp_data['comp_shape'])
#
#         # 获取pin信息确定不同器件之间的连接关系
#         for pin_number, pin in comp.get('comp_pin', {}).items():
#             pin_x = pin.get('pin_x')
#             pin_y = pin.get('pin_y')
#             # 得把pin_x和pin_y换成相对于comp_coord的坐标，因为测试集是这样的形式给进去的
#             pin_rotation = pin.get('pin_rotation')
#             pin_x_grid = round((pin_x) / 0.1)
#             pin_y_grid = round((pin_y) / 0.1)
#             pin_data = {
#                 'comp_name': comp_name,
#                 'pin_net_name': pin.get('pin_net_name'),
#                 'pin_x': pin_x_grid,
#                 'pin_y': pin_y_grid,  # 这两个也要换成相对坐标的形式
#                 'pin_rotation': pin_rotation,
#             }
#             pins.append(pin_data)
#
#     return area_shape, comps_name, comps_shape, pins

def build_graph(comps_name, pins):
    # Build net to components mapping
    net_to_comps = defaultdict(set)

    for pin in pins:
        net_name = pin['pin_net_name']
        comp_name = pin['comp_name']
        if net_name:
            net_to_comps[net_name].add(comp_name)

    # 输出字典：每个器件对应的管脚集合
    comp_pins = defaultdict(set)

    # 遍历每个网络和其对应的器件
    for net, comps in net_to_comps.items():
        for comp in comps:
            comp_pins[comp].add(net)  # 将网名添加到对应的器件中

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

    comp_name_to_id = {comps_name[idx]: idx for idx in range(len(comps_name))}       # {'CD52': 0, 'LD51': 1, 'RF56': 2, 'RD59': 3}
    id_to_comp_name = {idx: comps_name[idx] for idx in range(len(comps_name))}      # {0: 'CD52', 1: 'LD51', 2: 'RF56', 3: 'RD59', 4: 'RD58'}
    edges_id = [(comp_name_to_id[src], comp_name_to_id[dst]) for src, dst in edges]
    print(net_to_comps)
    return edges_id, comp_name_to_id, net_to_comps, comp_pins

if __name__ == "__main__":
    # area_shape, comps_shape, comps_name, pins, comps_type = get_data()
    area_shape, comps_shape, comps_name, comp_coords, pins, comps_type = get_data()
    edges_id, comp_name_to_id, net_to_comps, comp_pins = build_graph(comps_name, pins)
    env = PlacementEnv(area_shape=area_shape, comp_shapes=comps_shape, comp_names=comps_name,
                       comp_coords=comp_coords, edges_id=edges_id, comp_name_to_id=comp_name_to_id,
                       comps_type=comps_type,
                       net_to_comps=net_to_comps, comp_pins=comp_pins, pins=pins)
    n_states = env.observation_space.shape[0] * env.observation_space.shape[1]
    n_actions = env.action_space.n * 4

    agent = PPO(n_states=n_states,  # 状态数
                n_hiddens=n_hiddens,  # 隐含层神经元数（示例值）
                n_actions=n_actions,  # 动作数
                actor_lr=1e-4,  # 策略网络学习率（示例值）
                critic_lr=1e-4,  # 价值网络学习率（示例值）
                lmbda=0.95,  # 优势函数的缩放因子
                epochs=10,  # 一组序列训练的轮次
                eps=0.2,  # PPO中截断范围的参数
                gamma=0.99,  # 折扣因子
                device=DEVICE
                )

    # 加载模型
    load_model(agent, MODEL_PATH)

    # 测试并保存动画
    run(agent, env, pins)

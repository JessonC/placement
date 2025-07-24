# 代码用于离散环境的模型
import copy

import torch
# from torch.nn.functional import embedding

from Placement_Env import *
from model import *
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')

# ----------------------------------------- #
# 参数设置
# ----------------------------------------- #

# ----------------------------------------- #
# 参数设置
# ----------------------------------------- #

num_episodes = 500  # 总迭代次数
gamma = 0.95  # 折扣因子
# embedding_lr = 0.001    # 嵌入网络的学习率
# actor_lr = 0.0008  # 策略网络的学习率
# critic_lr = 0.002  # 价值网络的学习率
lr = 3e-4
n_hiddens = 64  # 隐含层神经元个数
return_list = []  # 保存每个回合的return
num_list = []     # 保存每个回合放置的数量
batch_size = 16
save_path = 'ppo_model_7.pth'
# 定义需要提取的关键词
keywords = ['IC', 'SMD', 'TP', 'MC', 'NPO', 'RES', 'DCDC', '电感', 'Inductor', 'DIO']


# ----------------------------------- #
# 构建模型
# ----------------------------------- #

class PPO:
    def __init__(self, n_states, n_hiddens, n_actions,
                 lr, lmbda, epochs, eps, gamma, device):
        # 实例化策略网络
        input_dim = 128
        # self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        output_dim = n_actions
        self.output_dim = output_dim

        # 策略-价值网络
        self.network = ActorCritic(input_dim, output_dim).to(device)
        # 网络优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.gamma = gamma  # 折扣因子
        self.gae_lambda = lmbda  # GAE优势函数的缩放系数
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.clip_eps = eps  # PPO中截断范围的参数
        self.entropy_coeff = 0.01
        self.device = device


    def take_action(self, env, state, steps, deterministic=False):
        """
        根据当前状态选择一个动作，支持旋转
        :param env: 环境对象，包含有效动作列表 `env.valid_grid_ids`，格式为 [(y, x, rotation), ...]
        :param state: 当前状态
        :param deterministic: 是否为确定性选择
        :return: 动作索引和对应的有效动作
        """
        # 初始化 valid_mask 并计算动作概率
        valid_mask = np.zeros(self.output_dim, dtype=bool)  # self.output_dim 是扩展后的动作空间维度

        # 遍历有效动作列表，将有效动作的 valid_mask 置为 1
        for valid_idx, valid_id in enumerate(env.valid_grid_ids):  # valid_id = (y, x, rotation)
            # 计算 (y, x, rotation) 在扩展动作空间中的索引
            index = (valid_id[2] * env.area_width * env.area_height) + (valid_id[0] * env.area_width) + valid_id[1]
            valid_mask[index] = 1

        # 将 valid_mask 转换为张量
        valid_mask = torch.tensor(valid_mask, dtype=torch.float32, device=self.device)      # 4 * width * height

        # 确定性策略，用于模型测试
        if deterministic:
            # 第一步时选择固定的动作
            if steps == 0:
                # 计算距离
                y = round(env.area_height / 2 - env.sorted_comp_grid_shapes[steps].shape[0] / 2 + 5)
                x = round(env.area_width / 2 - env.sorted_comp_grid_shapes[steps].shape[1] / 2 - 15)
                rotation = 3
                action = rotation * env.area_width * env.area_height + y * env.area_width + x
                dist, value = self.network(state, mask=valid_mask)
                action_tensor = torch.tensor(action, dtype=torch.long, device=device)
                logp = dist.log_prob(action_tensor)  # 计算固定动作的log_prob，第一个维度是batch_size

            else:
                dist, value = self.network(state, mask=valid_mask)  # 动作的概率分布 (shape: [output_dim]), 一定是大于0的值
                action = torch.argmax(dist.probs)  # 选择概率最大的动作索引
                logp = action_list.log_prob(action)     # 计算动作的log_prob
                action = action.item()

        else:
            if steps == 0:  # 第一步采用固定动作
                y = round(env.area_height / 2 - env.sorted_comp_grid_shapes[steps].shape[0] / 2 + 5)
                x = round(env.area_width / 2 - env.sorted_comp_grid_shapes[steps].shape[1] / 2 - 15)
                rotation = 3
                action = rotation * env.area_width * env.area_height + y * env.area_width + x
                dist, value = self.network(state, mask=valid_mask)
                action_tensor = torch.tensor(action, dtype=torch.long, device=device)
                logp = dist.log_prob(action_tensor)  # 计算固定动作的log_prob，第一个维度是batch_size
            else:
                # embedding_value = self.embedding(state)
                dist, value = self.network(state, mask=valid_mask)
                action = dist.sample()  # 根据概率随机选择动作
                logp = dist.log_prob(action)  # 计算动作的log_prob
                action = action.item()

        # 使用detach截断计算图，避免梯度重复反向传播（这里为什么squeeze？）
        value, logp = value.squeeze(0).detach(), logp.detach()

        return action, value, logp

    # 训练
    def learn(self, batch_data):
        # ==== 获取训练模型所需数据 ====
        # 提取数据集
        states, actions, old_logps, values, rewards, next_states, dones, game_overs = zip(*batch_data)
        # 状态合并，生成批次图
        states = self.merge_states(states)  # 合并样本，生成一个批次图
        next_states = self.merge_states(next_states)
        # -------------- cpu ---------------
        # 对奖励进行归一化（标准化）
        rewards = self.normalize_rewards(rewards)  # 归一化奖励
        # 计算折扣奖励 returns
        returns = []
        discounted = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                discounted = 0
            discounted = r + self.gamma * discounted
            returns.insert(0, discounted)
        # 计算 GAE 优势
        advantages = self.compute_gae(rewards, values, dones, game_overs)
        # ------------------------------------

        # ==== 更新模型 ====
        # 转换为张量，维度匹配模型输出的state_value的维度，并迁移到GPU上
        actions = torch.tensor(actions, dtype=torch.long, device=device).view(-1, 1)    # (n * num_steps, 1)
        old_logps = torch.stack(old_logps).unsqueeze(-1)    # stack是在新维度上拼接
        returns = torch.tensor(returns, dtype=torch.float,device=device).unsqueeze(-1)
        advantages = torch.tensor(advantages, dtype=torch.float, device=device).unsqueeze(-1)

        # 一组数据训练 epochs轮
        # progress_bar = tqdm(range(self.epochs), unit="epoch", desc="training model...")
        for i in range(self.epochs):
            # 1) 重新计算动作概率与状态值
            dist, values = self.network(states)     # 新策略
            new_logps = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
            entropy = dist.entropy().mean()  # 平均熵

            # 2) 计算新旧策略比例 ratio 与剪切目标
            ratios = torch.exp(new_logps - old_logps)  # r_t(θ)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns)
            total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy

            # 3) 梯度下降一步
            self.optimizer.zero_grad()  # 清零上一轮的梯度
            total_loss.backward()       # 反向传播，计算梯度
            self.optimizer.step()       # 利用梯度更新模型权重

        # 清空数据，为下次采样做准备
        # 每次训练完后要将batch_data清空，否则就不是该策略下的采样了
        batch_data.clear()


    def compute_gae(self, rewards, values, dones, game_overs):
        """
        使用广义优势估计 (GAE) 计算优势函数
        rewards: list of r_t
        values: list of V(s_t)
        dones:   list of episode done flags
        返回: advantages
        """
        advantages = []
        gae = 0
        # 在 values 末尾添加一个 0，方便计算最后一步的 delta
        values = list(values) + [0]
        # 反向遍历时间步，计算每一步的 GAE
        for step in reversed(range(len(rewards))):
            # δ_t = r_t + γ * V(s_{t+1}) * (1−done) − V(s_t)
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step])*(1-game_overs[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step])*(1-game_overs[step]) * gae
            advantages.insert(0, gae)
        return advantages

    # def normalize_rewards(self, rewards, gamma=0.99, epsilon=1e-8):
    #     """
    #     对奖励进行归一化
    #
    #     参数:
    #         rewards (torch.Tensor): 奖励张量
    #         gamma (float): 衰减因子，用于计算加权平均
    #         epsilon (float): 防止除以零的小常数
    #
    #     返回:
    #         torch.Tensor: 归一化后的奖励
    #     """
    #     mean = rewards.mean().item()  # 计算奖励的均值
    #     std = rewards.std().item()  # 计算奖励的标准差
    #
    #     # 归一化奖励
    #     normalized_rewards = (rewards - mean) / (std + epsilon)
    #
    #     return normalized_rewards

    def normalize_rewards(self, reward_tuple, epsilon=1e-8):
        """
        对 PPO 中存储在元组中的奖励进行标准化处理。

        参数：
            reward_tuple (tuple or list of floats): 原始奖励，可能是多个 episode 的合并。
            epsilon (float): 防止除以零的小常数。

        返回：
            normalized_rewards (np.ndarray): 归一化后的奖励数组。
        """
        rewards = np.array(reward_tuple, dtype=np.float32)
        mean = np.mean(rewards)
        std = np.std(rewards)
        normalized = (rewards - mean) / (std + epsilon)
        return tuple(normalized)

    def merge_states(self, states):
        """
            把多个图合并成一个批处理图，服务于按batch训练
        """
        # 初始化空的列表来存储合并后的数据
        all_x = []
        all_edge_index = []
        all_batch = []

        # 当前节点索引的偏移量
        node_offset = 0
        batch_offset = 0  # 用于批处理信息

        for idx, state in enumerate(states):
            # 获取当前图的节点特征和边
            x, edge_index = state.x, state.edge_index

            # 1. 合并节点特征
            all_x.append(x)

            # 2. 合并边的连接信息
            # 为了处理每个图的边，我们需要对边的节点索引进行偏移
            edge_index = edge_index + node_offset  # 增加偏移量
            all_edge_index.append(edge_index)

            # 3. 生成批处理信息，每个图的所有节点的 batch 值是当前图的索引
            batch = torch.full((x.size(0),), batch_offset, dtype=torch.long, device=device)
            all_batch.append(batch)

            # 更新偏移量
            node_offset += x.size(0)
            batch_offset += 1  # 每个图的 batch 值递增

        # 将所有部分合并成一个张量
        x = torch.cat(all_x, dim=0)  # 合并所有节点特征
        edge_index = torch.cat(all_edge_index, dim=1)  # 合并所有边的连接信息
        batch = torch.cat(all_batch, dim=0)  # 合并所有的 batch 信息

        # 创建最终的 Data 对象
        merged_data = Data(x=x, edge_index=edge_index, batch=batch)

        return merged_data

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

    def calculate_roundness(self, area, perimeter):
        """
            计算圆度，表征形状接近圆形的程度
            area: 面积
            perimeter: 周长
        """
        if perimeter == 0:  # 防止除以零
            return 0
        roundness = (4 * np.pi * area) / (perimeter ** 2)
        return roundness

    def generate_state(self, comps_name, comp_coords, comps_shape, pins, reward):
        # 生成未放置器件时的状态
        if not comps_name:
            x = torch.zeros(1, 6)    # 创建虚拟节点
            edge_index = torch.empty((2, 0), dtype=torch.long)
            return Data(x=x, edge_index=edge_index).to(device)


        # 将每个 ndarray 转换为 list
        comps_shape_list = [arr.tolist() for arr in comps_shape]
        output_data = {}

        # 整合当前环境信息到output_data中，{器件a的名：a的信息字典}
        for i, comp_name in enumerate(comps_name):
            comp_x, comp_y, comp_rotation = comp_coords[i]
            comp_shape = comps_shape_list[i]
            comp_pins = {}

            # 提取当前器件的引脚信息
            for pin in pins:
                if pin["comp_name"] == comp_name:
                    pin_name = pin["pin_net_name"]  # 假设引脚名称为 pin_net_name
                    comp_pins[pin_name] = {
                        "pin_name": pin_name,
                        "pin_net_name": pin["pin_net_name"],
                        "pin_x": pin["pin_x"],
                        "pin_y": pin["pin_y"],
                        "pin_rotation": pin["pin_rotation"],
                        "pin_shape": []  # 这里可以根据需要添加引脚形状
                    }

            # 构建器件信息
            output_data[comp_name] = {
                "comp_name": comp_name,
                "comp_x": comp_x,
                "comp_y": comp_y,
                "comp_rotation": comp_rotation,
                "comp_shape": comp_shape,
                "comp_comment": "",  # 这里可以添加注释信息
                "comp_pin": comp_pins
            }

        # 添加 reward 字段
        reward = {
            "value": reward  # 你可以根据需要设置 reward 的值
        }
        output_data["reward"] = reward

        # 整合所有pin信息（连线的单位是pin而不是器件）
        if len(output_data.get('comps_info', {})) == 0:
            comps_info = output_data
        else:
            comps_info = output_data.get('comps_info', {})
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

        # print("pins", pins)
        # num_nodes = len(pins)
        # if num_nodes == 0:
        #     # 处理没有节点的情况，可以跳过或返回空图
        #     raise ValueError(f"No pins found in file: {filepath}")

        # 计算节点特征（周长和面积）todo 增加特征
        node_features = []
        for pin in pins:
            perimeter, area = self.compute_perimeter_area(pin['comp_shape'])
            shape_features = compute_shape_features(pin)  # 宽度、高度
            comp_shape_cur = pin['comp_shape']
            x_coords = [coord[0] for coord in comp_shape_cur]
            y_coords = [coord[1] for coord in comp_shape_cur]
            ratio = shape_features[0] / shape_features[1]   # 宽高比
            convexity = self.calculate_roundness(area, perimeter)   # 圆度
            node_features.append([perimeter, area] + shape_features + [ratio] + [convexity])

        x = torch.tensor(node_features, dtype=torch.float)  # 节点特征，[num_nodes, 2]

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


        # 创建 PyG 的 Data 对象（创建图对象）
        data_pyg = Data(x=x, edge_index=edge_index).to(device)

        # if self.transform:
        #     data_pyg = self.transform(data_pyg)
        return data_pyg



def train_ppo2():
    n_states = env.observation_space.shape[0] * env.observation_space.shape[1]
    n_actions = env.action_space.n * 4  # 每个位置还对应4个旋转角度
    agent = PPO(n_states=n_states,  # 状态数
                n_hiddens=n_hiddens,  # 隐含层数
                n_actions=n_actions,  # 动作数
                # actor_lr=actor_lr,  # 策略网络学习率
                # critic_lr=critic_lr,  # 价值网络学习率
                lr=lr,
                lmbda=0.99,  # 优势函数的缩放因子
                epochs=50,  # 一组序列训练的轮次
                eps=0.2,  # PPO中截断范围的参数
                gamma=gamma,  # 折扣因子
                device=device
                )

    # ----------------------------------------- #
    # 训练--回合更新 on_policy
    # ----------------------------------------- #

    # transition_buffer = deque(maxlen=10000) # 缓冲区，用来存储过渡样本
    # 在每个回合内存储当前回合的样本
    best_return = float('-inf')  # 初始化为负无穷
    batch_data = []
    prog_bar = tqdm(range(num_episodes), desc="Training PPO Agent...", unit="episode")
    for i in prog_bar:
        # print("episodes", i)
        episode_data = []
        num = 0
        reward = 0
        grid, mask, current_place = env.reset()  # 环境重置
        state =  agent.generate_state([], comp_coords, env.placement_shape, env.pins, reward)
        done = False        # 任务结束标记，一般是没地放了
        game_over = False   # 任务完成的标记，布局完成了
        episode_return = 0  # 累计每回合的reward
        steps = 0

        while not (game_over or done):

            action, value, logp = agent.take_action(env, state, steps, deterministic=False)  # 动作选择
            grid, mask, current_place, reward, done, game_over = env.step(action, steps)  # 环境更新
            real_coord = [((x1 + x2) * grid_space + env.area_min_x, env.area_max_y - (y1 + y2) * grid_space, rotation * 90)
                          for (y1, x1, rotation), (y2, x2) in zip(env.action_set, env.placement_set)]
            if not (game_over or done):
                next_state = agent.generate_state(env.sorted_comp_names[:steps + 1], real_coord, env.placement_shape, env.pins, reward)
            else:
                next_state = copy.deepcopy(state)   # 如果游戏结束，那next_state == state
            # print("reward", reward)
            # env.plot_area_demo(env.area_grid_shape, steps)
            steps += 1
            num += 1
            episode_data.append((
                state,
                action,
                logp,
                value.item(),
                reward,
                next_state,
                done,
                game_over))

            # transition_buffer.extend(episode_data)

            # 更新状态
            state = next_state
            # 累计回合奖励
            episode_return += reward
            # print("done", done)
            # print("game_over", game_over)


        # if (len(transition_buffer) > batch_size):
        #     # 从缓冲区中随机选择 batch_size 个样本
        #     batch_data = random.sample(transition_buffer, batch_size)
        #     # 模型训练
        #     agent.learn(batch_data)
        #     # agent.learn(transition_dict)
        #
        #     # 打印回合信息
        #     print(f'iter:{i}, return:{np.mean(return_list[-10:])}')


        batch_data.extend(episode_data)
        # print(batch_data)
        if(len(batch_data)>batch_size):
            agent.learn(batch_data)     # 更新策略网络和价值网络
        # 保存每个回合的return

        num_list.append(num)
        return_list.append(episode_return)
        plot_rewards(return_list)
        plot_nums(num_list)
        # 打印回合信息
        # print(f'iter:{i}, return:{np.mean(return_list[-10:])}')

        # -------------------------------------- #
        # 保存奖励最大的模型
        # -------------------------------------- #
        if episode_return > best_return:  # 如果当前回合的奖励超过历史最高奖励
            best_return = episode_return  # 更新最高奖励
            torch.save({
                "actor_critic_optimizer_state_dict": agent.network.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                'return_list': return_list,  # 可选：保存训练过程中的累计奖励
                'best_return': best_return  # 保存当前的最大奖励
            }, save_path)

            # print(f"New best model saved with return: {best_return} to {save_path}")

    # print(f"Model saved to {save_path}")


    # -------------------------------------- #
    # 绘图
    # -------------------------------------- #

    plt.plot(return_list)
    plt.title('return')
    plt.show()

# 绘制奖励图
def plot_nums(rewards_log):
    plt.figure(figsize=(10, 10))
    plt.plot(rewards_log, label="Nums log", color='blue', alpha=0.7)

    # 计算滑动平均（可选）
    window_size = 10
    if len(rewards_log) >= window_size:
        moving_avg = np.convolve(rewards_log, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(rewards_log)), moving_avg, label="Moving Average (10 Episodes)",
                 color='orange', linewidth=2)

    plt.title("nums per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Nums")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("nums_log_plot.png")  # 保存图像为文件
    # plt.show()

# 绘制奖励图
def plot_rewards(rewards_log):
    plt.figure(figsize=(10, 10))
    plt.plot(rewards_log, label="Episode Reward", color='blue', alpha=0.7)

    # 计算滑动平均（可选）
    window_size = 10
    if len(rewards_log) >= window_size:
        moving_avg = np.convolve(rewards_log, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(rewards_log)), moving_avg, label="Moving Average (10 Episodes)",
                 color='orange', linewidth=2)

    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("rewards_log_plot.png")  # 保存图像为文件
    # plt.show()

def load_module_data(filename):
    with open(filename, 'r') as f:
        module_data = json.load(f)
    return module_data

def get_data():
    file_path = '../data_test'
    id = 1
    filename = f"{file_path}/data{id}.json"

    module_data = load_module_data(filename)
    area = module_data.get('area', [])
    area_shape = np.array(area)
    comps_name = []
    comps_shape = []
    comps_comment = []
    comps_grid_shape = []
    comp_coords = []
    pins = []
    # components, pins, area_features, area_origin = process_module_data_test(module_data)
    comps_info = module_data.get('comps_info', {})

    for comp_name, comp in comps_info.items():
        comp_data = {
            'comp_name': comp_name,
            'comp_rotation': comp.get('comp_rotation', 0.0),
            'comp_shape': comp.get('comp_shape', []),
            'comp_x': comp.get('comp_x', 0.0),
            'comp_y': comp.get('comp_y', 0.0),
            'comp_coords': (comp.get('comp_x', 0.0), comp.get('comp_y', 0.0)),
            'comp_comment': comp.get('comp_comment', [])
        }
        comps_name.append(comp_data['comp_name'])
        comps_shape.append(comp_data['comp_shape'])
        comps_comment.append(comp_data['comp_comment'])
        comp_coords.append(comp_data['comp_coords'])

        # 获取pin信息确定不同器件之间的连接关系
        for pin_number, pin in comp.get('comp_pin', {}).items():
            pin_x = pin.get('pin_x')
            pin_y = pin.get('pin_y')
            # 得把pin_x和pin_y换成相对于comp_coord的坐标，因为测试集是这样的形式给进去的
            pin_rotation = pin.get('pin_rotation')
            pin_data = {
                'comp_name': comp_name,
                'pin_net_name': pin.get('pin_net_name'),
                'pin_x': pin_x - comp_data['comp_x'],
                'pin_y': pin_y - comp_data['comp_y'],  # 这两个也要换成相对坐标的形式
                'pin_rotation': pin_rotation,
            }
            pins.append(pin_data)

    comps_type = extract_keywords(comps_comment, keywords)
    comp_coords = np.array(comp_coords)

    return area_shape, comps_shape, comps_name, comp_coords, pins, comps_type

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

    comp_name_to_id = {comps_name[idx]: idx for idx in range(len(comps_name))}
    id_to_comp_name = {idx: comps_name[idx] for idx in range(len(comps_name))}
    edges_id = [(comp_name_to_id[src], comp_name_to_id[dst]) for src, dst in edges]

    return edges_id, comp_name_to_id, net_to_comps, comp_pins

def extract_keywords(device_descriptions, keywords):
    extracted_info = []

    # 遍历每个器件描述
    for description in device_descriptions:
        current_info = []

        # 检查关键词是否存在于描述中
        for keyword in keywords:
            if keyword in description:
                current_info.append(keyword)

            # 去重后添加到结果中
            # extracted_info.append(list(set(current_info)))
        extracted_info.append(current_info if current_info else ["-"])
    return extracted_info

if __name__ == '__main__':
    area_shape, comps_shape, comps_name, comp_coords, pins, comps_type = get_data()
    # print("comp_coodes", comp_coords)
    edges_id, comp_name_to_id, net_to_comps, comp_pins = build_graph(comps_name, pins)
    env = PlacementEnv(area_shape=area_shape, comp_shapes=comps_shape, comp_names=comps_name,
                       comp_coords=comp_coords, edges_id=edges_id, comp_name_to_id=comp_name_to_id,
                       comps_type=comps_type,
                       net_to_comps=net_to_comps, comp_pins=comp_pins, pins=pins)

    train_ppo2()




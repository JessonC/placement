import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from placement_game import PCBRLEnv
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import random
import time
import os
import json
# 读取state
with open('pcb_fused_rl_state.pkl', 'rb') as f:
    fused_state = pickle.load(f)['pcb_cells_1.json']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 简单GCN实现
class SimpleGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, hidden_feats)
        self.fc2 = nn.Linear(hidden_feats, out_feats)

    def forward(self, node_feats, edges):
        h = F.relu(self.fc1(node_feats))
        h = self.fc2(h)
        h = torch.mean(h, dim=0)  # 全图平均特征
        return h

# 小ResNet-like CNN
class SimpleCNN(nn.Module):
    def __init__(self, out_feats):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(16, out_feats)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Transformer编码器
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, out_feats):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, out_feats)

    def forward(self, seq):
        embed = self.embedding(seq).permute(1, 0, 2)
        h = self.transformer(embed)
        h = h.mean(dim=0)
        return self.fc(h)

# PPO actor模型
class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dims):  # action_dims: [2, 16, 5, 5, 5, 4]
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(128, dim) for dim in action_dims])

    def forward(self, state):
        feat = self.fc(state)
        outs = [head(feat) for head in self.heads]  # list of [batch, out_dim]
        return outs  # 每个action分支


# 主模型
class PCBPolicy(nn.Module):
    def __init__(self, action_dims):
        super().__init__()
        self.gcn = SimpleGCN(3, 32, 32)
        self.cnn = SimpleCNN(32)
        self.transformer = SimpleTransformer(2000, 32, 4, 32)
        self.fc_fusion = nn.Linear(96, 128)
        self.actor = PPOActor(128, action_dims)  # <-- 注意这里

    def forward(self, graph, image, sequence):
        graph_feat = self.gcn(graph['node_feats'], graph['edges'])
        img_feat = self.cnn(image)
        seq_feat = self.transformer(sequence)

        # 确保全部是一维向量
        if graph_feat.dim() > 1:
            graph_feat = graph_feat.view(-1)
        if img_feat.dim() > 1:
            img_feat = img_feat.view(-1)
        if seq_feat.dim() > 1:
            seq_feat = seq_feat.view(-1)
        fused_feat = torch.cat([graph_feat, img_feat, seq_feat], dim=-1)
        fused_feat = F.relu(self.fc_fusion(fused_feat))
        action_logits = self.actor(fused_feat)
        return action_logits


# PPO框架
class PPO:
    def __init__(self, policy, lr=1e-4):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

    def select_action(self, state):
        outs = self.policy(*state)  # 6个logits
        actions, log_probs = [], []
        for out in outs:
            prob = F.softmax(out, dim=-1)
            action = torch.multinomial(prob, num_samples=1).item()
            actions.append(action)
            log_probs.append(torch.log(prob[action]))
        return actions, sum(log_probs)

    def update(self, log_prob, reward):
        loss = -(log_prob * reward).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# 创建环境
env = PCBRLEnv('pcb_pre_jsons/pcb_cells_1.json')
env.reset()

# 模型与PPO优化器
action_dims = [2, 16, 5, 5, 5, 4]
policy = PCBPolicy(action_dims=action_dims).to(device)
ppo = PPO(policy)

# 环境交互并训练
# === numpy转torch.tensor 并放到正确device上 ===
graph = {
    'node_feats': torch.tensor(fused_state['graph']['node_feats'], dtype=torch.float32, device=device),
    'edges': torch.tensor(fused_state['graph']['edges'], dtype=torch.long, device=device),
}
image = torch.tensor(fused_state['image'], dtype=torch.float32, device=device).unsqueeze(0)
sequence = torch.tensor(fused_state['sequence'], dtype=torch.long, device=device).unsqueeze(0)


# 训练示例 (单次迭代)
state = (graph, image, sequence)
action_list, log_probs = [], []

# 逐模块action决策
for module_idx in range(env.n_cells - 1):
    action, log_prob = ppo.select_action(state)
    action_list.append(action)
    log_probs.append(log_prob)

# 执行动作，获取reward
_, reward, done, _ = env.step(action_list)

# PPO update
reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
log_probs = torch.stack(log_probs)
ppo.update(log_probs, reward_tensor)

print(f"Reward: {reward}")

def train_ppo(env, fused_state, policy, total_epochs=500, gamma=0.99, clip_eps=0.2, batch_size=8, lr=3e-4, logdir="runs/pcbppo"):
    device = next(policy.parameters()).device
    action_dims = [2, 16, 5, 5, 5, 4]

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    writer = SummaryWriter(logdir)
    global_step = 0

    # graph特征只需要初始化一次
    graph = {
        'node_feats': torch.tensor(fused_state['graph']['node_feats'], dtype=torch.float32, device=device),
        'edges': torch.tensor(fused_state['graph']['edges'], dtype=torch.long, device=device),
    }
    base_image = torch.tensor(fused_state['image'], dtype=torch.float32, device=device).unsqueeze(0)
    base_sequence = torch.tensor(fused_state['sequence'], dtype=torch.long, device=device).unsqueeze(0)

    for epoch in range(1, total_epochs + 1):
        state = (graph, base_image, base_sequence)
        ep_rewards = []
        ep_policy_loss = []
        ep_value_loss = []
        ep_advantages = []

        for episode in range(batch_size):
            # 重置env到初始状态
            env.reset()
            action_list, log_probs = [], []

            # PPO仅示例主干，action采样与env步进
            for module_idx in range(env.n_cells - 1):
                outs = policy(*state)   # actor多头输出
                actions, action_log_probs = [], []
                for out in outs:
                    prob = F.softmax(out, dim=-1)
                    action = torch.multinomial(prob, num_samples=1).item()
                    actions.append(action)
                    action_log_probs.append(torch.log(prob[action]))
                action_list.append(actions)
                log_probs.append(sum(action_log_probs))
            # env步进
            _, reward, done, _ = env.step(action_list)
            ep_rewards.append(reward)
            log_probs = torch.stack(log_probs)
            # 假定无值函数，这里简单用reward做优势
            advantages = torch.tensor(reward, dtype=torch.float32, device=device)
            ep_advantages.append(advantages)

            # policy loss（仅演示，不是真正的PPO损失）
            policy_loss = -(log_probs.mean() * advantages)
            ep_policy_loss.append(policy_loss.item())

            # 反向与更新
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
            global_step += 1

        # TensorBoard logging
        mean_reward = np.mean(ep_rewards)
        mean_policy_loss = np.mean(ep_policy_loss)
        mean_advantage = np.mean([a.cpu().numpy() for a in ep_advantages])
        writer.add_scalar("Reward/mean", mean_reward, epoch)
        writer.add_scalar("PolicyLoss/mean", mean_policy_loss, epoch)
        writer.add_scalar("Advantage/mean", mean_advantage, epoch)
        print(f"Epoch {epoch}: Reward={mean_reward:.2f}, PolicyLoss={mean_policy_loss:.4f}, Advantage={mean_advantage:.2f}")

    writer.close()

def visualize_pcb_image_from_env(env):
    # 重新build image信息，用当前env.cell_list和contour
    img = np.zeros((256, 256), dtype=np.float32)
    num_cells = len(env.cell_list)
    for idx, cell in enumerate(env.cell_list):
        contour = np.array(eval(cell["contour"]), dtype=np.int32)
        gray_val = (idx + 1) / num_cells
        cv2.fillPoly(img, [contour], color=gray_val)
    img = np.clip(img, 0, 1)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.title('PCB layout after PPO')
    plt.axis('on')
    plt.show()

def color_palette(n):
    # 随机调色盘，不重复
    import matplotlib
    colors = list(matplotlib.colormaps['tab20'].colors)
    random.shuffle(colors)
    while len(colors) < n:
        colors += colors
    return colors[:n]

def plot_pcb_from_env(env, title="PCB (Current Env)"):
    # env应有cell_list和net_list
    cell_list = env.cell_list
    net_list = getattr(env, 'net_list', [])
    fig, ax = plt.subplots(figsize=(8,8))
    color_map = {}
    colors = color_palette(len(cell_list))
    for idx, cell in enumerate(cell_list):
        c = colors[idx]
        contour = eval(cell["contour"])
        xs = [pt[0] for pt in contour] + [contour[0][0]]
        ys = [pt[1] for pt in contour] + [contour[0][1]]
        ax.plot(xs, ys, color=c, linewidth=2)
        cell_center = eval(cell["center"])
        ax.text(cell_center[0], cell_center[1], cell["cellName"],
                color=c, fontsize=9, ha='center', va='center', bbox=dict(facecolor='white', edgecolor=c, boxstyle='round,pad=0.3', alpha=0.7))
        for pin in cell["pinList"]:
            pcenter = eval(pin["center"])
            ax.plot(pcenter[0], pcenter[1], marker='o', color=c, markersize=4, alpha=0.7)
        color_map[cell["cellName"]] = c
    for net in net_list:
        pinList = net["pinList"]
        if len(pinList) < 2:
            continue
        pin_coords = []
        for pin in pinList:
            cell = next((c for c in cell_list if c["cellName"]==pin["cellName"]), None)
            if cell:
                pinitem = next((p for p in cell["pinList"] if p["pinName"]==pin["pinName"]), None)
                if pinitem:
                    pin_coords.append(eval(pinitem["center"]))
        for i in range(len(pin_coords)-1):
            x1, y1 = pin_coords[i]
            x2, y2 = pin_coords[i+1]
            ax.plot([x1, x2], [y1, y2], color='gray', linestyle='-', linewidth=1, alpha=0.8)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_pcb_compare_side_by_side(env, pre_json_folder="pcb_cell_jsons", title="PCB Optimization Comparison"):
    import os
    import json
    import matplotlib.pyplot as plt
    import numpy as np

    def color_palette(n):
        import matplotlib
        colors = list(matplotlib.colormaps['tab20'].colors)
        random.shuffle(colors)
        while len(colors) < n:
            colors += colors
        return colors[:n]

    # 1. 找到原始布局文件
    # 优先用env.raw_data里的文件名或主芯片推测
    fname = env.raw_data.get("json_name", None)
    if fname is None:
        main_name = env.cell_list[env.main_idx]["cellName"]
        for fn in os.listdir(pre_json_folder):
            if fn.endswith(".json") and main_name in fn:
                fname = fn
                break
    if fname is None:
        fname = os.listdir(pre_json_folder)[0]  # 兜底任选一个
    with open(os.path.join(pre_json_folder, fname), "r", encoding="utf-8") as f:
        pre_pcb_data = json.load(f)

    pre_cell_list = pre_pcb_data["cellList"]
    pre_net_list = pre_pcb_data.get("netList", [])

    post_cell_list = env.cell_list
    post_net_list = env.net_list

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    titles = ["优化前布局 (PRE)", "优化后布局 (POST)"]
    lists = [(pre_cell_list, pre_net_list, color_palette(len(pre_cell_list))),
             (post_cell_list, post_net_list, color_palette(len(post_cell_list)))]

    for idx, (cell_list, net_list, colors) in enumerate(lists):
        ax = axes[idx]
        for j, cell in enumerate(cell_list):
            c = colors[j]
            contour = eval(cell["contour"])
            xs = [pt[0] for pt in contour] + [contour[0][0]]
            ys = [pt[1] for pt in contour] + [contour[0][1]]
            ax.plot(xs, ys, color=c, linewidth=2)
            cell_center = eval(cell["center"])
            ax.text(cell_center[0], cell_center[1], cell["cellName"],
                    color=c, fontsize=9, ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor=c, boxstyle='round,pad=0.3', alpha=0.8))
            for pin in cell["pinList"]:
                pcenter = eval(pin["center"])
                ax.plot(pcenter[0], pcenter[1], marker='o', color=c, markersize=4, alpha=0.7)
        # 画net连线
        for net in net_list:
            pinList = net["pinList"]
            if len(pinList) < 2:
                continue
            pin_coords = []
            for pin in pinList:
                cell = next((c for c in cell_list if c["cellName"]==pin["cellName"]), None)
                if cell:
                    pinitem = next((p for p in cell["pinList"] if p["pinName"]==pin["pinName"]), None)
                    if pinitem:
                        pin_coords.append(eval(pinitem["center"]))
            for i in range(len(pin_coords)-1):
                x1, y1 = pin_coords[i]
                x2, y2 = pin_coords[i+1]
                ax.plot([x1, x2], [y1, y2], color='gray', linestyle='-', linewidth=1, alpha=0.8)
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.set_title(titles[idx])
        ax.set_aspect('equal')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()



# -------------------------------
# 主入口
if __name__ == "__main__":
    from placement_game import PCBRLEnv
    import pickle
    # 只用pcb_cells_1.json为例
    with open('pcb_fused_rl_state.pkl', 'rb') as f:
        fused_state = pickle.load(f)['pcb_cells_1.json']
    env = PCBRLEnv('pcb_pre_jsons/pcb_cells_1.json')
    action_dims = [2, 16, 5, 5, 5, 4]
    policy = PCBPolicy(action_dims=action_dims).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    train_ppo(env, fused_state, policy, total_epochs=200, batch_size=4, logdir="runs/pcbppo")
    plot_pcb_compare_side_by_side(env)
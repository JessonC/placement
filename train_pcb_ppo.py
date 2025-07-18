import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from placement_game import PCBRLEnv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import random
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('pcb_fused_rl_state.pkl', 'rb') as f:
    fused_state = pickle.load(f)['pcb_cells_1.json']
# 模型定义
class SimpleGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, hidden_feats)
        self.fc2 = nn.Linear(hidden_feats, out_feats)

    def forward(self, node_feats, edges):
        h = F.relu(self.fc1(node_feats))
        h = self.fc2(h)
        return torch.mean(h, dim=0)

class SimpleCNN(nn.Module):
    def __init__(self, out_feats):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2), nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(16, out_feats)

    def forward(self, x):
        x = self.cnn(x.unsqueeze(1))
        return self.fc(x.view(x.size(0), -1)).squeeze(0)

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, out_feats):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        encoder = nn.TransformerEncoderLayer(embed_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
        self.fc = nn.Linear(embed_dim, out_feats)

    def forward(self, seq):
        emb = self.embed(seq).permute(1, 0, 2)
        h = self.transformer(emb).mean(0)
        return self.fc(h).squeeze(0)

class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dims):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(128, d) for d in action_dims])

    def forward(self, state):
        feat = self.fc(state)
        return [head(feat) for head in self.heads]

class PCBPolicy(nn.Module):
    def __init__(self, action_dims):
        super().__init__()
        self.gcn = SimpleGCN(3, 32, 32)
        self.cnn = SimpleCNN(32)
        self.transformer = SimpleTransformer(2000, 32, 32)
        self.fc_fusion = nn.Linear(96, 128)
        self.actor = PPOActor(128, action_dims)

    def forward(self, graph, image, sequence):
        g_feat = self.gcn(graph['node_feats'], graph['edges'])
        i_feat = self.cnn(image)
        s_feat = self.transformer(sequence)
        feat = torch.cat([g_feat, i_feat, s_feat], dim=-1)
        feat = F.relu(self.fc_fusion(feat))
        return self.actor(feat)

# 载入初始特征
graph = {
    'node_feats': torch.tensor(fused_state['graph']['node_feats'], dtype=torch.float32, device=device),
    'edges': torch.tensor(fused_state['graph']['edges'], dtype=torch.long, device=device),
}

# 初始化环境
env = PCBRLEnv('pcb_pre_jsons/pcb_cells_1.json')

graph = {
    'node_feats': torch.tensor(fused_state['graph']['node_feats'], dtype=torch.float32, device=device),
    'edges': torch.tensor(fused_state['graph']['edges'], dtype=torch.long, device=device),
}
base_image = torch.tensor(fused_state['image'], dtype=torch.float32, device=device).unsqueeze(0)
base_sequence = torch.tensor(fused_state['sequence'], dtype=torch.long, device=device).unsqueeze(0)

env = PCBRLEnv('pcb_pre_jsons/pcb_cells_1.json')
state = (graph, base_image, base_sequence)

action_dims = [2, 16, 5, 5, 5, 4]
policy = PCBPolicy(action_dims).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-4)

writer = SummaryWriter(log_dir='runs/pcb_ppo')

# PPO采样action函数
def select_action(policy, state):
    outs = policy(*state)
    actions, log_probs = [], []
    for out in outs:
        prob = F.softmax(out, dim=-1)
        a = torch.multinomial(prob, 1).item()
        actions.append(a)
        log_probs.append(torch.log(prob[a]))
    return actions, sum(log_probs)

def show_image(image, title="Current PCB Image"):
    plt.figure(figsize=(6, 6))
    plt.imshow(image.cpu().squeeze(0).numpy(), cmap='gray', vmin=0, vmax=1)
    plt.title(title)
    plt.axis('on')
    plt.show()

def build_graph_info(env):
    nodes = []
    for cell in env.cell_list:
        for pin in cell["pinList"]:
            nodes.append((cell["cellName"], pin["pinName"]))
    return {"nodes": nodes}

def get_pin_center(cell_list, cell_name, pin_name):
    cell = next((c for c in cell_list if c["cellName"] == cell_name), None)
    if cell:
        pin = next((p for p in cell["pinList"] if p["pinName"] == pin_name), None)
        if pin:
            return np.array(eval(pin["center"]))
    return np.zeros(2)

def build_sequence_info(env):
    SOS, SOE, MOS, MOE = 0, 1, 2, 3
    token_seq = [SOS]

    graph_info = build_graph_info(env)
    node_key2idx = {key: idx for idx, key in enumerate(graph_info["nodes"])}

    cell_name2idx = {cell["cellName"]: idx + 10 for idx, cell in enumerate(env.cell_list)}

    for cell in env.cell_list:
        module_tokens = []
        cell_idx = cell_name2idx[cell["cellName"]]

        for net in env.net_list:
            pins_in_cell = [p for p in net["pinList"] if p["cellName"] == cell["cellName"]]
            pins_outside_cell = [p for p in net["pinList"] if p["cellName"] != cell["cellName"]]

            for p_in in pins_in_cell:
                key_in = (p_in["cellName"], p_in["pinName"])
                if key_in not in node_key2idx:
                    continue
                node_in = node_key2idx[key_in]
                for p_out in pins_outside_cell:
                    key_out = (p_out["cellName"], p_out["pinName"])
                    if key_out not in node_key2idx:
                        continue
                    node_out = node_key2idx[key_out]

                    pin_in_center = get_pin_center(env.cell_list, *key_in)
                    pin_out_center = get_pin_center(env.cell_list, *key_out)
                    dist = int(np.linalg.norm(pin_in_center - pin_out_center))

                    module_tokens.extend([node_in + 100, node_out + 100, dist])

        if module_tokens:
            token_seq.extend([MOS, cell_idx])
            token_seq.extend(module_tokens)
            token_seq.append(MOE)

    token_seq.append(SOE)
    return np.array(token_seq, dtype=np.int32)

def update_features(env, show=False):
    img = np.zeros((256, 256), dtype=np.float32)
    for idx, cell in enumerate(env.cell_list):
        contour = np.array(eval(cell["contour"]))
        cv2.fillPoly(img, [contour], (idx + 1) / len(env.cell_list))
    img = np.clip(img, 0, 1)
    image = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0)

    # if show:
    #     show_image(image)

    sequence = build_sequence_info(env)
    sequence = torch.tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)

    return image, sequence

# PPO训练循环
policy = PCBPolicy(action_dims).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-4)
writer = SummaryWriter(log_dir='runs/pcb_ppo')

state = (graph, *update_features(env, show=True))

epochs = 1000
for epoch in range(epochs):
    env.reset()
    log_probs_epoch = []

    action_list = []
    for module_idx in range(env.n_cells - 1):
        action, log_prob = select_action(policy, state)
        action_list.append(action)
        log_probs_epoch.append(log_prob)

    # print(f"\n[Epoch {epoch+1}] Actions:")
    cell_names = env.get_cell_names()
    main_idx = env.main_idx
    for i, act in enumerate(action_list):
        cell_idx = i if i < main_idx else i+1
        cell_name = cell_names[cell_idx]
        # print(f"  {cell_name}: {act}")

    _, reward, _, _ = env.step(action_list)

    base_image, base_sequence = update_features(env, show=True)
    state = (graph, base_image, base_sequence)

    loss = -torch.stack(log_probs_epoch).mean() * reward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('Reward', reward, epoch)
    writer.add_scalar('Loss', loss.item(), epoch)

    print(f"[Epoch {epoch+1}] Reward: {reward:.2f}, Loss: {loss.item():.4f}")

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

# 布局可视化函数
def plot_pcb(env):
    title = "Test"
    fig, ax = plt.subplots(figsize=(8,8))
    colors = plt.cm.tab20.colors
    for idx, cell in enumerate(env.cell_list):
        c = colors[idx % len(colors)]
        contour = np.array(eval(cell["contour"]))
        ax.plot(*np.append(contour, [contour[0]], axis=0).T, c=c)
        center = eval(cell["center"])
        ax.text(center[0], center[1], cell["cellName"], fontsize=8, ha='center', bbox=dict(facecolor='white', alpha=0.7))
        for pin in cell["pinList"]:
            pc = eval(pin["center"])
            ax.plot(pc[0], pc[1], 'o', c=c)
    ax.set_xlim(0,256)
    ax.set_ylim(0,256)
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

# 训练结束后可视化结果
plot_pcb(env)

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
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Feature extractors ────────────────────────────────────────────────────────
class SimpleGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, hidden_feats)
        self.fc2 = nn.Linear(hidden_feats, out_feats)
    def forward(self, node_feats, edges):
        h = F.relu(self.fc1(node_feats))
        h = self.fc2(h)
        return h.mean(dim=0)  # global mean

class SimpleCNN(nn.Module):
    def __init__(self, out_feats):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2), nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(16, out_feats)
    def forward(self, img):
        x = self.conv(img.unsqueeze(1))
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(0)

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, out_feats):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        encoder = nn.TransformerEncoderLayer(embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
        self.fc = nn.Linear(embed_dim, out_feats)
    def forward(self, seq):
        x = self.embed(seq)               # (B, L, E)
        h = self.transformer(x)           # (B, L, E)
        h = h.mean(dim=1)                 # (B, E)
        return self.fc(h).squeeze(0)      # (out_feats,)

# ─── Policy ───────────────────────────────────────────────────────────────────
class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dims):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(128, d) for d in action_dims])
    def forward(self, feat):
        h = self.fc(feat)
        return [head(h) for head in self.heads]

class PCBPolicy(nn.Module):
    def __init__(self, action_dims):
        super().__init__()
        self.gcn = SimpleGCN(3, 32, 32)
        self.cnn = SimpleCNN(32)
        self.trf = SimpleTransformer(2000, 32, 32)
        self.fuse = nn.Linear(96, 128)
        self.actor = PPOActor(128, action_dims)
    def forward(self, graph, image, sequence):
        g = self.gcn(graph['node_feats'], graph['edges'])
        i = self.cnn(image)
        s = self.trf(sequence)
        feat = torch.cat([g, i, s], dim=-1)
        feat = F.relu(self.fuse(feat))
        return self.actor(feat)

# ─── Utilities ────────────────────────────────────────────────────────────────
def build_graph_info(env):
    nodes = []
    for cell in env.cell_list:
        for pin in cell['pinList']:
            nodes.append((cell['cellName'], pin['pinName']))
    return {'nodes': nodes}

def get_pin_center(cell_list, cell_name, pin_name):
    cell = next(c for c in cell_list if c['cellName']==cell_name)
    pin  = next(p for p in cell['pinList'] if p['pinName']==pin_name)
    return np.array(eval(pin['center']))

def build_sequence_info(env):
    SOS, SOE, MOS, MOE = 0,1,2,3
    seq = [SOS]
    graph_info = build_graph_info(env)
    pk2i = {k:i for i,k in enumerate(graph_info['nodes'])}
    c2i = {cell['cellName']:10+idx for idx,cell in enumerate(env.cell_list)}
    for cell in env.cell_list:
        tokens = []
        cid = c2i[cell['cellName']]
        for net in env.net_list:
            inc = [p for p in net['pinList'] if p['cellName']==cell['cellName']]
            out = [p for p in net['pinList'] if p['cellName']!=cell['cellName']]
            for pi in inc:
                ki = (pi['cellName'],pi['pinName'])
                if ki not in pk2i: continue
                ni = pk2i[ki]
                for po in out:
                    ko = (po['cellName'],po['pinName'])
                    if ko not in pk2i: continue
                    no = pk2i[ko]
                    d = int(np.linalg.norm(get_pin_center(env.cell_list,*ki)-get_pin_center(env.cell_list,*ko)))
                    tokens += [ni+100, no+100, d]
        if tokens:
            seq += [MOS, cid] + tokens + [MOE]
    seq += [SOE]
    return np.array(seq, dtype=np.int32)

def update_features(env):
    # image
    img = np.zeros((256,256),dtype=np.float32)
    for idx,cell in enumerate(env.cell_list):
        contour = np.array(eval(cell['contour']))
        cv2.fillPoly(img, [contour], (idx+1)/len(env.cell_list))
    img = np.clip(img,0,1)
    image = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0)
    # sequence
    seq = build_sequence_info(env)
    sequence = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    return image, sequence

def select_action(policy, state):
    outs = policy(*state)
    acts, lps = [], []
    for out in outs:
        p = F.softmax(out, dim=-1)
        a = torch.multinomial(p,1).item()
        acts.append(a)
        lps.append(torch.log(p[a]))
    return tuple(acts), torch.stack(lps).sum()

def visualize_pcb(env):
    fig,ax = plt.subplots(figsize=(6,6))
    for cell in env.cell_list:
        cnt = np.array(eval(cell['contour']))
        xs,ys = cnt[:,0],cnt[:,1]
        ax.plot(np.append(xs,xs[0]), np.append(ys,ys[0]), 'k-')
        cx,cy = np.array(eval(cell['center']))
        ax.text(cx,cy,cell['cellName'],ha='center',va='center')
        for pin in cell['pinList']:
            px,py = np.array(eval(pin['center']))
            ax.plot(px,py,'ro',markersize=3)
    for net in env.net_list:
        coords=[]
        for p in net['pinList']:
            c=next(c for c in env.cell_list if c['cellName']==p['cellName'])
            pr=next(pi for pi in c['pinList'] if pi['pinName']==p['pinName'])
            coords.append(np.array(eval(pr['center'])))
        for i in range(len(coords)-1):
            ax.plot([coords[i][0],coords[i+1][0]],
                    [coords[i][1],coords[i+1][1]],'gray',lw=1)
    ax.set_xlim(0,255); ax.set_ylim(0,255); ax.set_aspect('equal')
    plt.show()

# ─── Main training ──────────────────────────────────────────────────────────

# load fused_state
with open('pcb_fused_rl_state.pkl','rb') as f:
    fused_state = pickle.load(f)['pcb_cells_1.json']
graph = {
    'node_feats': torch.tensor(fused_state['graph']['node_feats'],dtype=torch.float32,device=device),
    'edges':      torch.tensor(fused_state['graph']['edges'],     dtype=torch.long,   device=device),
}

env = PCBRLEnv('pcb_pre_jsons/pcb_cells_1.json')
# dynamic action dims
action_dims = [len(env.non_main), env.n_move_dir, env.n_rot]

policy = PCBPolicy(action_dims).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-4)
writer = SummaryWriter('runs/pcb_ppo_v2')

# initial state
image, sequence = update_features(env)
state = (graph, image, sequence)

epochs = 200
max_steps = 80
patience  = 10  # 连续变差次数阈值
best_return = -float('inf')
best_env_data = None

# 初始化
best_list = []  # 存放 (return, data_snapshot)
max_keep = 5
for ep in range(epochs):
    env.reset()
    image, sequence = update_features(env)
    state = (graph, image, sequence)

    log_probs = []
    rewards   = []
    prev_total = env._compute_reward()

    for step in range(max_steps):
        action, lp = select_action(policy, state)
        _, _, _, _ = env.step(action)

        # step-wise reward
        curr_total = env._compute_reward()
        r = curr_total - prev_total
        prev_total = curr_total

        log_probs.append(lp)
        rewards.append(r)

        # 提前终止判断略…

        image, sequence = update_features(env)
        state = (graph, image, sequence)

    # 本轮 return & update
    G    = sum(rewards)
    loss = -torch.stack(log_probs).mean() * G

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # —— 维护 best_list ——
    # 深拷贝 env.data（包含最新 cellList & netList）
    data_snapshot = copy.deepcopy(env.data)
    best_list.append((G, data_snapshot))
    # 按 return 倒序，保留 top 5
    best_list = sorted(best_list, key=lambda x: x[0], reverse=True)[:max_keep]

    writer.add_scalar('Return/ep', G, ep)
    writer.add_scalar('Loss/ep',   loss.item(), ep)
    if (ep+1) % 50 == 0:
        print(f"[Epoch {ep+1}/{epochs}] Return={G:.2f}  Loss={loss.item():.4f}  Top1={best_list[0][0]:.2f}")

writer.close()
out = {}
for rank, (ret, data_snap) in enumerate(best_list, start=1):
    out[f"rank{rank}_return_{ret:.2f}"] = data_snap

with open("best5_envs.pkl", "wb") as f:
    pickle.dump(out, f)
print("Saved top5 env snapshots to best5_envs.pkl")

# （可选）可视化 top1：
top1_data = best_list[0][1]
env.data     = top1_data
env.cell_list = env.data["cellList"]
env.net_list  = env.data["netList"]
visualize_pcb(env)

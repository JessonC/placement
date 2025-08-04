import json
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical
from torchvision import models

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
from pcb_rl_env_final import PCBRLEnv  # 确保使用最新环境

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1) Graph Encoder (GAT)
class NodeEncoder(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats)
        )
    def forward(self, x):
        # x: [node_feat_dim]
        return self.fc(x)  # [out_feats]

class GraphEncoder(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_feats, hid_feats, heads=heads, concat=True)
        self.conv2 = GATConv(hid_feats * heads, out_feats, heads=1, concat=False)

    def forward(self, x, edge_index, edge_attr=None):
        h = F.elu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h.mean(dim=0)  # 全局平均池化

# 2) Image Encoder (ResNet18 without pretrained weights)
class ImageEncoder(nn.Module):
    def __init__(self, out_feats):
        super().__init__()
        # 不加载预训练权重
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, out_feats)

    def forward(self, img):
        # img: [B,3,256,256]
        return self.backbone(img)  # [B, out_feats]

# 3) Policy Network (Actor-Critic)
class GATResNetPolicy(nn.Module):
    def __init__(self, node_feat_dim, graph_hid, img_hid, fusion_dim):
        super().__init__()
        # 图编码
        self.graph_enc = GraphEncoder(node_feat_dim, graph_hid, fusion_dim)
        # 图像编码
        self.img_enc   = ImageEncoder(img_hid)
        # 节点编码
        self.node_enc  = NodeEncoder(node_feat_dim, fusion_dim)
        # 融合：fusion_dim(graph) + img_hid + fusion_dim(node)
        self.fusion    = nn.Linear(fusion_dim + img_hid + fusion_dim, fusion_dim)
        self.actor     = nn.Linear(fusion_dim, 1)
        self.critic    = nn.Linear(fusion_dim, 1)

    def forward(self, graph_data, image, node_feat, action_count):
        # graph_data.x: [N, node_feat_dim], we already passed x
        g_feat = self.graph_enc(graph_data.x, graph_data.edge_index)    # [fusion_dim]
        i_feat = self.img_enc(image).squeeze(0)                          # [img_hid]
        n_feat = self.node_enc(node_feat)                               # [fusion_dim]
        # 三路拼接
        fused = torch.cat([g_feat, i_feat, n_feat], dim=-1)             # [fusion_dim+img_hid+fusion_dim]
        fused = F.relu(self.fusion(fused))                              # [fusion_dim]
        logit0 = self.actor(fused)                                      # [1]
        logits = logit0.repeat(action_count)                            # [action_count]
        value  = self.critic(fused)                                     # [1]
        return logits, value


# 4) State Extraction Helpers
def extract_graph(env: PCBRLEnv):
    N = env.n_cells
    feats = []
    for idx in range(N):
        # bounding box size
        contour = np.array(json.loads(env.cells[idx]['contour']), int)
        w = (contour[:,0].max() - contour[:,0].min())/256
        h = (contour[:,1].max() - contour[:,1].min())/256
        cx, cy = env.cell_centers[idx] / 256
        placed = 1.0 if idx < env.step_idx else -1.0
        feats.append([w,h,cx,cy,placed])
    x = torch.tensor(feats, device=device, dtype=torch.float)

    rows, cols, edge_feats = [], [], []
    for i in range(N):
        for j in range(N):
            if i==j: continue
            rows.append(i); cols.append(j)
            pi = env.cell_centers[i].astype(float)
            pj = env.cell_centers[j].astype(float)
            d  = np.linalg.norm(pi-pj)/(256*np.sqrt(2))
            edge_feats.append([d])
    edge_index = torch.tensor([rows,cols], device=device, dtype=torch.long)
    edge_attr  = torch.tensor(edge_feats, device=device, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def render_image(env: PCBRLEnv):
    canvas = np.zeros((256,256,3),dtype=np.uint8)
    cmap = matplotlib.colormaps['tab20']
    for idx,c in enumerate(env.cells):
        pts = np.array(json.loads(c['contour']),int)
        color = (np.array(cmap(idx)[:3])*255).astype(np.uint8).tolist()
        cv2.fillPoly(canvas, [pts], color)
    for net in env.nets:
        pts=[]
        for p in net['pinList']:
            cname,pname=p['cellName'],p['pinName']
            for cell in env.cells:
                if cell['cellName']==cname:
                    for pin in cell['pinList']:
                        if pin['pinName']==pname:
                            pts.append(tuple(json.loads(pin['center'])))
                            break
                    break
        if len(pts)>1:
            cv2.polylines(canvas, [np.array(pts)], False, (255,255,255),1)
    img = torch.tensor(canvas/255.0,device=device,dtype=torch.float)
    img = img.permute(2,0,1).unsqueeze(0)
    return img

# 5) Training Loop (REINFORCE + baseline)
step_size    = 2
deltas       = list(range(-128, 129, step_size))
rotations    = [0,1,2,3]  # 分别对应 0°/90°/180°/270°
action_space = [(dx, dy, r) for r in rotations for dx in deltas for dy in deltas]
action_count = len(action_space)

GAMMA       = 0.99
GAE_LAMBDA  = 0.95
CLIP_EPS    = 0.2
PPO_EPOCHS  = 4
ENT_COEF    = 0.01
CRITIC_COEF = 0.5

def train():
    env = PCBRLEnv('pcb_pre_jsons/pcb_cells_1.json', gamma=1.0)
    policy = GATResNetPolicy(node_feat_dim=5, graph_hid=32, img_hid=64, fusion_dim=128).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    for ep in range(1, 201):
        # ---------- 1. 采集完整一条轨迹 ----------
        graph_xs, edge_idxs, edge_attrs = [], [], []
        images, node_feats = [], []
        action_idxs, old_logps, values = [], [], []
        rewards, dones = [], []

        env.reset(); done=False
        step = 0
        while not done:
            # extract state
            gdata = extract_graph(env)
            img   = render_image(env)
            idx   = env.step_idx
            node_feat = gdata.x[idx]

            # forward policy
            logits, value = policy(gdata, img, node_feat, action_count)
            probs  = F.softmax(logits, dim=-1)
            dist   = Categorical(probs)
            a_idx  = dist.sample()

            # execute
            _, r, done, _ = env.step(action_space[a_idx.item()])

            # store rollout data
            graph_xs.append(gdata.x.clone())
            edge_idxs.append(gdata.edge_index.clone())
            edge_attrs.append(gdata.edge_attr.clone())
            images.append(img.clone())
            node_feats.append(node_feat.clone())

            action_idxs.append(a_idx)
            old_logps.append(dist.log_prob(a_idx))
            values.append(value.squeeze())
            rewards.append(r)
            dones.append(done)

            step += 1

        # 转为 tensor
        old_logps = torch.stack(old_logps)
        values    = torch.stack(values)
        rewards   = torch.tensor(rewards, device=device, dtype=torch.float)
        dones     = torch.tensor(dones, device=device, dtype=torch.float)

        # ---------- 2. 用 GAE 计算 advantage 和 returns ----------
        advantages = torch.zeros_like(rewards, device=device)
        last_gae = 0
        # 在最后一步之后的 value = 0
        next_value = 0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + GAMMA * next_value * mask - values[t]
            advantages[t] = last_gae = delta + GAMMA * GAE_LAMBDA * mask * last_gae
            next_value = values[t]
        returns = advantages + values

        # ---------- 3. 多轮 PPO 更新 ----------
        for _ in range(PPO_EPOCHS):
            # 对同一条轨迹逐步计算新的 logπ、value、entropy
            new_logps = []
            new_values = []
            entropies  = []

            for t in range(len(rewards)):
                # 重构 state
                gdata = Data(x=graph_xs[t], edge_index=edge_idxs[t], edge_attr=edge_attrs[t])
                img   = images[t]
                nf    = node_feats[t]
                # 前向
                logits, value_pred = policy(gdata, img, nf, action_count)
                dist = Categorical(F.softmax(logits, dim=-1))

                new_logp = dist.log_prob(action_idxs[t])
                new_logps.append(new_logp)
                new_values.append(value_pred.squeeze())
                entropies.append(dist.entropy())

            new_logps  = torch.stack(new_logps)
            new_values = torch.stack(new_values)
            entropies  = torch.stack(entropies)

            # 计算比率与裁剪后的目标
            ratio = (new_logps - old_logps).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0-CLIP_EPS, 1.0+CLIP_EPS) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - new_values).pow(2).mean()
            entropy_loss = -entropies.mean()

            loss = actor_loss + CRITIC_COEF * critic_loss + ENT_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        print(f"[Ep {ep:03d}] Return={returns[0].item():.4f}  ActorLoss={actor_loss.item():.4f}  CriticLoss={critic_loss.item():.4f}")

        if ep % 50 == 0:
            env.visualize()

if __name__=='__main__':
    train()

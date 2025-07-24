"""ppo_tensorboard.py (improved‑v2)
============================================================
• 兼容无 `matplotlib` 环境：`try‒except` 检测后再绘图；日志仍写入 TensorBoard。
• 其余改动保持上一版本（mini‑batch PPO、几何缓存、TensorBoard 等）。
"""

from __future__ import annotations
import json, random, gc, copy, math, os
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from shapely.geometry import Polygon
from tqdm import tqdm

# ─── 可选 matplotlib ─────────────────────────────
try:
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_PLT = True
except ModuleNotFoundError:
    _HAS_PLT = False
    print("[INFO] matplotlib not found ‒ skipping local plots; use TensorBoard instead.")

# ─── 项目本地依赖 ───────────────────────────────
from Placement_Env import PlacementEnv        # 环境类
from model        import ActorCritic          # GNN‑ActorCritic 网络

# ─── 设备 & 超参 ────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPISODES   = 500
GAMMA          = 0.95
LR             = 3e-4
LAMBDA_GAE     = 0.95
EPOCHS_PPO     = 8
CLIP_EPS       = 0.2
ENTROPY_COEFF  = 0.01
BATCH_CAPACITY = 4096
MINIBATCH_SIZE = 512
SAVE_PATH      = 'ppo_model_best.pth'
TBOARD_DIR     = 'runs/ppo_placement'
KEYWORDS       = ['IC','SMD','TP','MC','NPO','RES','DCDC','电感','Inductor','DIO']

# ─── 工具函数 ───────────────────────────────────
def moving_average(arr:List[float], k:int=10):
    if len(arr) < k:
        return arr
    arr_np = np.asarray(arr)
    ma = np.convolve(arr_np, np.ones(k)/k, mode='valid')
    return np.concatenate([arr_np[:k-1], ma])

# ─── PPO Agent ──────────────────────────────────
class PPO:
    def __init__(self, n_actions:int, lmbda:float, clip_eps:float, gamma:float, device:torch.device = DEVICE):
        self.device      = device
        self.n_actions   = n_actions
        self.clip_eps    = clip_eps
        self.gamma       = gamma
        self.gae_lambda  = lmbda

        self.net = ActorCritic(in_dim=128, out_dim=n_actions).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=LR)

        self.geom_cache: Dict[str, List[float]] = {}
        self.last_metrics: Dict[str, float] | None = None

    # ── 动作选择 ──
    @torch.no_grad()
    def take_action(self, env:PlacementEnv, state:Data):
        mask = torch.zeros(self.n_actions, dtype=torch.float32, device=self.device)
        for y,x,r in env.valid_grid_ids:
            idx = r*env.area_width*env.area_height + y*env.area_width + x
            mask[idx] = 1
        dist, value = self.net(state, mask=mask)
        action = dist.sample()
        logp   = dist.log_prob(action)
        return action.item(), value.squeeze(0).detach(), logp.detach()

    # ── 学习 ──
    def learn(self, batch:list[Tuple]):
        S,A,old_logp,V,R,S_,D,GO = zip(*batch)
        states   = self._merge_states(S)
        actions  = torch.tensor(A, dtype=torch.long, device=self.device).view(-1,1)
        old_logp = torch.stack(old_logp).unsqueeze(-1)

        rewards  = self._normalize(np.array(R, dtype=np.float32))
        returns, adv = self._calc_returns_adv(rewards, V, D, GO)
        returns  = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(-1)
        adv      = torch.tensor(adv, dtype=torch.float32, device=self.device).unsqueeze(-1)

        idx = np.arange(len(actions))
        for _ in range(EPOCHS_PPO):
            np.random.shuffle(idx)
            for start in range(0, len(idx), MINIBATCH_SIZE):
                mb = idx[start:start+MINIBATCH_SIZE]
                dist, values = self.net(states[mb])
                logp = dist.log_prob(actions[mb].squeeze(-1)).unsqueeze(-1)
                ratio = torch.exp(logp - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv[mb]
                actor_l  = -torch.min(surr1, surr2).mean()
                critic_l = nn.MSELoss()(values, returns[mb])
                entropy  = dist.entropy().mean()
                loss = actor_l + 0.5*critic_l - ENTROPY_COEFF*entropy
                self.opt.zero_grad(); loss.backward(); self.opt.step()

        self.last_metrics = {
            'actor_loss': actor_l.item(),
            'critic_loss': critic_l.item(),
            'entropy':     entropy.item(),
            'advantage':   adv.mean().item(),
        }
        batch.clear()

    # ────────────────
    def _calc_returns_adv(self, rew, vals, dones, overs):
        adv, ret, gae, disc = [], [], 0., 0.
        vals = list(vals)+[0]
        for t in reversed(range(len(rew))):
            disc = rew[t] + self.gamma*disc*(1-dones[t])*(1-overs[t])
            ret.insert(0, disc)
            delta = rew[t] + self.gamma*vals[t+1]*(1-dones[t])*(1-overs[t]) - vals[t]
            gae   = delta + self.gamma*self.gae_lambda*(1-dones[t])*(1-overs[t])*gae
            adv.insert(0, gae)
        return ret, adv

    def _normalize(self, x:np.ndarray, eps=1e-8):
        return (x - x.mean()) / (x.std()+eps)

    def _merge_states(self, graphs:List[Data]):
        xs, ei, batches, off = [], [], [], 0
        for b,g in enumerate(graphs):
            xs.append(g.x)
            ei.append(g.edge_index+off)
            batches.append(torch.full((g.x.size(0),), b, dtype=torch.long, device=self.device))
            off += g.x.size(0)
        return Data(x=torch.cat(xs), edge_index=torch.cat(ei,1), batch=torch.cat(batches))

    # 几何特征缓存
    def get_geom_feat(self, comp:str, shape:List[List[float]]):
        if comp in self.geom_cache:
            return self.geom_cache[comp]
        poly = Polygon(shape)
        peri, area = poly.length, poly.area
        xs, ys = zip(*shape)
        w, h = max(xs)-min(xs), max(ys)-min(ys)
        ratio = w/(h+1e-6)
        roundness = (4*math.pi*area)/(peri**2+1e-6)
        feat = [peri, area, w, h, ratio, roundness]
        self.geom_cache[comp] = feat
        return feat

def load_module_data(filename):
    with open(filename, 'r') as f:
        module_data = json.load(f)
    return module_data
# ─── 数据与环境 (get_data / build_graph) 保持原实现 ───
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

    comps_type = extract_keywords(comps_comment, KEYWORDS)
    comp_coords = np.array(comp_coords)

    return area_shape, comps_shape, comps_name, comp_coords, pins, comps_type
# ─── 训练主循环 ─────────────────────────────────────
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
def train(env:PlacementEnv):
    writer = SummaryWriter(TBOARD_DIR)
    agent  = PPO(env.action_space.n*4, LAMBDA_GAE, CLIP_EPS, GAMMA)
    best_ret, buf, returns = -1e9, [], []

    for ep in tqdm(range(NUM_EPISODES), unit='ep'):
        _grid,_,_ = env.reset()
        state = agent._merge_states([Data()])  # 空图
        ep_ret, steps, done, game_over = 0.,0,False,False
        while not (done or game_over):
            act,val,logp = agent.take_action(env,state)
            *_rest, reward, done, game_over = env.step(act,steps)
            next_state = state if (done or game_over) else agent._merge_states([Data()])
            buf.append((state,act,logp,val.item(),reward,next_state,done,game_over))
            state, ep_ret, steps = next_state, ep_ret+reward, steps+1
            if len(buf)>=BATCH_CAPACITY:
                agent.learn(buf)
        returns.append(ep_ret)
        writer.add_scalar('Reward/ep', ep_ret, ep)
        if agent.last_metrics:
            for k,v in agent.last_metrics.items():
                writer.add_scalar(f'{k}', v, ep)

        if ep_ret>best_ret:
            best_ret=ep_ret
            torch.save({'net':agent.net.state_dict(),'ret':best_ret}, SAVE_PATH)

    writer.close()
    if _HAS_PLT:
        plt.plot(returns); plt.title('Episode Rewards'); plt.show()

# ─── main ───────────────────────────────────────────
if __name__ == '__main__':
    area_shape, comps_shape, comps_name, comp_coords, pins, comps_type = get_data()
    edges_id, comp2id, net2comps, comp_pins = build_graph(comps_name, pins)
    env = PlacementEnv(area_shape, comps_shape, comps_name, comp_coords,
                       edges_id, comp2id, comps_type, net2comps, comp_pins, pins)
    train(env)

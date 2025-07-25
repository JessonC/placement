# random_greedy_with_mask.py

import random
import numpy as np
import json
from copy import deepcopy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def _safe_eval(x):
    return eval(x) if isinstance(x, str) else x

def transform_shape(pts, c_old, c_new, dθ):
    arr = np.array(pts, float)
    arr -= c_old
    θ = np.deg2rad(dθ)
    R = np.array([[np.cos(θ), -np.sin(θ)],
                  [np.sin(θ),  np.cos(θ)]])
    arr = arr @ R.T
    arr += c_new
    return arr.astype(int).tolist()

class PCBRLEnv:
    """
    顺序放置环境，新增 get_action_mask 方法，保证
    放置新器件时不与已放置器件重叠或越界。
    """
    def __init__(self, pcb_json_path, gamma=1.0):
        with open(pcb_json_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        self.gamma = gamma
        self.grid_size = 256
        self.reset()

    def reset(self):
        d = deepcopy(self.raw_data)
        self.cell_list = d["cellList"]
        self.net_list  = d["netList"]
        self.n_dev     = len(self.cell_list)
        self.step_idx  = 0
        # 主芯片固定放置在 index=0
        self._refresh_cache()
        return self._get_state()

    def _refresh_cache(self):
        self.pos = []
        self.rot = []
        for cell in self.cell_list:
            c = np.array(_safe_eval(cell["center"]), int)
            r = int(cell.get("rotation","0"))
            self.pos.append(c)
            self.rot.append(r)

    def _get_state(self):
        # state 只返回当前 step_idx
        return self.step_idx

    def get_action_mask(self, rel_range=128, step=32):
        """
        只为当前器件(step_idx)生成合法 (dx,dy,rot_idx) 列表：
        - 放置后不越界、不与任何已放置(0..step_idx-1)器件重叠。
        """
        if self.step_idx >= self.n_dev:
            return []

        idx = self.step_idx
        main_c = self.pos[0]  # 主芯片坐标
        mask = []
        # dx, dy 枚举
        rng = range(-rel_range, rel_range+1, step)
        for rot_idx in range(4):
            for dx in rng:
                for dy in rng:
                    new_c = main_c + np.array([dx,dy])
                    if not (0 <= new_c[0] < self.grid_size and 0 <= new_c[1] < self.grid_size):
                        continue
                    if self._check_valid(idx, new_c, rot_idx):
                        mask.append((dx, dy, rot_idx))
        return mask

    def _check_valid(self, idx, new_c, rot_idx):
        """
        检查将 cell_list[idx] 放到 new_c 并旋转 rot_idx*90°
        是否与已放置的 cell_list[0..step_idx-1] 重叠或越界。
        """
        cell = self.cell_list[idx]
        c_old = self.pos[idx]
        r_old = self.rot[idx]
        contour0 = _safe_eval(cell["contour"])
        # 计算新轮廓
        new_pts = transform_shape(contour0, c_old, new_c, rot_idx*90 - r_old)
        arr = np.array(new_pts, int)
        # 越界检查
        if (arr < 0).any() or (arr[:,0] >= self.grid_size).any() or (arr[:,1] >= self.grid_size).any():
            return False
        # 与已放置器件 AABB 重叠检查
        x0,x1 = arr[:,0].min(), arr[:,0].max()
        y0,y1 = arr[:,1].min(), arr[:,1].max()
        for j in range(self.step_idx):
            op = np.array(_safe_eval(self.cell_list[j]["contour"]), int)
            ox0,ox1 = op[:,0].min(), op[:,0].max()
            oy0,oy1 = op[:,1].min(), op[:,1].max()
            if not (x1 < ox0 or ox1 < x0 or y1 < oy0 or oy1 < y0):
                return False
        return True

    def step(self, action):
        dx, dy, rot_idx = action
        idx = self.step_idx
        cell = self.cell_list[idx]

        main_c = self.pos[0]
        new_c  = main_c + np.array([dx, dy])
        # 更新 cell
        cell["center"]   = str(new_c.tolist())
        cell["rotation"] = str(rot_idx * 90)

        # 更新轮廓 & pin
        contour0 = _safe_eval(cell["contour"])
        old_c, old_r = self.pos[idx], self.rot[idx]
        new_contour = transform_shape(contour0, old_c, new_c, rot_idx*90 - old_r)
        cell["contour"] = str(new_contour)
        for pin in cell["pinList"]:
            p0 = _safe_eval(pin.get("center", pin["contour"]))
            new_pc = transform_shape([p0], old_c, new_c, rot_idx*90 - old_r)[0]
            pin["center"]  = str(new_pc)
            pin["contour"] = str(transform_shape(_safe_eval(pin["contour"]), old_c, new_c, rot_idx*90 - old_r))

        # 刷新缓存，步进
        self._refresh_cache()
        self.step_idx += 1

        # 只有最后一步给 reward
        if self.step_idx < self.n_dev:
            return self._get_state(), 0.0, False, {}
        wl = self._compute_wl(self.gamma)
        reward = 1.0 / (wl + 1e-6)
        return self._get_state(), reward, True, {}

    def _compute_wl(self, gamma):
        total_wl = 0.0
        for net in self.net_list:
            xs, ys = [], []
            for p in net["pinList"]:
                # 找到 pin 的 center
                for cell in self.cell_list:
                    if cell["cellName"] == p["cellName"]:
                        for pp in cell["pinList"]:
                            if pp["pinName"] == p["pinName"]:
                                xk, yk = map(float, _safe_eval(pp["center"]))
                                xs.append(xk); ys.append(yk)
                                break
                        break
            if not xs:
                continue
            xs, ys = np.array(xs), np.array(ys)
            pos_x = np.log(np.exp(xs/gamma).sum() + 1e-12)
            neg_x = np.log(np.exp(-xs/gamma).sum() + 1e-12)
            pos_y = np.log(np.exp(ys/gamma).sum() + 1e-12)
            neg_y = np.log(np.exp(-ys/gamma).sum() + 1e-12)
            wl_e = gamma * (pos_x + neg_x + pos_y + neg_y)
            total_wl += wl_e
        return float(total_wl)

    def visualize(self):
        fig, ax = plt.subplots(figsize=(8,8))
        for cell in self.cell_list:
            pts = np.array(_safe_eval(cell["contour"]), int)
            center = np.array(_safe_eval(cell["center"]), int)
            real = pts + center
            poly = plt.Polygon(real, fill=None, edgecolor='r')
            ax.add_patch(poly)
            ax.text(*center, cell["cellName"], ha='center')
        # 画连线
        for net in self.net_list:
            coords = []
            for p in net["pinList"]:
                for cell in self.cell_list:
                    if cell["cellName"] == p["cellName"]:
                        for pp in cell["pinList"]:
                            if pp["pinName"] == p["pinName"]:
                                coords.append(np.array(_safe_eval(pp["center"]),int))
                                break
                        break
            if len(coords)>=2:
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                ax.plot(xs, ys, '--', color='gray', alpha=0.7)
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        plt.title("PCB Layout with Wires")
        plt.show()

# ----- 搜索算法 -----

def evaluate_sequence(pcb_json, seq, gamma=1.0):
    env = PCBRLEnv(pcb_json, gamma)
    env.reset()
    r = 0.0
    for act in seq:
        _, r, done, _ = env.step(act)
    return r

def random_rollout(pcb_json, rel_range=128, step=32, gamma=1.0):
    env = PCBRLEnv(pcb_json, gamma)
    env.reset()
    seq = []
    n = env.n_dev
    for _ in range(n):
        mask = env.get_action_mask(rel_range, step)
        act = random.choice(mask) if mask else (0,0,0)
        seq.append(act)
        env.step(act)
    return seq, evaluate_sequence(pcb_json, seq, gamma)

def greedy_improve(pcb_json, base_seq, rel_range=128, step=32,
                   gamma=1.0, n_cand=10):
    best = list(base_seq)
    best_r = evaluate_sequence(pcb_json, best, gamma)
    for i in range(len(best)):
        # replay up to i
        env = PCBRLEnv(pcb_json, gamma); env.reset()
        for j in range(i):
            env.step(best[j])
        # mask for position i
        mask = env.get_action_mask(rel_range, step)
        local_best_r = best_r
        local_best_act = best[i]
        for _ in range(n_cand):
            act = random.choice(mask) if mask else (0,0,0)
            cand = best.copy(); cand[i] = act
            r = evaluate_sequence(pcb_json, cand, gamma)
            if r > local_best_r:
                local_best_r = r
                local_best_act = act
        best[i] = local_best_act
        best_r = local_best_r
    return best, best_r

def random_greedy_search(pcb_json, n_rollouts=20,
                         rel_range=128, step=32, gamma=1.0):
    best_seq, best_r = None, -1.0
    for _ in range(n_rollouts):
        seq, r = random_rollout(pcb_json, rel_range, step, gamma)
        if r > best_r:
            best_seq, best_r = seq, r
    print(f"[Random]   best reward = {best_r:.6f}")
    imp_seq, imp_r = greedy_improve(pcb_json, best_seq,
                                    rel_range, step, gamma, n_cand=10)
    print(f"[Greedy]   improved reward = {imp_r:.6f}")
    return best_seq, best_r, imp_seq, imp_r

def visualize_with_mask(pcb_json, seq, gamma=1.0):
    env = PCBRLEnv(pcb_json, gamma)
    env.reset()
    for act in seq:
        env.step(act)
    env.visualize()

if __name__ == "__main__":
    pcb_json = "pcb_pre_jsons/pcb_cells_1.json"
    best_seq, best_r, imp_seq, imp_r = random_greedy_search(
        pcb_json, n_rollouts=50, rel_range=128, step=32, gamma=1.0
    )
    print("Best random seq:", best_seq)
    print("Improved seq:",    imp_seq)
    visualize_with_mask(pcb_json, imp_seq, gamma=1.0)

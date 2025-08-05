# pcb_rl_env_two_layers.py

import json
import numpy as np
import cv2
from copy import deepcopy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def safe_eval(x):
    return json.loads(x) if isinstance(x, str) else x


class PCBRLEnv:
    def __init__(self,
                 pcb_json_path: str,
                 gamma: float = 1.0,
                 grid_size: int = 256,
                 nc_weight: float = 1.0,
                 density_weight: float = 1.0,
                 density_grid_N: int = 16,
                 step_size: int = 32,
                 layer_penalty: float = 1.25,
                 fd_weight: float = 1.0):
        with open(pcb_json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        self.gamma          = gamma
        self.grid_size      = grid_size
        self.nc_weight      = nc_weight
        self.density_weight = density_weight
        self.density_grid_N = density_grid_N
        self.step_size      = step_size
        self.layer_penalty  = layer_penalty
        self.fd_weight      = fd_weight
        self.reset()

    def reset(self):
        self.data           = deepcopy(self.raw_data)
        self.cells          = self.data['cellList']
        self.nets           = self.data['netList']
        self.n_cells        = len(self.cells)
        self.step_idx       = 1  # 主芯片索引0固定不动

        # 两层占位矩阵
        self.occupancy = {
            0: np.zeros((self.grid_size, self.grid_size), dtype=np.int8),
            1: np.zeros((self.grid_size, self.grid_size), dtype=np.int8),
        }
        # 缓存
        self.cell_centers   = []
        self.cell_rotations = []
        self.rel_contours   = []
        self.pin_rel_pos    = {}
        self.cell_layers    = [0]*self.n_cells

        for idx, cell in enumerate(self.cells):
            ctr     = np.array(safe_eval(cell['center']), dtype=int)
            rot     = int(cell.get('rotation', '0'))
            contour = np.array(safe_eval(cell['contour']), dtype=int)
            rel_ct  = contour - ctr
            self.cell_centers.append(ctr)
            self.cell_rotations.append(rot)
            self.rel_contours.append(rel_ct)
            for pin in cell['pinList']:
                pc = np.array(safe_eval(pin['center']), dtype=int)
                self.pin_rel_pos[(cell['cellName'], pin['pinName'])] = pc - ctr

        # 主芯片占 Top(0)
        self._update_occupancy(0, self.cell_centers[0], self.cell_rotations[0], layer=0)
        return self.step_idx

    def _rotate_points(self, pts: np.ndarray, angle: float) -> np.ndarray:
        rad = np.deg2rad(angle)
        R   = np.array([[np.cos(rad), -np.sin(rad)],
                        [np.sin(rad),  np.cos(rad)]])
        return pts @ R.T

    def _update_occupancy(self, idx: int, center: np.ndarray,
                          rotation: float, layer: int):
        pts = (self._rotate_points(self.rel_contours[idx], rotation) + center).astype(np.int32)
        cv2.fillPoly(self.occupancy[layer], [pts], -1)

    def _check_overlap(self, idx: int, center: np.ndarray,
                       rotation: float, layer: int) -> bool:
        pts = (self._rotate_points(self.rel_contours[idx], rotation) + center).astype(np.int32)
        if np.any(pts < 0) or np.any(pts >= self.grid_size):
            return True
        tmp = np.zeros_like(self.occupancy[layer], dtype=np.uint8)
        cv2.fillPoly(tmp, [pts], 1)
        return np.any((tmp == 1) & (self.occupancy[layer] == -1))

    def _find_non_overlap(self, idx, base_ctr, rotation, layer):
        """
        Direct-Forcing: 从 base_ctr 开始，尝试四个方向的 step_size 微移，
        找到第一个不重叠的位置；若都重叠，则返回 base_ctr 本身。
        """
        if not self._check_overlap(idx, base_ctr, rotation, layer):
            return base_ctr
        for dx, dy in [(self.step_size,0),(-self.step_size,0),
                       (0,self.step_size),(0,-self.step_size)]:
            trial = base_ctr + np.array([dx,dy])
            if not self._check_overlap(idx, trial, rotation, layer):
                return trial
        return base_ctr

    def _local_wirelength(self, idx: int, center: np.ndarray,
                          rotation: float) -> float:
        lw = 0.0
        name = self.cells[idx]['cellName']
        for net in self.nets:
            pins_my = [p for p in net['pinList'] if p['cellName']==name]
            others  = [p for p in net['pinList'] if p['cellName']!=name]
            for p in pins_my:
                rel = self.pin_rel_pos[(p['cellName'],p['pinName'])]
                pa  = center + self._rotate_points(rel, rotation)
                for q in others:
                    j = next(i for i,c in enumerate(self.cells) if c['cellName']==q['cellName'])
                    ctr2, rot2 = self.cell_centers[j], self.cell_rotations[j]
                    rel2 = self.pin_rel_pos[(q['cellName'],q['pinName'])]
                    pb   = ctr2 + self._rotate_points(rel2, rot2)
                    lw  += np.linalg.norm(pa-pb)
        return lw

    def step(self, action):
        dx, dy = action
        idx     = self.step_idx
        main_ctr= self.cell_centers[0]
        rot     = 0  # 旋转交给后处理
        base_ctr= main_ctr + np.array([dx,dy])

        # 如果 Top 无重叠，直接放 Top
        if not self._check_overlap(idx, base_ctr, rot, layer=0):
            chosen_layer, final_ctr = 0, base_ctr
        else:
            # 1) FD 微移 on Top
            top_fd_ctr = self._find_non_overlap(idx, base_ctr, rot, 0)
            lw_fd      = self._local_wirelength(idx, top_fd_ctr, rot)

            # 2) Layer-Flip on Bottom
            #    先寻找 Bottom 上的不重叠位置
            bot_ctr0 = self._find_non_overlap(idx, base_ctr, rot, 1)
            lw_bot   = self._local_wirelength(idx, bot_ctr0, rot)
            cost_flip= lw_bot * self.layer_penalty
            cost_fd  = lw_fd

            if cost_flip < cost_fd:
                chosen_layer, final_ctr = 1, bot_ctr0
            else:
                chosen_layer, final_ctr = 0, top_fd_ctr

        # 同步状态
        self.cell_layers[idx]    = chosen_layer
        self.cell_centers[idx]   = final_ctr
        self.cell_rotations[idx] = rot
        self._update_occupancy(idx, final_ctr, rot, layer=chosen_layer)

        # 前进
        self.step_idx += 1
        done = (self.step_idx >= self.n_cells)

        reward = 0.0
        if done:
            wl  = self._compute_wirelength()
            nc  = self._compute_net_crossing()
            dn  = self._compute_density(self.density_grid_N)
            cost= wl + self.nc_weight*nc + self.density_weight*dn
            reward = 1000.0 / (cost + 1e-6)

        return self.step_idx, reward, done, {}

    def _compute_wirelength(self) -> float:
        total = 0.0
        for net in self.nets:
            xs, ys = [], []
            for p in net['pinList']:
                cname, pname = p['cellName'], p['pinName']
                i = next(i for i,c in enumerate(self.cells) if c['cellName']==cname)
                ctr, rot = self.cell_centers[i], self.cell_rotations[i]
                rel      = self.pin_rel_pos[(cname,pname)]
                pa       = ctr + self._rotate_points(rel, rot)
                xs.append(pa[0]); ys.append(pa[1])
            if xs:
                xs, ys = np.array(xs), np.array(ys)
                total += (xs.max()-xs.min()) + (ys.max()-ys.min())
        return float(total)

    def _compute_net_crossing(self) -> int:
        segs = []
        for net in self.nets:
            pins = []
            for p in net['pinList']:
                cname, pname = p['cellName'], p['pinName']
                i = next(i for i,c in enumerate(self.cells) if c['cellName']==cname)
                ctr, rot = self.cell_centers[i], self.cell_rotations[i]
                rel      = self.pin_rel_pos[(cname,pname)]
                pa       = ctr + self._rotate_points(rel, rot)
                pins.append(tuple(pa.tolist()))
            if len(pins)<2: continue
            src = pins[0]
            for sink in pins[1:]:
                segs.append((src, sink))
        cnt = 0
        for i in range(len(segs)):
            for j in range(i+1,len(segs)):
                if self._segments_cross(segs[i], segs[j]):
                    cnt += 1
        return cnt

    def _segments_cross(self, s1, s2) -> bool:
        def ccw(A,B,C): return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        A,B = s1; C,D = s2
        return (ccw(A,C,D)!=ccw(B,C,D)) and (ccw(A,B,C)!=ccw(A,B,D))

    def _compute_density(self, N: int) -> float:
        grid = np.zeros((N,N), dtype=float)
        unit = self.grid_size//N
        mask = np.zeros((self.grid_size,self.grid_size), dtype=np.uint8)
        # 合并所有已放器件
        for idx in range(self.n_cells):
            pts = (self._rotate_points(self.rel_contours[idx],
                                       self.cell_rotations[idx])
                   + self.cell_centers[idx]).astype(np.int32)
            tmp = np.zeros_like(mask)
            cv2.fillPoly(tmp, [pts], 1)
            mask |= tmp
        # 各小格计数
        for i in range(N):
            x0,x1 = i*unit,(i+1)*unit
            for j in range(N):
                y0,y1 = j*unit,(j+1)*unit
                grid[i,j] = mask[x0:x1,y0:y1].sum()
        mu = grid.mean()
        return float(((grid-mu)**2).mean())

    def visualize(self):
        fig, ax = plt.subplots(figsize=(8,8))
        cmap = plt.cm.get_cmap('tab20', self.n_cells)
        for idx, cell in enumerate(self.cells):
            pts = (self._rotate_points(self.rel_contours[idx],
                                       self.cell_rotations[idx])
                   + self.cell_centers[idx])
            layer = self.cell_layers[idx]
            ls = '-' if layer==0 else '--'
            ax.add_patch(plt.Polygon(pts, fill=None,
                                     edgecolor=cmap(idx),
                                     linestyle=ls))
            ax.text(*self.cell_centers[idx], cell['cellName'],
                    ha='center', color=cmap(idx))
        # 飞线
        for net in self.nets:
            path=[]
            for p in net['pinList']:
                cname,pname=p['cellName'],p['pinName']
                i = next(i for i,c in enumerate(self.cells) if c['cellName']==cname)
                ctr,rot=self.cell_centers[i],self.cell_rotations[i]
                rel = self.pin_rel_pos[(cname,pname)]
                path.append(ctr + self._rotate_points(rel,rot))
            path=np.array(path)
            if len(path)>1:
                ax.plot(path[:,0],path[:,1],'w-',linewidth=1)
        ax.set_xlim(0,self.grid_size)
        ax.set_ylim(0,self.grid_size)
        ax.set_aspect('equal')
        plt.title("PCB (solid=top, dashed=bottom)")
        plt.show()

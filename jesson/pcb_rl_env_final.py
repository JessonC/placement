import json
import numpy as np
import cv2
from copy import deepcopy
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
                 step: int = 32):
        with open(pcb_json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)

        self.gamma          = gamma
        self.grid_size      = grid_size
        self.nc_weight      = nc_weight
        self.density_weight = density_weight
        self.density_grid_N = density_grid_N


        self.reset()

    def reset(self):
        self.data = deepcopy(self.raw_data)
        self.cells = self.data['cellList']
        self.nets  = self.data['netList']

        # 从第1个（索引1）开始放，索引0主芯片已固定
        self.step_idx = 1
        self.n_cells  = len(self.cells)

        # 占位矩阵
        self.occupancy_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # 缓存
        self.cell_centers   = []
        self.cell_rotations = []
        self.rel_contours   = []
        self.pin_rel_pos    = {}

        for idx, cell in enumerate(self.cells):
            ctr    = np.array(safe_eval(cell['center']), dtype=int)
            rot    = int(cell.get('rotation', '0'))
            contour= np.array(safe_eval(cell['contour']), dtype=int)
            rel_ct = contour - ctr

            self.cell_centers.append(ctr)
            self.cell_rotations.append(rot)
            self.rel_contours.append(rel_ct)

            for pin in cell['pinList']:
                pc   = np.array(safe_eval(pin['center']), dtype=int)
                self.pin_rel_pos[(cell['cellName'], pin['pinName'])] = pc - ctr

        # 标记主芯片占位
        self._update_occupancy(0, self.cell_centers[0], self.cell_rotations[0])
        return self.step_idx

    def _rotate_points(self, pts: np.ndarray, angle: float) -> np.ndarray:
        rad = np.deg2rad(angle)
        R   = np.array([[np.cos(rad), -np.sin(rad)],
                        [np.sin(rad),  np.cos(rad)]])
        return pts @ R.T

    def _update_occupancy(self, idx: int, center: np.ndarray, rotation: float):
        pts = (self._rotate_points(self.rel_contours[idx], rotation) + center).astype(np.int32)
        cv2.fillPoly(self.occupancy_grid, [pts], -1)

    def _check_overlap(self, idx: int, center: np.ndarray, rotation: float) -> bool:
        pts = (self._rotate_points(self.rel_contours[idx], rotation) + center).astype(np.int32)
        # 越界
        if np.any(pts < 0) or np.any(pts >= self.grid_size):
            return True
        # 与已占位区域重叠
        tmp = np.zeros_like(self.occupancy_grid, dtype=np.uint8)
        cv2.fillPoly(tmp, [pts], 1)
        return np.any((tmp == 1) & (self.occupancy_grid == -1))

    def step(self, action):
        """
        action: (dx, dy, rot_idx)
        如果 overlap 立即 reward=-1 并保持在同一个 step_idx，不前进；
        否则放置、前进，最后一步完成后按综合成本给正 reward。
        """
        dx, dy, r_idx = action
        idx     = self.step_idx
        main_ctr= self.cell_centers[0]
        rot     = [0,90,180,270][r_idx]
        new_ctr = main_ctr + np.array([dx, dy])

        # 若非法——重叠或越界，立即惩罚
        if self._check_overlap(idx, new_ctr, rot):
            return idx, -1.0, False, {}

        # 合法——同步状态
        self.cell_centers[idx]   = new_ctr
        self.cell_rotations[idx] = rot
        self._update_occupancy(idx, new_ctr, rot)

        # 前进到下一个器件
        self.step_idx += 1
        done = (self.step_idx >= self.n_cells)

        reward = 0.0
        if done:
            wl   = self._compute_wirelength()
            nc   = self._compute_net_crossing()
            den  = self._compute_density(self.density_grid_N)
            cost = wl + self.nc_weight*nc + self.density_weight*den
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
                rel  = self.pin_rel_pos[(cname,pname)]
                abs_ = ctr + self._rotate_points(rel, rot)
                xs.append(abs_[0]); ys.append(abs_[1])
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
                rel  = self.pin_rel_pos[(cname,pname)]
                pins.append(tuple((ctr + self._rotate_points(rel, rot)).tolist()))
            if len(pins) < 2: continue
            src = pins[0]
            for sink in pins[1:]:
                segs.append((src, sink))

        cnt = 0
        for i in range(len(segs)):
            for j in range(i+1, len(segs)):
                if self._segments_cross(segs[i], segs[j]):
                    cnt += 1
        return cnt

    def _segments_cross(self, s1, s2) -> bool:
        def ccw(A,B,C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        A,B = s1; C,D = s2
        return (ccw(A,C,D) != ccw(B,C,D)) and (ccw(A,B,C) != ccw(A,B,D))

    def _compute_density(self, N: int) -> float:
        grid = np.zeros((N,N), dtype=float)
        unit = self.grid_size // N
        mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # 合并所有已放器件到 mask 中
        for idx in range(self.n_cells):
            pts = (self._rotate_points(self.rel_contours[idx],
                                       self.cell_rotations[idx])
                   + self.cell_centers[idx]).astype(np.int32)
            tmp = np.zeros_like(mask)
            cv2.fillPoly(tmp, [pts], 1)
            mask |= tmp

        # 统计每个小格占用像素数
        for i in range(N):
            x0,x1 = i*unit, (i+1)*unit
            for j in range(N):
                y0,y1 = j*unit, (j+1)*unit
                grid[i,j] = mask[x0:x1, y0:y1].sum()

        mu = grid.mean()
        return float(((grid - mu)**2).mean())

    def visualize(self):
        fig, ax = plt.subplots(figsize=(8,8))
        # 器件
        for idx, cell in enumerate(self.cells):
            pts = (self._rotate_points(self.rel_contours[idx],
                                       self.cell_rotations[idx])
                   + self.cell_centers[idx])
            ax.add_patch(plt.Polygon(pts, fill=None, edgecolor='r'))
            ax.text(*self.cell_centers[idx], cell['cellName'], ha='center')

        # 飞线
        for net in self.nets:
            path = []
            for p in net['pinList']:
                cname, pname = p['cellName'], p['pinName']
                i = next(i for i,c in enumerate(self.cells) if c['cellName']==cname)
                ctr, rot = self.cell_centers[i], self.cell_rotations[i]
                rel  = self.pin_rel_pos[(cname,pname)]
                path.append(ctr + self._rotate_points(rel, rot))
            path = np.array(path)
            if len(path) > 1:
                ax.plot(path[:,0], path[:,1], 'b-')

        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        plt.title("Final PCB Layout")
        plt.show()
